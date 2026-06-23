"""
isaac_headless_bridge.py — headless Digital Twin bridge for Isaac Sim
=====================================================================
Run with Isaac Sim's own python.sh:

    ~/isaacsim_5_1/python.sh scripts/isaac_headless_bridge.py \\
        --usd /home/aneta/aspasova@172.20.20.234/Warehouse_95x108x15_racks_V14.usd

    ~/isaacsim_5_1/python.sh scripts/isaac_headless_bridge.py \\
        --usd /home/aneta/aspasova@172.20.20.234/Warehouse_95x108x15_racks_V14.usd \\
        --scenario near_miss

Arguments
---------
  --usd PATH          Warehouse USD (local path or omniverse:// URL).
                      Omit for demo mode (no stage loaded).
  --url URL           Digital Twin API endpoint (default: http://localhost:8003/inject)
  --hz  FLOAT         Inject frequency in Hz (default: 10)
  --duration SECONDS  Run for N seconds then exit (default: forever)
  --scenario NAME     normal | near_miss (default: normal)

What it does
------------
1. Boots Isaac Sim headless
2. Loads your warehouse USD stage
3. If the stage has no robot prims, spawns Roby forklifts + Carter AMRs as
   USD references and animates them via xform manipulation (kinematic, no
   full physics articulation needed)
4. Streams agent poses to the Digital Twin /inject endpoint at --hz
5. Safety detectors fire on the Digital Twin side in real-time
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# SimulationApp MUST be first — before any omni.* imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--usd",      type=str,   default="")
parser.add_argument("--url",      type=str,   default="http://localhost:8003/inject")
parser.add_argument("--hz",       type=float, default=10.0)
parser.add_argument("--duration", type=float, default=None)
parser.add_argument("--scenario", type=str,   default="normal",
                    choices=["normal", "near_miss"])
parser.add_argument("--help", "-h", action="store_true")
_args, _ = parser.parse_known_args()

if _args.help:
    print(__doc__)
    sys.exit(0)

from isaacsim import SimulationApp  # noqa: E402
_sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import omni.usd                                              # noqa: E402
from isaacsim.core.utils.stage import open_stage             # noqa: E402
from pxr import Gf, Sdf, UsdGeom, Usd                       # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DT_API_URL   = _args.url
INJECT_HZ    = _args.hz
USD_PATH     = _args.usd
SCENARIO     = _args.scenario

# Real warehouse dimensions from the V14 USD filename (95 m × 108 m)
W_LEN, W_WID = 95.0, 108.0

# Robot asset paths (local installs)
_ASSETS_DIR = "/home/aneta/aspasova@172.20.20.234"
_ROBY_USD   = f"{_ASSETS_DIR}/Roby_04_04.usd"       # forklift: root=/mock_robot
_CARTER_USD = f"{_ASSETS_DIR}/aqita_carter_4.0_simple.usd"  # AMR: root=/vehicle


# ---------------------------------------------------------------------------
# Kinematic agent — animates via USD xform, no physics needed
# ---------------------------------------------------------------------------

class KinematicAgent:
    """
    An agent whose position is controlled directly by setting xform translations.
    Waypoints are given in world metres (X, Y floor plane).
    Isaac Sim uses Y-up: floor plane = X-Z, so we map DT-Y → USD-Z.
    """

    def __init__(self, aid: str, atype: str, prim_path: str,
                 x: float, y: float, heading: float, speed: float,
                 waypoints: list[tuple[float, float]]) -> None:
        self.aid     = aid
        self.atype   = atype
        self.path    = prim_path
        self.x       = x
        self.y       = y
        self.heading = heading
        self.speed   = speed
        self.waypoints = waypoints
        self._wpi    = 0
        self._xop    = None   # cached translate xform op

    @property
    def vx(self) -> float:
        return math.cos(self.heading) * self.speed

    @property
    def vy(self) -> float:
        return math.sin(self.heading) * self.speed

    def step(self, dt: float) -> None:
        if self.waypoints:
            tx, ty = self.waypoints[self._wpi % len(self.waypoints)]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.5:
                self._wpi += 1
            else:
                self.heading = math.atan2(dy, dx)
        self.x = max(1.0, min(W_LEN - 1.0, self.x + self.vx * dt))
        self.y = max(1.0, min(W_WID - 1.0, self.y + self.vy * dt))
        self._write_xform()

    def _write_xform(self) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.path)
        if not prim or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        if self._xop is None:
            xf.ClearXformOpOrder()
            self._xop = xf.AddTranslateOp()
        # DT floor plane: X=east, Y=north → Isaac Y-up: X=east, Z=north, Y=height
        self._xop.Set(Gf.Vec3d(self.x, 0.0, self.y))

    def to_dict(self) -> dict:
        return {
            "agent_id":    self.aid,
            "agent_type":  self.atype,
            "x":           round(self.x, 3),
            "y":           round(self.y, 3),
            "z":           0.0,
            "heading_rad": round(self.heading, 4),
            "vx":          round(self.vx, 3),
            "vy":          round(self.vy, 3),
            "confidence":  1.0,
        }


# ---------------------------------------------------------------------------
# Spawn agents into the loaded stage
# ---------------------------------------------------------------------------

def _spawn_agent(prim_path: str, asset_usd: str) -> bool:
    """Add asset_usd as a USD reference at prim_path. Returns True on success."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_usd)
    return prim.IsValid()


def _spawn_all(scenario: str) -> tuple[list[KinematicAgent], list[KinematicAgent], list[KinematicAgent]]:
    """Spawn robot USD references and return KinematicAgent wrappers."""
    stage = omni.usd.get_context().get_stage()

    # Define parent scopes
    for scope_path in ["/World/Forklifts", "/World/AMRs"]:
        if not stage.GetPrimAtPath(scope_path).IsValid():
            stage.DefinePrim(scope_path, "Scope")

    if scenario == "near_miss":
        agent_specs = [
            # (path, asset, aid, atype, x, y, heading, speed, waypoints)
            ("/World/Forklifts/Forklift_01", _ROBY_USD,
             "Forklift_01", "forklift", 20.0, 54.0, 0.0, 2.5,
             [(65.0, 54.0), (20.0, 54.0)]),
            ("/World/AMRs/AMR_01", _CARTER_USD,
             "AMR_01", "worker", 55.0, 54.0, math.pi, 0.5,
             [(40.0, 54.0), (55.0, 54.0)]),
            ("/World/AMRs/AMR_02", _CARTER_USD,
             "AMR_02", "amr", 40.0, 25.0, 0.0, 1.5,
             [(80.0, 25.0), (40.0, 25.0)]),
        ]
    else:  # normal
        agent_specs = [
            ("/World/Forklifts/Forklift_01", _ROBY_USD,
             "Forklift_01", "forklift", 15.0, 20.0, 0.0, 1.8,
             [(80.0, 20.0), (80.0, 80.0), (15.0, 80.0), (15.0, 20.0)]),
            ("/World/Forklifts/Forklift_02", _ROBY_USD,
             "Forklift_02", "forklift", 50.0, 50.0, math.pi, 2.0,
             [(10.0, 50.0), (10.0, 20.0), (50.0, 20.0), (50.0, 50.0)]),
            ("/World/AMRs/AMR_01", _CARTER_USD,
             "AMR_01", "amr", 60.0, 30.0, math.pi / 2, 1.5,
             [(60.0, 90.0), (80.0, 90.0), (80.0, 30.0), (60.0, 30.0)]),
            ("/World/AMRs/AMR_02", _CARTER_USD,
             "AMR_02", "worker", 30.0, 60.0, 0.0, 1.1,
             [(70.0, 60.0), (70.0, 40.0), (30.0, 40.0), (30.0, 60.0)]),
            ("/World/AMRs/AMR_03", _CARTER_USD,
             "AMR_03", "worker", 45.0, 15.0, 0.5, 1.2,
             [(85.0, 15.0), (85.0, 35.0), (45.0, 35.0), (45.0, 15.0)]),
        ]

    forklifts, workers, amrs = [], [], []
    for prim_path, asset, aid, atype, x, y, heading, speed, wps in agent_specs:
        ok = _spawn_agent(prim_path, asset)
        if ok:
            agent = KinematicAgent(aid, atype, prim_path, x, y, heading, speed, wps)
            agent._write_xform()   # set initial position
            if atype == "forklift":
                forklifts.append(agent)
            elif atype == "worker":
                workers.append(agent)
            else:
                amrs.append(agent)
            print(f"[bridge] Spawned {atype} '{aid}' at ({x:.0f}, {y:.0f}) → {prim_path}")
        else:
            print(f"[bridge] WARNING: failed to spawn {prim_path}")

    return forklifts, workers, amrs


# ---------------------------------------------------------------------------
# HTTP POST
# ---------------------------------------------------------------------------

def _post(payload: dict) -> None:
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            DT_API_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            r = json.loads(resp.read())
            n = r.get("safety_incidents_detected", 0)
            mark = f"  ⚠ {n} incident(s)" if n else ""
            print(f"  frame {payload['frame_index']:>5}  agents={r.get('agents', 0)}{mark}",
                  flush=True)
    except Exception as exc:
        print(f"  POST error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if USD_PATH:
        print(f"[bridge] Loading warehouse: {USD_PATH}")
        open_stage(USD_PATH)
        print(f"[bridge] Spawning robots (scenario={SCENARIO!r}) …")
        forklifts, workers, amrs = _spawn_all(SCENARIO)
        print(f"[bridge] Ready: {len(forklifts)} forklifts, "
              f"{len(workers)} workers, {len(amrs)} AMRs")
    else:
        print("[bridge] No --usd supplied — pure demo mode (no stage)")
        from scripts.sim_replay import _build_near_miss, _build_normal
        build = _build_near_miss if SCENARIO == "near_miss" else _build_normal
        _fl, _wk, _amr = build()
        forklifts, workers, amrs = list(_fl), list(_wk), list(_amr)

    dt      = 1.0 / INJECT_HZ
    frame   = 0
    t_start = time.perf_counter()
    use_usd = bool(USD_PATH)

    print(f"\n[bridge] Streaming to {DT_API_URL} at {INJECT_HZ} Hz")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if _args.duration and (time.perf_counter() - t_start) >= _args.duration:
                break
            t0 = time.perf_counter()

            if use_usd:
                for a in forklifts + workers + amrs:
                    a.step(dt)
                fl_data = [a.to_dict() for a in forklifts]
                wk_data = [a.to_dict() for a in workers]
                amr_data = [a.to_dict() for a in amrs]
            else:
                for a in forklifts + workers + amrs:
                    a.step(dt)
                fl_data  = [a.to_payload() for a in forklifts]
                wk_data  = [a.to_payload() for a in workers]
                amr_data = [a.to_payload() for a in amrs]

            _post({
                "forklifts":   fl_data,
                "workers":     wk_data,
                "amrs":        amr_data,
                "pallets":     [],
                "incidents":   [],
                "frame_index": frame,
            })
            frame += 1

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\n[bridge] Stopped.")
    finally:
        _sim_app.close()

    print(f"\n[bridge] Sent {frame} frames.")


if __name__ == "__main__":
    main()
