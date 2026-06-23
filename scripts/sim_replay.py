"""
sim_replay.py — Standalone Isaac Sim mock for testing the Digital Twin
=======================================================================
Runs WITHOUT Isaac Sim.  Simulates moving agents in a 120 m × 60 m warehouse
and sends them to the Digital Twin /inject endpoint so you can verify the BEV
map, safety detectors, and incident stream without a real sim.

Usage
-----
  cd warehousegpt
  source .venv/bin/activate
  python scripts/sim_replay.py                     # default: normal ops
  python scripts/sim_replay.py --scenario near_miss
  python scripts/sim_replay.py --scenario zone_violation
  python scripts/sim_replay.py --scenario collision
  python scripts/sim_replay.py --hz 20             # faster update rate

Scenarios
---------
  normal          — agents move around normally, no incidents expected
  near_miss       — forklift closes to 1.5 m of a worker at 2.5 m/s
  zone_violation  — worker walks into the forklift charging bay (0,0)-(12,15)
  collision       — forklift and worker meet at the same point
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DT_API_URL = "http://localhost:8003/inject"

# Warehouse dimensions (matches warehouse_default.yaml)
W_LEN = 120.0   # metres X
W_WID =  60.0   # metres Y


# ---------------------------------------------------------------------------
# Agent simulation
# ---------------------------------------------------------------------------

@dataclass
class SimAgent:
    agent_id: str
    agent_type: str       # "forklift" | "worker" | "amr"
    x: float
    y: float
    heading: float        # radians, 0 = +X direction
    speed: float          # m/s
    # Optional scripted path: list of (x, y) waypoints
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    _wp_idx: int = 0

    @property
    def vx(self) -> float:
        return math.cos(self.heading) * self.speed

    @property
    def vy(self) -> float:
        return math.sin(self.heading) * self.speed

    def step(self, dt: float) -> None:
        """Update position; follow waypoints if set, else bounce off walls."""
        if self.waypoints:
            tx, ty = self.waypoints[self._wp_idx % len(self.waypoints)]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.5:
                self._wp_idx += 1
            else:
                self.heading = math.atan2(dy, dx)
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Bounce off warehouse walls
        if self.x < 0.5 or self.x > W_LEN - 0.5:
            self.heading = math.pi - self.heading
            self.x = max(0.5, min(W_LEN - 0.5, self.x))
        if self.y < 0.5 or self.y > W_WID - 0.5:
            self.heading = -self.heading
            self.y = max(0.5, min(W_WID - 0.5, self.y))

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_id":    self.agent_id,
            "agent_type":  self.agent_type,
            "x":           round(self.x, 3),
            "y":           round(self.y, 3),
            "z":           0.0,
            "heading_rad": round(self.heading, 4),
            "vx":          round(self.vx, 3),
            "vy":          round(self.vy, 3),
            "confidence":  1.0,
        }


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _build_normal() -> tuple[list[SimAgent], list[SimAgent], list[SimAgent]]:
    forklifts = [
        SimAgent("FL-01", "forklift", x=20.0, y=15.0, heading=0.0,      speed=1.8,
                 waypoints=[(90.0, 15.0), (90.0, 45.0), (20.0, 45.0), (20.0, 15.0)]),
        SimAgent("FL-02", "forklift", x=60.0, y=30.0, heading=math.pi,  speed=2.0,
                 waypoints=[(10.0, 30.0), (10.0, 10.0), (60.0, 10.0), (60.0, 30.0)]),
        SimAgent("FL-03", "forklift", x=90.0, y=50.0, heading=math.pi/2, speed=1.5,
                 waypoints=[(90.0, 10.0), (40.0, 10.0), (40.0, 50.0), (90.0, 50.0)]),
    ]
    workers = [
        SimAgent("WK-01", "worker", x=30.0, y=20.0, heading=0.3,  speed=1.1,
                 waypoints=[(50.0, 20.0), (50.0, 40.0), (30.0, 40.0), (30.0, 20.0)]),
        SimAgent("WK-02", "worker", x=70.0, y=35.0, heading=1.9,  speed=1.2,
                 waypoints=[(80.0, 35.0), (80.0, 55.0), (70.0, 55.0), (70.0, 35.0)]),
        SimAgent("WK-03", "worker", x=45.0, y=10.0, heading=0.0,  speed=1.0,
                 waypoints=[(110.0, 10.0), (110.0, 25.0), (45.0, 25.0), (45.0, 10.0)]),
        SimAgent("WK-04", "worker", x=15.0, y=50.0, heading=2.5,  speed=1.1,
                 waypoints=[(15.0, 25.0), (35.0, 25.0), (35.0, 50.0), (15.0, 50.0)]),
    ]
    amrs = [
        SimAgent("AMR-01", "amr", x=55.0, y=20.0, heading=math.pi/2, speed=1.5,
                 waypoints=[(55.0, 50.0), (75.0, 50.0), (75.0, 20.0), (55.0, 20.0)]),
        SimAgent("AMR-02", "amr", x=25.0, y=40.0, heading=0.0,       speed=1.4,
                 waypoints=[(95.0, 40.0), (95.0, 15.0), (25.0, 15.0), (25.0, 40.0)]),
    ]
    return forklifts, workers, amrs


def _build_near_miss() -> tuple[list[SimAgent], list[SimAgent], list[SimAgent]]:
    """Forklift FL-01 closes on WK-01 at 2.5 m/s from 12 m out, stops at 1.5 m."""
    forklifts = [
        SimAgent("FL-01", "forklift", x=30.0, y=30.0, heading=0.0, speed=2.5,
                 waypoints=[(60.0, 30.0), (30.0, 30.0)]),
        SimAgent("FL-02", "forklift", x=80.0, y=15.0, heading=math.pi, speed=1.5,
                 waypoints=[(20.0, 15.0), (80.0, 15.0)]),
    ]
    workers = [
        SimAgent("WK-01", "worker", x=43.0, y=30.0, heading=math.pi, speed=0.5,
                 waypoints=[(35.0, 30.0), (43.0, 30.0)]),
        SimAgent("WK-02", "worker", x=70.0, y=40.0, heading=0.0,     speed=1.1,
                 waypoints=[(100.0, 40.0), (70.0, 40.0)]),
    ]
    amrs = [
        SimAgent("AMR-01", "amr", x=55.0, y=50.0, heading=math.pi/2, speed=1.5,
                 waypoints=[(55.0, 10.0), (55.0, 50.0)]),
    ]
    return forklifts, workers, amrs


def _build_zone_violation() -> tuple[list[SimAgent], list[SimAgent], list[SimAgent]]:
    """WK-01 walks into the charging bay (0–12 m, 0–15 m)."""
    forklifts = [
        SimAgent("FL-01", "forklift", x=6.0, y=8.0, heading=0.0, speed=0.0),  # parked
        SimAgent("FL-02", "forklift", x=60.0, y=20.0, heading=0.0, speed=1.8,
                 waypoints=[(100.0, 20.0), (60.0, 20.0)]),
    ]
    workers = [
        SimAgent("WK-01", "worker", x=14.0, y=8.0, heading=math.pi, speed=1.1,
                 waypoints=[(2.0, 8.0), (14.0, 8.0)]),   # walks left into bay
        SimAgent("WK-02", "worker", x=50.0, y=40.0, heading=0.0, speed=1.2,
                 waypoints=[(90.0, 40.0), (50.0, 40.0)]),
    ]
    amrs = [
        SimAgent("AMR-01", "amr", x=70.0, y=30.0, heading=math.pi/2, speed=1.4,
                 waypoints=[(70.0, 55.0), (70.0, 30.0)]),
    ]
    return forklifts, workers, amrs


def _build_collision() -> tuple[list[SimAgent], list[SimAgent], list[SimAgent]]:
    """FL-01 and WK-01 converge on the same point at speed."""
    forklifts = [
        SimAgent("FL-01", "forklift", x=20.0, y=25.0, heading=0.0, speed=2.5,
                 waypoints=[(70.0, 25.0), (20.0, 25.0)]),
    ]
    workers = [
        SimAgent("WK-01", "worker", x=70.0, y=25.0, heading=math.pi, speed=1.3,
                 waypoints=[(20.0, 25.0), (70.0, 25.0)]),
        SimAgent("WK-02", "worker", x=50.0, y=40.0, heading=0.0, speed=1.1,
                 waypoints=[(90.0, 40.0), (50.0, 40.0)]),
    ]
    amrs = []
    return forklifts, workers, amrs


_SCENARIOS = {
    "normal":         _build_normal,
    "near_miss":      _build_near_miss,
    "zone_violation": _build_zone_violation,
    "collision":      _build_collision,
}


# ---------------------------------------------------------------------------
# HTTP POST
# ---------------------------------------------------------------------------

def _post(payload: dict) -> None:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DT_API_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            result = json.loads(resp.read())
            detected = result.get("safety_incidents_detected", 0)
            fl = len(payload["forklifts"])
            wk = len(payload["workers"])
            amr = len(payload["amrs"])
            frame = payload["frame_index"]
            mark = f"  ⚠ {detected} incident(s)" if detected else ""
            print(f"  frame {frame:>5}  FL={fl} WK={wk} AMR={amr}{mark}", flush=True)
    except Exception as exc:
        print(f"  POST error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def run(scenario: str, hz: float, duration: float | None) -> None:
    build_fn = _SCENARIOS[scenario]
    forklifts, workers, amrs = build_fn()

    all_agents = forklifts + workers + amrs
    dt = 1.0 / hz
    frame = 0
    t_start = time.perf_counter()

    print(f"\nWarehouseGPT sim-replay: scenario={scenario!r}  hz={hz}  "
          f"endpoint={DT_API_URL}")
    print(f"Agents: {len(forklifts)} forklifts, {len(workers)} workers, {len(amrs)} AMRs")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.perf_counter()

            # Step simulation
            for agent in all_agents:
                agent.step(dt)

            payload = {
                "forklifts":   [a.to_payload() for a in forklifts],
                "workers":     [a.to_payload() for a in workers],
                "amrs":        [a.to_payload() for a in amrs],
                "pallets":     [],
                "incidents":   [],
                "frame_index": frame,
            }
            _post(payload)
            frame += 1

            if duration and (time.perf_counter() - t_start) >= duration:
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nSent {frame} frames.")


def main() -> None:
    global DT_API_URL
    p = argparse.ArgumentParser(description="WarehouseGPT Digital Twin mock replay")
    p.add_argument(
        "--scenario",
        choices=list(_SCENARIOS),
        default="normal",
        help="Safety scenario to simulate (default: normal)",
    )
    p.add_argument("--hz",       type=float, default=10.0,  help="Inject frequency (default: 10 Hz)")
    p.add_argument("--duration", type=float, default=None,  help="Run for N seconds then exit (default: forever)")
    p.add_argument("--url",      type=str,   default=DT_API_URL, help="Digital Twin API URL")
    args = p.parse_args()

    DT_API_URL = args.url
    run(scenario=args.scenario, hz=args.hz, duration=args.duration)


if __name__ == "__main__":
    main()
