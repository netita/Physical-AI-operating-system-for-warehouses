"""
WarehouseGPT — Isaac Sim HTTP Bridge
=====================================
Paste this entire script into Isaac Sim's Script Editor and click Run.

What it does:
  - Every 100 ms reads world-frame positions of all prims whose names
    contain "forklift", "worker", "human", "amr", or "robot"
  - POSTs them to the WarehouseGPT digital twin at localhost:8003/inject
  - The BEV map at localhost:8003/state/bev updates in real time

To stop:  call  _bridge.stop()  in the Script Editor and Run again.

Coordinate mapping
------------------
Isaac Sim uses Z-up, metres.  The digital twin BEV renderer expects
X = horizontal (0–100 m), Y = depth (0–60 m).
If your stage uses a different scale or axis, adjust SCALE and AXIS_MAP below.
"""

import json
import time
import urllib.request
import urllib.error

import numpy as np
import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf

# ---------------------------------------------------------------------------
# Configuration — adjust to match your stage
# ---------------------------------------------------------------------------

DIGITAL_TWIN_URL = "http://localhost:8003/inject"
UPDATE_HZ = 10          # pushes per second (10 = 100 ms interval)

# Scale factor: if your stage is in centimetres set to 0.01, metres keep 1.0
SCALE = 1.0

# Which USD prim name substrings map to which agent type
AGENT_PATTERNS = {
    "forklift":  "forklift",
    "Forklift":  "forklift",
    "worker":    "worker",
    "Worker":    "worker",
    "human":     "worker",
    "Human":     "worker",
    "person":    "worker",
    "amr":       "amr",
    "AMR":       "amr",
    "robot":     "amr",
    "Robot":     "amr",
    "pallet":    "pallet",
    "Pallet":    "pallet",
    "box":       "pallet",
    "Box":       "pallet",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_world_xform(prim: "Usd.Prim"):
    """Return (translation_gf, rotation_quat_gf) in world frame."""
    xformable = UsdGeom.Xformable(prim)
    xform_mat: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation: Gf.Vec3d = xform_mat.ExtractTranslation()
    rotation: Gf.Rotation = xform_mat.ExtractRotationQuat()
    return translation, rotation


def _quat_to_yaw(quat: "Gf.Quatd") -> float:
    """Extract yaw (Z-axis rotation) from a quaternion."""
    img = quat.GetImaginary()
    real = quat.GetReal()
    return float(2.0 * np.arctan2(float(img[2]), float(real)))


def _post(payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DIGITAL_TWIN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.4):
            pass
    except urllib.error.URLError as exc:
        carb.log_warn(f"[WarehouseGPT] POST failed: {exc.reason}")
    except Exception as exc:
        carb.log_warn(f"[WarehouseGPT] POST error: {exc}")


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

class IsaacSimBridge:
    def __init__(self):
        self._sub = None
        self._frame = 0
        self._interval = 1.0 / UPDATE_HZ
        self._last_t = 0.0

    def start(self):
        self._sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="warehousegpt_bridge")
        )
        carb.log_info("[WarehouseGPT] Isaac Sim bridge started — pushing to " + DIGITAL_TWIN_URL)
        print("[WarehouseGPT] Bridge started. BEV map: http://localhost:8003/state/bev")

    def stop(self):
        self._sub = None
        carb.log_info("[WarehouseGPT] Isaac Sim bridge stopped.")
        print("[WarehouseGPT] Bridge stopped.")

    def _on_update(self, _event):
        now = time.monotonic()
        if now - self._last_t < self._interval:
            return
        self._last_t = now
        self._push()

    def _push(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        forklifts, workers, amrs, pallets = [], [], [], []

        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            if not prim.IsA(UsdGeom.Xformable):
                continue

            name = prim.GetName()
            agent_type = None
            for pattern, atype in AGENT_PATTERNS.items():
                if pattern in name:
                    agent_type = atype
                    break
            if agent_type is None:
                continue

            try:
                trans, rot = _get_world_xform(prim)
                x = float(trans[0]) * SCALE
                y = float(trans[1]) * SCALE
                z = float(trans[2]) * SCALE
                heading = _quat_to_yaw(rot)
            except Exception as exc:
                carb.log_warn(f"[WarehouseGPT] Skipping {prim.GetPath()}: {exc}")
                continue

            agent_id = f"{name[:12]}"

            if agent_type == "forklift":
                forklifts.append({
                    "agent_id": agent_id, "agent_type": "forklift",
                    "x": x, "y": y, "z": z, "heading_rad": heading,
                    "vx": 0.0, "vy": 0.0, "confidence": 1.0,
                })
            elif agent_type == "worker":
                workers.append({
                    "agent_id": agent_id, "agent_type": "worker",
                    "x": x, "y": y, "z": z, "heading_rad": heading,
                    "vx": 0.0, "vy": 0.0, "confidence": 1.0,
                })
            elif agent_type == "amr":
                amrs.append({
                    "agent_id": agent_id, "agent_type": "amr",
                    "x": x, "y": y, "z": z, "heading_rad": heading,
                    "vx": 0.0, "vy": 0.0, "confidence": 1.0,
                })
            elif agent_type == "pallet":
                pallets.append({
                    "item_id": agent_id, "barcode": "",
                    "x": x, "y": y, "z": z, "zone": "unknown",
                })

        self._frame += 1
        _post({
            "forklifts": forklifts,
            "workers":   workers,
            "amrs":      amrs,
            "pallets":   pallets,
            "incidents": [],
            "frame_index": self._frame,
        })


# ---------------------------------------------------------------------------
# Entry point — runs when you click Run in Script Editor
# ---------------------------------------------------------------------------

# Stop any previously running bridge first
if "_bridge" in dir():
    try:
        _bridge.stop()
    except Exception:
        pass

_bridge = IsaacSimBridge()
_bridge.start()
