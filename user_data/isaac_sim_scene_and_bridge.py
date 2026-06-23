"""
WarehouseGPT — Scene Setup + Live Bridge
=========================================
1. Creates warehouse agents (forklifts, workers, AMRs, pallets) in the stage
2. Animates them so they move around
3. Pushes their positions to the digital twin every 100 ms

Paste into Isaac Sim Script Editor and click Run.
Watch the BEV map update live at: http://localhost:8003/state/bev
"""

import json
import math
import time
import urllib.request
import urllib.error

import numpy as np
import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf

DIGITAL_TWIN_URL = "http://localhost:8003/inject"

# ---------------------------------------------------------------------------
# Step 1 — Build the warehouse scene
# ---------------------------------------------------------------------------

def create_agent(stage, path: str, x: float, y: float, z: float = 0.0):
    prim = stage.DefinePrim(path, "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    xform.AddRotateZOp().Set(0.0)
    # Add a visible cube so you can see it in the viewport
    cube_path = path + "/Shape"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(1.0)
    return prim

stage = omni.usd.get_context().get_stage()

# Clear old warehouse prims if re-running
for name in ["/Warehouse"]:
    if stage.GetPrimAtPath(name):
        stage.RemovePrim(name)

# Root
stage.DefinePrim("/Warehouse", "Xform")

# Forklifts
create_agent(stage, "/Warehouse/Forklift_01", x=15.0, y=30.0)
create_agent(stage, "/Warehouse/Forklift_02", x=45.0, y=12.0)
create_agent(stage, "/Warehouse/Forklift_03", x=75.0, y=48.0)

# Workers
create_agent(stage, "/Warehouse/Worker_01", x=20.0, y=20.0)
create_agent(stage, "/Warehouse/Worker_02", x=55.0, y=35.0)
create_agent(stage, "/Warehouse/Worker_03", x=30.0, y=50.0)
create_agent(stage, "/Warehouse/Worker_04", x=80.0, y=10.0)

# AMRs
create_agent(stage, "/Warehouse/AMR_01", x=60.0, y=25.0)
create_agent(stage, "/Warehouse/AMR_02", x=25.0, y=42.0)

# Pallets
create_agent(stage, "/Warehouse/Pallet_01", x=10.0, y=5.0)
create_agent(stage, "/Warehouse/Pallet_02", x=35.0, y=8.0)
create_agent(stage, "/Warehouse/Pallet_03", x=62.0, y=5.0)

print("[WarehouseGPT] Scene created: 3 forklifts, 4 workers, 2 AMRs, 3 pallets")

# ---------------------------------------------------------------------------
# Step 2 — Animation paths (simple circular / linear orbits)
# ---------------------------------------------------------------------------

ANIMATIONS = {
    "/Warehouse/Forklift_01": lambda t: (15.0 + 8.0 * math.cos(t * 0.3),  30.0 + 5.0 * math.sin(t * 0.3),  t * 0.3),
    "/Warehouse/Forklift_02": lambda t: (45.0 + 10.0 * math.cos(t * 0.2 + 1), 12.0 + 8.0 * math.sin(t * 0.2 + 1), t * 0.2),
    "/Warehouse/Forklift_03": lambda t: (75.0 - 6.0 * math.cos(t * 0.25), 48.0 + 4.0 * math.sin(t * 0.25), t * 0.25 + math.pi),
    "/Warehouse/Worker_01":   lambda t: (20.0 + 4.0 * math.cos(t * 0.5),  20.0 + 4.0 * math.sin(t * 0.5),  0.0),
    "/Warehouse/Worker_02":   lambda t: (55.0 + 3.0 * math.cos(t * 0.4 + 2), 35.0 + 3.0 * math.sin(t * 0.4 + 2), 0.0),
    "/Warehouse/Worker_03":   lambda t: (30.0 + 5.0 * math.sin(t * 0.35), 50.0,                               0.0),
    "/Warehouse/Worker_04":   lambda t: (80.0,                              10.0 + 6.0 * math.sin(t * 0.45),  0.0),
    "/Warehouse/AMR_01":      lambda t: (60.0 + 12.0 * math.cos(t * 0.6), 25.0 + 12.0 * math.sin(t * 0.6),  t * 0.6 + math.pi / 2),
    "/Warehouse/AMR_02":      lambda t: (25.0 + 10.0 * math.cos(t * 0.55 + 3), 42.0 + 10.0 * math.sin(t * 0.55 + 3), t * 0.55),
}

# ---------------------------------------------------------------------------
# Step 3 — HTTP push helper
# ---------------------------------------------------------------------------

def post_state(payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DIGITAL_TWIN_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.4):
            pass
    except Exception as exc:
        carb.log_warn(f"[WarehouseGPT] POST failed: {exc}")

# ---------------------------------------------------------------------------
# Step 4 — Update loop (runs every 100 ms on Isaac Sim's update event)
# ---------------------------------------------------------------------------

_start_time = time.monotonic()
_frame = 0
_last_push = 0.0
_sub = None

def _on_update(_event):
    global _frame, _last_push

    now = time.monotonic()
    if now - _last_push < 0.1:
        return
    _last_push = now

    t = now - _start_time
    _frame += 1

    forklifts, workers, amrs, pallets = [], [], [], []

    for path, anim_fn in ANIMATIONS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue

        x, y, heading = anim_fn(t)

        # Update prim position in the viewport
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(x, y, 0.0))
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                op.Set(math.degrees(heading))

        name = prim.GetName().lower()
        agent_id = prim.GetName()

        if "forklift" in name:
            forklifts.append({
                "agent_id": agent_id, "agent_type": "forklift",
                "x": x, "y": y, "z": 0.0, "heading_rad": heading,
                "vx": 0.0, "vy": 0.0, "confidence": 1.0,
            })
        elif "worker" in name:
            workers.append({
                "agent_id": agent_id, "agent_type": "worker",
                "x": x, "y": y, "z": 0.0, "heading_rad": heading,
                "vx": 0.0, "vy": 0.0, "confidence": 1.0,
            })
        elif "amr" in name:
            amrs.append({
                "agent_id": agent_id, "agent_type": "amr",
                "x": x, "y": y, "z": 0.0, "heading_rad": heading,
                "vx": 0.0, "vy": 0.0, "confidence": 1.0,
            })

    # Static pallets
    for path in ["/Warehouse/Pallet_01", "/Warehouse/Pallet_02", "/Warehouse/Pallet_03"]:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        pos = ops[0].Get() if ops else Gf.Vec3d(0, 0, 0)
        pallets.append({
            "item_id": prim.GetName(), "barcode": "",
            "x": float(pos[0]), "y": float(pos[1]), "z": 0.0, "zone": "A",
        })

    post_state({
        "forklifts": forklifts, "workers": workers,
        "amrs": amrs, "pallets": pallets,
        "incidents": [], "frame_index": _frame,
    })

# Stop previous subscription if re-running
if "_sub" in dir() and _sub is not None:
    _sub = None

_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_update, name="warehousegpt_bridge"
)

print("[WarehouseGPT] Scene + bridge running!")
print("[WarehouseGPT] Open BEV map: http://localhost:8003/state/bev")
print("[WarehouseGPT] Agents are moving — refresh the BEV map to see updates")
