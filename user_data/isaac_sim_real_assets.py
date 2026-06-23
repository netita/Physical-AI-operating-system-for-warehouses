"""
WarehouseGPT — Real Warehouse Scene + Live Bridge
===================================================
Step 1: Clears ALL old prims from previous scripts
Step 2: Loads NVIDIA real warehouse + forklift + Nova Carter + worker assets
Step 3: Animates them and streams positions to the digital twin

Paste into Isaac Sim Script Editor and click Run.
Assets download from NVIDIA CDN on first run (~30-60 s).

Live BEV: http://localhost:8003/bev/live
"""

import json, math, time
import urllib.request, urllib.error
import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf

DIGITAL_TWIN_URL = "http://localhost:8003/inject"

# ---------------------------------------------------------------------------
# Step 0 — Stop any old subscription and clear the stage
# ---------------------------------------------------------------------------

# Stop previous bridge subscription if this script is re-run
if "_wgpt_sub" in dir():
    try:
        _wgpt_sub = None   # release the subscription object
    except Exception:
        pass

stage = omni.usd.get_context().get_stage()

# Remove ALL old WarehouseGPT prims
for old_root in ["/Warehouse", "/WarehouseGPT", "/World/Warehouse"]:
    p = stage.GetPrimAtPath(old_root)
    if p and p.IsValid():
        stage.RemovePrim(old_root)
        print(f"[WarehouseGPT] Removed old prim: {old_root}")

# ---------------------------------------------------------------------------
# Step 1 — Resolve NVIDIA asset root (Nucleus or CDN fallback)
# ---------------------------------------------------------------------------

try:
    from isaacsim.core.utils.nucleus import get_assets_root_path
    ASSETS_ROOT = get_assets_root_path()
    if not ASSETS_ROOT:
        raise RuntimeError("nucleus returned None")
    print(f"[WarehouseGPT] Asset root: {ASSETS_ROOT}")
except Exception as e:
    ASSETS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    print(f"[WarehouseGPT] Nucleus unavailable ({e}), using CDN: {ASSETS_ROOT}")

WAREHOUSE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
FORKLIFT_USD  = f"{ASSETS_ROOT}/Isaac/Props/Forklift/forklift.usd"
AMR_USD       = f"{ASSETS_ROOT}/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
WORKER_USD    = f"{ASSETS_ROOT}/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd"

print(f"[WarehouseGPT] Warehouse: {WAREHOUSE_USD}")

# ---------------------------------------------------------------------------
# Step 2 — Build the scene
# ---------------------------------------------------------------------------

root = stage.DefinePrim("/WarehouseGPT", "Xform")

# Warehouse environment — reference directly on the wrapper (no extra ops needed)
env = stage.DefinePrim("/WarehouseGPT/Environment", "Xform")
env.GetReferences().AddReference(WAREHOUSE_USD)

def add_agent(path, usd_ref, x, y, z=0.0, rot_z=0.0, scale=1.0):
    # Wrapper Xform owns our translate/rotate — avoids conflict with asset's own ops
    wrapper = stage.DefinePrim(path, "Xform")
    xf = UsdGeom.Xformable(wrapper)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    xf.AddRotateZOp().Set(rot_z)
    if scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    # Reference the USD asset as a child so its own ops don't conflict
    model = stage.DefinePrim(f"{path}/Model", "Xform")
    model.GetReferences().AddReference(usd_ref)
    return wrapper

# Forklifts — forklift.usd is authored in centimetres, scale to metres
add_agent("/WarehouseGPT/Forklift_01", FORKLIFT_USD,  4.0, -4.0, scale=0.01)
add_agent("/WarehouseGPT/Forklift_02", FORKLIFT_USD, 12.0, -4.0, scale=0.01)
add_agent("/WarehouseGPT/Forklift_03", FORKLIFT_USD, 20.0, -4.0, scale=0.01)

# AMRs — Nova Carter
add_agent("/WarehouseGPT/AMR_01", AMR_USD,  8.0, 2.0)
add_agent("/WarehouseGPT/AMR_02", AMR_USD, 16.0, 2.0)

# Workers
add_agent("/WarehouseGPT/Worker_01", WORKER_USD,  2.0, 4.0)
add_agent("/WarehouseGPT/Worker_02", WORKER_USD,  6.0, 4.0)
add_agent("/WarehouseGPT/Worker_03", WORKER_USD, 14.0, 4.0)
add_agent("/WarehouseGPT/Worker_04", WORKER_USD, 22.0, 4.0)

print("[WarehouseGPT] Scene prims created (3 forklifts, 2 AMRs, 4 workers)")
print("[WarehouseGPT] Assets loading in viewport — may take 30-60 s on first run")

# ---------------------------------------------------------------------------
# Step 3 — Animation + bridge
# ---------------------------------------------------------------------------

_start_t = time.monotonic()
_frame   = 0
_last_t  = 0.0

ANIM = {
    "/WarehouseGPT/Forklift_01": lambda t: ( 4.0 + 6.0*math.cos(t*0.25),       -4.0 + 3.0*math.sin(t*0.25),      t*0.25),
    "/WarehouseGPT/Forklift_02": lambda t: (12.0 + 7.0*math.cos(t*0.2+2.0),    -4.0 + 4.0*math.sin(t*0.2+2.0),   t*0.2),
    "/WarehouseGPT/Forklift_03": lambda t: (20.0 + 5.0*math.cos(t*0.3+1.0),    -4.0 + 2.0*math.sin(t*0.3+1.0),   t*0.3+math.pi),
    "/WarehouseGPT/AMR_01":      lambda t: ( 8.0 + 8.0*math.cos(t*0.5),          2.0 + 8.0*math.sin(t*0.5),        t*0.5+math.pi/2),
    "/WarehouseGPT/AMR_02":      lambda t: (16.0 + 6.0*math.cos(t*0.45+3.0),     2.0 + 6.0*math.sin(t*0.45+3.0),   t*0.45),
    "/WarehouseGPT/Worker_01":   lambda t: ( 2.0 + 2.0*math.cos(t*0.6),           4.0 + 2.0*math.sin(t*0.6),        0.0),
    "/WarehouseGPT/Worker_02":   lambda t: ( 6.0 + 3.0*math.sin(t*0.4),           4.0,                               0.0),
    "/WarehouseGPT/Worker_03":   lambda t: (14.0 + 2.5*math.cos(t*0.5+1.0),       4.0 + 2.5*math.sin(t*0.5+1.0),    0.0),
    "/WarehouseGPT/Worker_04":   lambda t: (22.0,                                  4.0 + 3.0*math.sin(t*0.55),        0.0),
}

TYPES = {"Forklift": "forklift", "AMR": "amr", "Worker": "worker"}

def _post(payload):
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            DIGITAL_TWIN_URL, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=0.4):
            pass
    except Exception as exc:
        carb.log_warn(f"[WarehouseGPT] {exc}")

def _on_update(_event):
    global _frame, _last_t
    now = time.monotonic()
    if now - _last_t < 0.1:
        return
    _last_t = now
    t = now - _start_t
    _frame += 1

    forklifts, workers, amrs = [], [], []

    for path, fn in ANIM.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        x, y, h = fn(t)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(x, y, 0.0))
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                op.Set(math.degrees(h))

        name = prim.GetName()
        atype = next((v for k, v in TYPES.items() if name.startswith(k)), None)
        if not atype:
            continue
        entry = {"agent_id": name, "agent_type": atype,
                 "x": round(x,3), "y": round(y,3), "z": 0.0,
                 "heading_rad": round(h,4), "vx": 0.0, "vy": 0.0, "confidence": 1.0}
        {"forklift": forklifts, "amr": amrs, "worker": workers}[atype].append(entry)

    _post({"forklifts": forklifts, "workers": workers, "amrs": amrs,
           "pallets": [], "incidents": [], "frame_index": _frame})

_wgpt_sub = (omni.kit.app.get_app()
             .get_update_event_stream()
             .create_subscription_to_pop(_on_update, name="wgpt_real"))

print("[WarehouseGPT] Bridge running! Open: http://localhost:8003/bev/live")
print("[WarehouseGPT] To stop: _wgpt_sub = None")
