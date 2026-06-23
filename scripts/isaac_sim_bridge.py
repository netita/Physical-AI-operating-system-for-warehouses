# isaac_sim_bridge.py
# Open via Script Editor folder icon, then Run.
# Uses the Isaac Sim update loop (main thread) for USD ops - no freeze.
# To stop: bridge.stop()
#
# Scene: loads the official NVIDIA Isaac Warehouse automatically if the
# stage is empty. You can also pre-load the scene manually before running.
import json
import math
import threading
import time
import urllib.request
import omni.usd
import omni.kit.app
from pxr import Gf, UsdGeom
from isaacsim.core.utils.stage import add_reference_to_stage

DT_API_URL        = "http://localhost:8003/inject"
DT_CAMERA_URL     = "http://localhost:8003/inject/camera"
INJECT_HZ         = 10.0
CAMERA_EVERY_N    = 10   # post a camera frame every N inject ticks (~1 Hz at 10 Hz inject rate)

# Isaac Warehouse scene dimensions (metres)
W_LEN, W_WID = 50.0, 50.0

_WAREHOUSE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Stages/IsaacWarehouse.usd"
_FORKLIFT  = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/IsaacSim/ForkliftB/forklift_b_sensor.usd"
_AMR       = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Clearpath/Dingo/dingo.usd"

# Agent specs: (prim_path, asset_url, agent_id, agent_type, x, y, heading_rad, speed_m_s, waypoints)
SPECS = [
    ("/World/Forklifts/Forklift_01", _FORKLIFT, "Forklift_01", "forklift", 8.0,  25.0, 0.0,      2.5, [(40.0,25.0),(8.0,25.0)]),
    ("/World/AMRs/AMR_01",           _AMR,      "AMR_01",      "worker",   30.0, 22.0, math.pi,  0.5, [(15.0,22.0),(30.0,22.0)]),
    ("/World/AMRs/AMR_02",           _AMR,      "AMR_02",      "amr",      20.0, 10.0, 0.0,      1.5, [(40.0,10.0),(20.0,10.0)]),
]

def _stage():
    return omni.usd.get_context().get_stage()

class _Agent:
    def __init__(self, aid, atype, path, x, y, hdg, spd, wps):
        self.aid=aid; self.atype=atype; self.path=path
        self.x=x; self.y=y; self.hdg=hdg; self.spd=spd
        self.wps=wps; self._wpi=0; self._xop=None; self._rop=None
    def vx(self): return math.cos(self.hdg)*self.spd
    def vy(self): return math.sin(self.hdg)*self.spd
    def step(self, dt):
        if self.wps:
            tx,ty=self.wps[self._wpi % len(self.wps)]
            dx,dy=tx-self.x, ty-self.y
            if math.hypot(dx,dy)<0.5: self._wpi+=1
            else: self.hdg=math.atan2(dy,dx)
        self.x=max(1.0,min(W_LEN-1,self.x+self.vx()*dt))
        self.y=max(1.0,min(W_WID-1,self.y+self.vy()*dt))
        s=_stage(); p=s.GetPrimAtPath(self.path)
        if p and p.IsValid():
            xf=UsdGeom.Xformable(p)
            if self._xop is None:
                xf.ClearXformOpOrder()
                self._xop=xf.AddTranslateOp()
                self._rop=xf.AddRotateZOp()
            self._xop.Set(Gf.Vec3d(self.x, self.y, 0.0))
            self._rop.Set(math.degrees(self.hdg))
    def to_dict(self):
        return {"agent_id":self.aid,"agent_type":self.atype,
                "x":round(self.x,3),"y":round(self.y,3),"z":0.0,
                "heading_rad":round(self.hdg,4),
                "vx":round(self.vx(),3),"vy":round(self.vy(),3),"confidence":1.0}

class WarehouseBridge:
    def __init__(self):
        self._agents = []
        self._frame  = 0
        self._t_last = time.perf_counter()
        self._spawned = False
        self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="warehouse_dt_bridge"
        )
        print("[bridge] started - update loop hooked")

    def _spawn(self):
        s = _stage()
        for sc in ["/World/Forklifts", "/World/AMRs"]:
            if not s.GetPrimAtPath(sc).IsValid():
                s.DefinePrim(sc, "Scope")
        agents = []
        for pp, asset, aid, atype, x, y, hdg, spd, wps in SPECS:
            prim = add_reference_to_stage(usd_path=asset, prim_path=pp)
            if prim is not None:
                ag = _Agent(aid, atype, pp, x, y, hdg, spd, wps)
                agents.append(ag)
                print("[bridge] spawned " + atype + " " + aid + " from " + asset.split("/")[-1])
            else:
                print("[bridge] FAILED " + asset)
        return agents

    def _on_update(self, event):
        now = time.perf_counter()
        dt  = now - self._t_last

        if not self._spawned:
            s = _stage()
            if s is None:
                return
            prim_count = len(list(s.Traverse()))
            # Stage is empty — load the Isaac Warehouse scene first
            if prim_count <= 3:
                print("[bridge] stage empty — loading IsaacWarehouse.usd ...")
                omni.usd.get_context().open_stage(_WAREHOUSE)
                return  # wait for next update tick; stage will be populated
            self._agents = self._spawn()
            self._spawned = True
            print("[bridge] streaming to " + DT_API_URL)
            print("[bridge] open http://localhost:8003/bev/live")
            return

        if dt < 1.0 / INJECT_HZ:
            return
        self._t_last = now

        for ag in self._agents:
            ag.step(dt)

        fl = [a.to_dict() for a in self._agents if a.atype=="forklift"]
        wk = [a.to_dict() for a in self._agents if a.atype=="worker"]
        am = [a.to_dict() for a in self._agents if a.atype=="amr"]

        payload = {"forklifts":fl,"workers":wk,"amrs":am,
                   "pallets":[],"incidents":[],"frame_index":self._frame}
        self._frame += 1

        threading.Thread(target=_post, args=(payload,), daemon=True).start()

        # Every CAMERA_EVERY_N frames capture the viewport and post to /inject/camera
        if self._frame % CAMERA_EVERY_N == 0:
            jpeg = _capture_viewport_jpeg()
            if jpeg:
                threading.Thread(
                    target=_post_camera,
                    args=(jpeg, self._frame),
                    daemon=True,
                ).start()

    def stop(self):
        self._sub = None
        print("[bridge] stopped")

def _capture_viewport_jpeg():
    """Grab the active Isaac Sim viewport as a JPEG byte string."""
    try:
        import omni.kit.viewport.utility as vp_util
        import omni.renderer_capture
        import tempfile, os
        vp = vp_util.get_active_viewport()
        if vp is None:
            return None
        cap = omni.renderer_capture.acquire_renderer_capture_interface()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        cap.capture_next_frame_swapchain(path)
        omni.kit.app.get_app().update()  # flush capture
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
            return data
        return None
    except Exception as e:
        print("[bridge] viewport capture err " + str(e))
        return None


def _post(payload):
    try:
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(DT_API_URL, data=body,
               headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            d  = json.loads(r.read())
            n  = d.get("safety_incidents_detected", 0)
            fi = payload["frame_index"]
            if n:
                print("[bridge] frame " + str(fi) + " *** " + str(n) + " INCIDENT(S) ***")
            elif fi % 50 == 0:
                print("[bridge] frame " + str(fi) + " agents=" + str(d.get("agents",0)))
    except Exception as e:
        print("[bridge] err " + str(e))


def _post_camera(jpeg_bytes, frame_index, camera_id="cam_0"):
    """POST a JPEG frame to /inject/camera for fire detection."""
    try:
        boundary = b"----BridgeBoundary"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="frame"; filename="frame.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="camera_id"\r\n\r\n'
            + camera_id.encode() + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="frame_index"\r\n\r\n'
            + str(frame_index).encode() + b"\r\n"
            b"--" + boundary + b"--\r\n"
        )
        req = urllib.request.Request(
            DT_CAMERA_URL, data=body,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary.decode()},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            d = json.loads(r.read())
            if d.get("fires_detected", 0):
                print("[bridge] FIRE DETECTED frame=" + str(frame_index)
                      + " fires=" + str(d["fires_detected"]))
    except Exception as e:
        print("[bridge] camera err " + str(e))

bridge = WarehouseBridge()
print("[bridge] waiting for stage to be ready...")
