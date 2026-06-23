"""
Quick smoke-test for Isaac Sim 5.1 warehouse scene.

Loads the built-in NVIDIA warehouse stage, captures a few frames with the
CosmosWriter (RGB + depth + segmentation), and saves them to _out_test/.

Run from your isaacsim_5_1 install:

    ~/isaacsim_5_1/python.sh isaac_sim/test_warehouse_scene.py --headless --frames 4

Or with GUI:

    ~/isaacsim_5_1/python.sh isaac_sim/test_warehouse_scene.py --frames 4
"""

from isaacsim import SimulationApp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--frames", type=int, default=4)
parser.add_argument("--out-dir", type=str, default="_out_test")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp(launch_config={"headless": args.headless})

import os

import carb
import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom

# ── Stage ──────────────────────────────────────────────────────────────────────
WAREHOUSE_STAGE = "/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd"

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Cannot find Isaac Sim assets root — is Nucleus reachable?")
    simulation_app.close()
    raise SystemExit(1)

stage_url = assets_root + WAREHOUSE_STAGE
print(f"[test] Opening: {stage_url}")
omni.usd.get_context().open_stage(stage_url)
simulation_app.update()

# ── Replicator settings ────────────────────────────────────────────────────────
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)          # DLSS Quality
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
rep.orchestrator.set_capture_on_play(False)

# ── Camera: use the stage's existing perspective camera ───────────────────────
stage = omni.usd.get_context().get_stage()

# The warehouse stage has several cameras; use the viewport perspective one.
camera_path = "/OmniverseKit_Persp"

rp = rep.create.render_product(camera_path, (1280, 720))

# ── Writer ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.getcwd(), args.out_dir)
print(f"[test] Output → {out_dir}")

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=out_dir,
    rgb=True,
    distance_to_image_plane=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True,
)
writer.attach(rp)

# ── Playback ───────────────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

print(f"[test] Capturing {args.frames} frames …")
for i in range(args.frames):
    print(f"  frame {i + 1}/{args.frames}")
    rep.orchestrator.step(pause_timeline=False)

rep.orchestrator.wait_until_complete()
print(f"[test] Done — frames saved to {out_dir}/")

writer.detach()
rp.destroy()
timeline.pause()

simulation_app.close()
