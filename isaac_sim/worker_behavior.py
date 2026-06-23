"""
Worker Behavior — NVIDIA Isaac Sim Character Animation

Manages animated humanoid workers in the warehouse scene. Each worker has an
animation state machine with three tasks: picking, walking, and loading. Pose
data is returned as SE(3) transforms for downstream CV model training.

Usage:
    from isaac_sim.worker_behavior import WorkerBehavior
    wb = WorkerBehavior(stage)
    wid = wb.spawn_worker((10.0, 20.0, 0.0))
    wb.set_task(wid, "picking")
    wb.simulate_step(dt=1/30)
    poses = wb.get_worker_poses()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    get_current_stage,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt


# ---------------------------------------------------------------------------
# Worker task types
# ---------------------------------------------------------------------------

WorkerTask = Literal["picking", "walking", "loading", "idle"]


# ---------------------------------------------------------------------------
# Animation clip paths on Nucleus
# ---------------------------------------------------------------------------

_ANIM_CLIPS: Dict[str, str] = {
    "idle":    "omniverse://localhost/NVIDIA/Assets/Characters/Animations/Idle_A.usd",
    "walking": "omniverse://localhost/NVIDIA/Assets/Characters/Animations/Walk_A.usd",
    "picking": "omniverse://localhost/NVIDIA/Assets/Characters/Animations/PickUp_A.usd",
    "loading": "omniverse://localhost/NVIDIA/Assets/Characters/Animations/Push_Cart_A.usd",
}

_WORKER_ASSETS: List[str] = [
    "omniverse://localhost/NVIDIA/Assets/Characters/Worker_Male_A/Worker_Male_A.usd",
    "omniverse://localhost/NVIDIA/Assets/Characters/Worker_Female_A/Worker_Female_A.usd",
    "omniverse://localhost/NVIDIA/Assets/Characters/Worker_Male_B/Worker_Male_B.usd",
    "omniverse://localhost/NVIDIA/Assets/Characters/Worker_Female_B/Worker_Female_B.usd",
]

# Speed (m/s) while walking
# Prim path to the animation prim inside NVIDIA animation USD files
_ANIM_CLIP_PRIM_PATH = "/Root/Anim"

_WALK_SPEED = 1.2
# Picking cycle duration (seconds)
_PICK_CYCLE_DURATION = 3.5
# Loading push cadence (seconds per push)
_LOAD_CYCLE_DURATION = 2.0


# ---------------------------------------------------------------------------
# Internal state per worker
# ---------------------------------------------------------------------------

@dataclass
class _WorkerState:
    worker_id: int
    prim_path: str
    task: WorkerTask = "idle"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spawn_z: float = 0.0              # original Z so idle sway oscillates around it
    heading_rad: float = 0.0          # yaw in radians
    waypoints: List[np.ndarray] = field(default_factory=list)
    task_timer: float = 0.0           # time elapsed in current task phase
    phase: int = 0                    # sub-phase index within state machine
    anim_time: float = 0.0            # playback position in current clip (s)


# ---------------------------------------------------------------------------
# WorkerBehavior
# ---------------------------------------------------------------------------

class WorkerBehavior:
    """
    Manages a pool of animated workers in the warehouse stage.

    The state machine supports four tasks:
      - ``idle``    : stand in place, breathing animation
      - ``walking`` : translate along waypoints at _WALK_SPEED
      - ``picking`` : multi-phase reach-grasp-lift-place cycle
      - ``loading`` : push-cart translation with torso lean

    Parameters
    ----------
    stage : Usd.Stage, optional
        Active stage. Uses current stage if None.
    workers_scope : str
        USD path where worker prims are created.
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        workers_scope: str = "/World/Actors/Workers",
    ) -> None:
        self._stage = stage or get_current_stage()
        self._scope = workers_scope
        self._workers: Dict[int, _WorkerState] = {}
        self._next_id = 0
        self._rng = random.Random(0)

        self._ensure_scope()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn_worker(
        self,
        position: Tuple[float, float, float],
        asset_path: Optional[str] = None,
        heading_deg: float = 0.0,
        task: WorkerTask = "idle",
        seed: Optional[int] = None,
    ) -> int:
        """
        Instantiate an animated humanoid worker at the given position.

        Parameters
        ----------
        position : tuple(x, y, z)
            World-space spawn position.
        asset_path : str, optional
            Nucleus path to character USD. Random variant chosen if None.
        heading_deg : float
            Initial heading in degrees (0 = +X direction).
        task : WorkerTask
            Initial task to assign.
        seed : int, optional
            RNG seed for asset selection.

        Returns
        -------
        int
            Unique worker ID.
        """
        rng = random.Random(seed) if seed is not None else self._rng
        if asset_path is None:
            asset_path = rng.choice(_WORKER_ASSETS)

        wid = self._next_id
        self._next_id += 1
        prim_path = f"{self._scope}/Worker_{wid:04d}"

        # Reference character USD
        ref_prim = add_reference_to_stage(
            usd_path=asset_path, prim_path=prim_path
        )
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            xf.AddRotateZOp().Set(heading_deg)

        state = _WorkerState(
            worker_id=wid,
            prim_path=prim_path,
            task=task,
            position=np.array(position, dtype=np.float64),
            spawn_z=float(position[2]),
            heading_rad=math.radians(heading_deg),
        )
        self._workers[wid] = state
        self._apply_animation(state)
        return wid

    def set_task(
        self,
        worker_id: int,
        task: WorkerTask,
        waypoints: Optional[List[Tuple[float, float, float]]] = None,
    ) -> None:
        """
        Assign a new task to a worker and reset the state machine.

        Parameters
        ----------
        worker_id : int
            ID returned by spawn_worker().
        task : WorkerTask
            ``"picking"`` | ``"walking"`` | ``"loading"`` | ``"idle"``
        waypoints : list of (x,y,z), optional
            Required for ``"walking"`` and ``"loading"`` tasks.
        """
        state = self._workers.get(worker_id)
        if state is None:
            raise KeyError(f"Worker {worker_id} not found.")

        state.task = task
        state.task_timer = 0.0
        state.phase = 0
        state.anim_time = 0.0

        if waypoints:
            state.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]

        self._apply_animation(state)

    def simulate_step(self, dt: float) -> None:
        """
        Advance all worker state machines by one timestep.

        Parameters
        ----------
        dt : float
            Simulation timestep in seconds.
        """
        for wid, state in self._workers.items():
            state.task_timer += dt
            state.anim_time += dt

            if state.task == "idle":
                self._step_idle(state, dt)
            elif state.task == "walking":
                self._step_walking(state, dt)
            elif state.task == "picking":
                self._step_picking(state, dt)
            elif state.task == "loading":
                self._step_loading(state, dt)

            # Update USD xform
            self._update_prim_transform(state)

    def get_worker_poses(self) -> Dict[int, np.ndarray]:
        """
        Return the SE(3) pose of every worker as a 4×4 homogeneous matrix.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from worker_id to 4×4 float64 pose matrix.
        """
        poses: Dict[int, np.ndarray] = {}
        for wid, state in self._workers.items():
            poses[wid] = self._build_se3(state.position, state.heading_rad)
        return poses

    def get_worker_count(self) -> int:
        return len(self._workers)

    def remove_worker(self, worker_id: int) -> None:
        """Remove a worker from the stage and internal state."""
        state = self._workers.pop(worker_id, None)
        if state is not None:
            prim = self._stage.GetPrimAtPath(state.prim_path)
            if prim.IsValid():
                self._stage.RemovePrim(prim.GetPath())

    def clear_all_workers(self) -> None:
        for wid in list(self._workers.keys()):
            self.remove_worker(wid)

    # ------------------------------------------------------------------
    # State machine steps
    # ------------------------------------------------------------------

    def _step_idle(self, state: _WorkerState, dt: float) -> None:
        # Minor sway on Z to simulate breathing (cosmetic) — oscillates around spawn_z
        state.position[2] = state.spawn_z + 0.001 * math.sin(2.0 * math.pi * state.task_timer * 0.25)

    def _step_walking(self, state: _WorkerState, dt: float) -> None:
        if not state.waypoints:
            state.task = "idle"
            self._apply_animation(state)
            return

        target = state.waypoints[0]
        diff = target[:2] - state.position[:2]
        dist = float(np.linalg.norm(diff))

        if dist < 0.1:
            state.waypoints.pop(0)
            if not state.waypoints:
                state.task = "idle"
                self._apply_animation(state)
            return

        direction = diff / dist
        move = min(_WALK_SPEED * dt, dist)
        state.position[:2] += direction * move
        state.heading_rad = math.atan2(direction[1], direction[0])

    def _step_picking(self, state: _WorkerState, dt: float) -> None:
        """
        Four-phase picking cycle:
          0 — reach down (0 → 0.8 s)
          1 — grasp + lift (0.8 → 2.0 s)
          2 — carry (2.0 → 3.0 s)
          3 — place + retract (3.0 → 3.5 s)
        Then loop.
        """
        cycle = state.task_timer % _PICK_CYCLE_DURATION
        if cycle < 0.8:
            state.phase = 0
        elif cycle < 2.0:
            state.phase = 1
        elif cycle < 3.0:
            state.phase = 2
        else:
            state.phase = 3

    def _step_loading(self, state: _WorkerState, dt: float) -> None:
        """
        Push-cart cycle: lean forward and advance along heading.
        """
        push_dist = 0.4  # metres per push cycle
        cycle = state.task_timer % _LOAD_CYCLE_DURATION
        if cycle < _LOAD_CYCLE_DURATION / 2:
            # Push phase: move forward
            state.position[0] += math.cos(state.heading_rad) * push_dist * dt / (_LOAD_CYCLE_DURATION / 2)
            state.position[1] += math.sin(state.heading_rad) * push_dist * dt / (_LOAD_CYCLE_DURATION / 2)
        # else: recovery phase (stationary)

    # ------------------------------------------------------------------
    # Animation helpers
    # ------------------------------------------------------------------

    def _apply_animation(self, state: _WorkerState) -> None:
        """
        Bind the animation clip for the current task via USD value clips on the
        SkelRoot prim. The SkelRoot is searched in the hierarchy because the
        referenced character asset's root prim is an Xform, not a SkelRoot.
        """
        clip_path = _ANIM_CLIPS.get(state.task)
        if clip_path is None:
            return

        prim = self._stage.GetPrimAtPath(state.prim_path)
        if not prim.IsValid():
            return

        skel_root_prim = self._find_skel_root(prim)
        if skel_root_prim is None:
            return

        try:
            clip_set_name = "default"
            clips_api = Usd.ClipsAPI(skel_root_prim)
            clips_api.SetClipAssetPaths(
                Vt.StringArray([clip_path]), clip_set_name
            )
            clips_api.SetClipPrimPath(_ANIM_CLIP_PRIM_PATH, clip_set_name)
            clips_api.SetClipActive(
                Vt.Vec2dArray([(0.0, 0)]), clip_set_name
            )
            clips_api.SetClipTimes(
                Vt.Vec2dArray([(0.0, 0.0), (100.0, 100.0)]), clip_set_name
            )
        except Exception:
            pass

    def _find_skel_root(self, prim: Usd.Prim) -> Optional[Usd.Prim]:
        """Return the first SkelRoot prim in the subtree rooted at prim."""
        if prim.IsA(UsdSkel.Root):
            return prim
        for child in prim.GetChildren():
            found = self._find_skel_root(child)
            if found is not None:
                return found
        return None

    # ------------------------------------------------------------------
    # USD transform update
    # ------------------------------------------------------------------

    def _update_prim_transform(self, state: _WorkerState) -> None:
        prim = self._stage.GetPrimAtPath(state.prim_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}

        if "xformOp:translate" in ops:
            ops["xformOp:translate"].Set(Gf.Vec3d(*state.position.tolist()))
        if "xformOp:rotateZ" in ops:
            ops["xformOp:rotateZ"].Set(math.degrees(state.heading_rad))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_se3(position: np.ndarray, heading_rad: float) -> np.ndarray:
        """Build a 4×4 SE(3) matrix from position + yaw."""
        c, s = math.cos(heading_rad), math.sin(heading_rad)
        mat = np.array([
            [c, -s, 0.0, position[0]],
            [s,  c, 0.0, position[1]],
            [0.0, 0.0, 1.0, position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return mat

    def _ensure_scope(self) -> None:
        prim = self._stage.GetPrimAtPath(self._scope)
        if not prim.IsValid():
            # Create parent scopes as needed
            parts = self._scope.split("/")
            current = ""
            for part in parts:
                if not part:
                    continue
                current += f"/{part}"
                if not self._stage.GetPrimAtPath(current).IsValid():
                    UsdGeom.Scope.Define(self._stage, current)
