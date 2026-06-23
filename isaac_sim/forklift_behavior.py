"""
Forklift Behavior — NVIDIA Isaac Sim

Manages autonomous and scripted forklift agents in the warehouse:
  - Spawning via USD reference on the stage
  - A* pathfinding on a 2-D navmesh grid
  - Pallet pickup / place sequences
  - Scripted safety incident replay (near-miss, collision, tip-over)

Leverages omni.isaac.wheeled_robots for low-level drive articulation.

Usage:
    from isaac_sim.forklift_behavior import ForkiftBehavior
    fb = ForkiftBehavior(stage)
    fid = fb.spawn_forklift((15.0, 8.0, 0.0))
    fb.navigate_to(fid, np.array([60.0, 20.0, 0.0]))
    fb.pickup_pallet(fid, pallet_id="Pallet_0042")
    fb.place_pallet(fid, location=np.array([70.0, 25.0, 0.0]))
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    get_current_stage,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


# ---------------------------------------------------------------------------
# Asset library
# ---------------------------------------------------------------------------

_FORKLIFT_ASSETS: Dict[str, str] = {
    "counterbalance": (
        "omniverse://localhost/NVIDIA/Assets/Warehouse/Equipment/Forklift_A.usd"
    ),
    "reach_truck": (
        "omniverse://localhost/NVIDIA/Assets/Warehouse/Equipment/Reach_Truck_A.usd"
    ),
    "pallet_jack": (
        "omniverse://localhost/NVIDIA/Assets/Warehouse/Equipment/PalletJack_A.usd"
    ),
    "order_picker": (
        "omniverse://localhost/NVIDIA/Assets/Warehouse/Equipment/OrderPicker_A.usd"
    ),
}

_DEFAULT_FORKLIFT_MODEL = "counterbalance"

# Drive constants
_MAX_SPEED_MS = 2.5          # m/s travel speed
_TURN_RATE_RPS = 0.8         # rad/s turning rate
_FORK_LIFT_SPEED = 0.1       # m/s vertical fork travel
_FORK_INSERT_SPEED = 0.3     # m/s horizontal insertion

# A* grid resolution
_NAV_GRID_CELL_M = 0.5       # metres per navmesh cell


# ---------------------------------------------------------------------------
# State machines
# ---------------------------------------------------------------------------

class ForkliftPhase(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    APPROACHING_PALLET = auto()
    INSERTING_FORKS = auto()
    LIFTING = auto()
    CARRYING = auto()
    LOWERING = auto()
    RETRACTING_FORKS = auto()
    INCIDENT = auto()


@dataclass
class _ForkliftState:
    forklift_id: int
    prim_path: str
    model: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_rad: float = 0.0
    fork_height: float = 0.0        # metres above ground
    fork_extension: float = 0.0     # metres of horizontal insertion
    phase: ForkliftPhase = ForkliftPhase.IDLE
    nav_path: List[np.ndarray] = field(default_factory=list)
    target_position: Optional[np.ndarray] = None
    carried_pallet_id: Optional[str] = None
    task_timer: float = 0.0
    incident_type: Optional[str] = None


# ---------------------------------------------------------------------------
# NavMesh: simple 2-D A* on a grid
# ---------------------------------------------------------------------------

class NavMesh:
    """
    2-D occupancy grid with A* path planning.

    Obstacles are marked as blocked cells; all other cells are free.
    """

    def __init__(
        self,
        width_m: float,
        height_m: float,
        cell_size_m: float = _NAV_GRID_CELL_M,
    ) -> None:
        self.cell = cell_size_m
        self.cols = int(math.ceil(width_m / cell_size_m))
        self.rows = int(math.ceil(height_m / cell_size_m))
        self.grid: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def mark_obstacle(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        c0 = max(0, int(x_min / self.cell))
        c1 = min(self.cols - 1, int(x_max / self.cell))
        r0 = max(0, int(y_min / self.cell))
        r1 = min(self.rows - 1, int(y_max / self.cell))
        self.grid[r0 : r1 + 1, c0 : c1 + 1] = 1

    def find_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> List[np.ndarray]:
        """
        Return a list of (x, y) waypoints from start to goal.
        Uses A* on the occupancy grid with 8-connectivity.
        """

        def to_cell(pt: np.ndarray) -> Tuple[int, int]:
            return (
                int(np.clip(pt[1] / self.cell, 0, self.rows - 1)),
                int(np.clip(pt[0] / self.cell, 0, self.cols - 1)),
            )

        def to_world(r: int, c: int) -> np.ndarray:
            return np.array(
                [(c + 0.5) * self.cell, (r + 0.5) * self.cell], dtype=np.float64
            )

        start = to_cell(start_xy)
        goal = to_cell(goal_xy)

        if self.grid[start] == 1:
            return []   # Start is blocked
        if self.grid[goal] == 1:
            return []   # Goal is blocked

        # Priority queue: (f, g, (row, col), parent)
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0.0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        neighbors_delta = [
            (-1, -1), (-1, 0), (-1, 1),
            (0,  -1),           (0,  1),
            (1,  -1), (1,  0), (1,  1),
        ]

        while open_set:
            _, g, current = heapq.heappop(open_set)

            if current == goal:
                path: List[np.ndarray] = []
                node: Optional[Tuple[int, int]] = current
                while node is not None:
                    path.append(to_world(*node))
                    node = came_from[node]
                path.reverse()
                return path

            if g > g_score.get(current, float("inf")):
                continue

            r, c = current
            for dr, dc in neighbors_delta:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc] == 1:
                    continue
                step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                new_g = g + step_cost
                neighbor = (nr, nc)
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    h = math.hypot(goal[0] - nr, goal[1] - nc)
                    heapq.heappush(open_set, (new_g + h, new_g, neighbor))
                    came_from[neighbor] = current

        return []  # No path found


# ---------------------------------------------------------------------------
# ForkiftBehavior
# ---------------------------------------------------------------------------

class ForkiftBehavior:
    """
    Manages a fleet of forklift agents in the warehouse.

    Parameters
    ----------
    stage : Usd.Stage, optional
    forklifts_scope : str
        USD path for forklift prims.
    floor_length : float
        Warehouse length (X) in metres — used for navmesh.
    floor_width : float
        Warehouse width (Y) in metres.
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        forklifts_scope: str = "/World/Actors/Forklifts",
        floor_length: float = 120.0,
        floor_width: float = 60.0,
    ) -> None:
        self._stage = stage or get_current_stage()
        self._scope = forklifts_scope
        self._forklifts: Dict[int, _ForkliftState] = {}
        self._next_id = 0
        self._navmesh = NavMesh(floor_length, floor_width)
        self._pallet_positions: Dict[str, np.ndarray] = {}

        self._ensure_scope()

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def spawn_forklift(
        self,
        position: Tuple[float, float, float],
        model: str = _DEFAULT_FORKLIFT_MODEL,
        heading_deg: float = 0.0,
    ) -> int:
        """
        Instantiate a forklift USD asset and register it.

        Parameters
        ----------
        position : (x, y, z)
            World-space spawn position.
        model : str
            Key in _FORKLIFT_ASSETS: ``"counterbalance"``, ``"reach_truck"``,
            ``"pallet_jack"``, ``"order_picker"``.
        heading_deg : float
            Initial heading.

        Returns
        -------
        int
            Unique forklift ID.
        """
        asset_path = _FORKLIFT_ASSETS.get(model, _FORKLIFT_ASSETS[_DEFAULT_FORKLIFT_MODEL])
        fid = self._next_id
        self._next_id += 1
        prim_path = f"{self._scope}/Forklift_{fid:04d}"

        ref_prim = add_reference_to_stage(
            usd_path=asset_path, prim_path=prim_path
        )
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            xf.AddRotateZOp().Set(heading_deg)
            if UsdPhysics.RigidBodyAPI.CanApply(prim):
                UsdPhysics.RigidBodyAPI.Apply(prim)

        state = _ForkliftState(
            forklift_id=fid,
            prim_path=prim_path,
            model=model,
            position=np.array(position, dtype=np.float64),
            heading_rad=math.radians(heading_deg),
        )
        self._forklifts[fid] = state
        return fid

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_to(
        self,
        forklift_id: int,
        target: np.ndarray,
        blocking: bool = False,
    ) -> List[np.ndarray]:
        """
        Plan an A* path from the forklift's current position to `target`
        and set the forklift into NAVIGATING state.

        Parameters
        ----------
        forklift_id : int
        target : np.ndarray, shape (2,) or (3,)
            Goal position in world space.
        blocking : bool
            If True, calls simulate_step() in a loop until arrival.

        Returns
        -------
        list[np.ndarray]
            Waypoints of the planned path (world XY).
        """
        state = self._get_state(forklift_id)
        path = self._navmesh.find_path(state.position[:2], target[:2])
        if not path:
            # Straight-line fallback if navmesh fails
            path = [target[:2].copy()]

        state.nav_path = path
        state.target_position = target.copy()
        state.phase = ForkliftPhase.NAVIGATING

        if blocking:
            while state.phase == ForkliftPhase.NAVIGATING:
                self.simulate_step(1.0 / 60.0)

        return path

    # ------------------------------------------------------------------
    # Pallet operations
    # ------------------------------------------------------------------

    def register_pallet(
        self, pallet_id: str, position: np.ndarray
    ) -> None:
        """Register a pallet location for pickup operations."""
        self._pallet_positions[pallet_id] = position.copy()

    def pickup_pallet(
        self, forklift_id: int, pallet_id: str
    ) -> None:
        """
        Begin the pickup sequence:
          1. Navigate to pallet position
          2. Approach + align
          3. Insert forks under pallet
          4. Lift to carry height (0.3 m)

        The sequence is executed via simulate_step() calls.

        Parameters
        ----------
        forklift_id : int
        pallet_id : str
            Must be registered via register_pallet().
        """
        state = self._get_state(forklift_id)
        pallet_pos = self._pallet_positions.get(pallet_id)
        if pallet_pos is None:
            raise ValueError(f"Pallet '{pallet_id}' not registered.")

        # Navigate to approach point (1 m back from pallet)
        approach = pallet_pos.copy()
        approach[0] -= 1.0
        self.navigate_to(forklift_id, approach, blocking=True)

        state.phase = ForkliftPhase.APPROACHING_PALLET
        state.carried_pallet_id = pallet_id
        state.task_timer = 0.0

    def place_pallet(
        self, forklift_id: int, location: np.ndarray
    ) -> None:
        """
        Begin pallet placement sequence:
          1. Navigate to location
          2. Lower forks to ground level
          3. Retract forks

        Parameters
        ----------
        forklift_id : int
        location : np.ndarray
            Target placement position.
        """
        state = self._get_state(forklift_id)
        if state.carried_pallet_id is None:
            raise RuntimeError(f"Forklift {forklift_id} is not carrying a pallet.")

        self.navigate_to(forklift_id, location, blocking=True)
        state.phase = ForkliftPhase.LOWERING
        state.task_timer = 0.0

    # ------------------------------------------------------------------
    # Incident simulation
    # ------------------------------------------------------------------

    def simulate_incident(
        self,
        forklift_id: int,
        incident_type: Literal["near_miss", "collision", "tip_over"],
        severity: float = 1.0,
    ) -> None:
        """
        Trigger a scripted safety incident.

        Parameters
        ----------
        forklift_id : int
        incident_type : str
            ``"near_miss"``  — forklift stops within 0.5 m of obstacle.
            ``"collision"``  — forklift impacts wall/rack with physics impulse.
            ``"tip_over"``   — rotational impulse applied to chassis.
        severity : float
            Incident severity scale 0–1 (affects impulse magnitude).
        """
        state = self._get_state(forklift_id)
        state.phase = ForkliftPhase.INCIDENT
        state.incident_type = incident_type
        state.task_timer = 0.0

        prim = self._stage.GetPrimAtPath(state.prim_path)
        if not prim.IsValid():
            return

        rb_api = UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None

        if incident_type == "near_miss":
            # Abrupt deceleration — reduce velocity to zero over 0.1 s
            if rb_api:
                vel_attr = prim.GetAttribute("physics:velocity")
                if vel_attr:
                    vel_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            # Stop navigation
            state.nav_path.clear()

        elif incident_type == "collision":
            # Apply linear impulse along current heading
            impulse_mag = 5000.0 * severity
            impulse = Gf.Vec3f(
                float(math.cos(state.heading_rad) * impulse_mag),
                float(math.sin(state.heading_rad) * impulse_mag),
                0.0,
            )
            if rb_api:
                force_attr = prim.GetAttribute("physics:angularVelocity")
                try:
                    from omni.physx import get_physx_interface
                    physx = get_physx_interface()
                    physx.apply_force_at_pos(
                        str(prim.GetPath()),
                        carb.Float3(impulse[0], impulse[1], impulse[2]),
                        carb.Float3(*state.position.tolist()),
                    )
                except Exception:
                    pass  # Physics API unavailable in test mode

        elif incident_type == "tip_over":
            # Apply rotational impulse to induce lateral roll
            angular_impulse = 2000.0 * severity
            try:
                from omni.physx import get_physx_interface
                import carb
                physx = get_physx_interface()
                physx.apply_torque(
                    str(prim.GetPath()),
                    carb.Float3(
                        float(math.cos(state.heading_rad) * angular_impulse),
                        float(math.sin(state.heading_rad) * angular_impulse),
                        0.0,
                    ),
                )
            except Exception:
                # Fallback: tilt the xform directly
                xf = UsdGeom.Xformable(prim)
                xf.AddRotateXOp().Set(45.0 * severity)

        state.phase = ForkliftPhase.IDLE

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def simulate_step(self, dt: float) -> None:
        """
        Advance all forklift state machines by one timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        """
        for fid, state in self._forklifts.items():
            state.task_timer += dt

            if state.phase == ForkliftPhase.NAVIGATING:
                self._step_navigate(state, dt)
            elif state.phase == ForkliftPhase.APPROACHING_PALLET:
                self._step_approach_pallet(state, dt)
            elif state.phase == ForkliftPhase.INSERTING_FORKS:
                self._step_insert_forks(state, dt)
            elif state.phase == ForkliftPhase.LIFTING:
                self._step_lift(state, dt)
            elif state.phase == ForkliftPhase.LOWERING:
                self._step_lower(state, dt)
            elif state.phase == ForkliftPhase.RETRACTING_FORKS:
                self._step_retract_forks(state, dt)

            self._update_prim_transform(state)

    # ------------------------------------------------------------------
    # State machine steps
    # ------------------------------------------------------------------

    def _step_navigate(self, state: _ForkliftState, dt: float) -> None:
        if not state.nav_path:
            state.phase = ForkliftPhase.IDLE
            return

        waypoint = state.nav_path[0]
        diff = waypoint - state.position[:2]
        dist = float(np.linalg.norm(diff))

        if dist < 0.15:
            state.nav_path.pop(0)
            return

        # Turn toward waypoint
        desired_heading = math.atan2(diff[1], diff[0])
        heading_error = self._wrap_angle(desired_heading - state.heading_rad)
        max_turn = _TURN_RATE_RPS * dt
        state.heading_rad += np.clip(heading_error, -max_turn, max_turn)

        # Drive forward if roughly aligned
        if abs(heading_error) < math.pi / 4:
            move = min(_MAX_SPEED_MS * dt, dist)
            state.position[0] += math.cos(state.heading_rad) * move
            state.position[1] += math.sin(state.heading_rad) * move

    def _step_approach_pallet(self, state: _ForkliftState, dt: float) -> None:
        pallet_pos = self._pallet_positions.get(state.carried_pallet_id or "")
        if pallet_pos is None:
            state.phase = ForkliftPhase.IDLE
            return
        diff = pallet_pos[:2] - state.position[:2]
        dist = float(np.linalg.norm(diff))
        if dist < 0.05:
            state.phase = ForkliftPhase.INSERTING_FORKS
            state.task_timer = 0.0
        else:
            move = min(0.3 * dt, dist)   # slow approach
            state.position[0] += (diff[0] / dist) * move
            state.position[1] += (diff[1] / dist) * move

    def _step_insert_forks(self, state: _ForkliftState, dt: float) -> None:
        # Extend forks horizontally by _FORK_INSERT_SPEED m/s
        state.fork_extension = min(1.0, state.fork_extension + _FORK_INSERT_SPEED * dt)
        if state.fork_extension >= 0.9:
            state.phase = ForkliftPhase.LIFTING
            state.task_timer = 0.0

    def _step_lift(self, state: _ForkliftState, dt: float) -> None:
        carry_height = 0.3  # metres
        state.fork_height = min(carry_height, state.fork_height + _FORK_LIFT_SPEED * dt)
        if state.fork_height >= carry_height - 0.01:
            state.phase = ForkliftPhase.CARRYING

    def _step_lower(self, state: _ForkliftState, dt: float) -> None:
        state.fork_height = max(0.0, state.fork_height - _FORK_LIFT_SPEED * dt)
        if state.fork_height <= 0.01:
            # Update pallet registered position
            if state.carried_pallet_id:
                self._pallet_positions[state.carried_pallet_id] = state.position.copy()
            state.phase = ForkliftPhase.RETRACTING_FORKS
            state.task_timer = 0.0

    def _step_retract_forks(self, state: _ForkliftState, dt: float) -> None:
        state.fork_extension = max(0.0, state.fork_extension - _FORK_INSERT_SPEED * dt)
        if state.fork_extension <= 0.01:
            state.carried_pallet_id = None
            state.phase = ForkliftPhase.IDLE

    # ------------------------------------------------------------------
    # USD transform
    # ------------------------------------------------------------------

    def _update_prim_transform(self, state: _ForkliftState) -> None:
        prim = self._stage.GetPrimAtPath(state.prim_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        if "xformOp:translate" in ops:
            ops["xformOp:translate"].Set(Gf.Vec3d(*state.position.tolist()))
        if "xformOp:rotateZ" in ops:
            ops["xformOp:rotateZ"].Set(math.degrees(state.heading_rad))

        # Update fork mast height if child prim "Mast" exists
        mast_prim = self._stage.GetPrimAtPath(f"{state.prim_path}/Mast")
        if mast_prim.IsValid():
            mast_xf = UsdGeom.Xformable(mast_prim)
            mast_ops = {op.GetOpName(): op for op in mast_xf.GetOrderedXformOps()}
            if "xformOp:translate" in mast_ops:
                current = mast_ops["xformOp:translate"].Get()
                if current:
                    mast_ops["xformOp:translate"].Set(
                        Gf.Vec3d(float(current[0]), float(current[1]), state.fork_height)
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self, forklift_id: int) -> _ForkliftState:
        state = self._forklifts.get(forklift_id)
        if state is None:
            raise KeyError(f"Forklift {forklift_id} not found.")
        return state

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _ensure_scope(self) -> None:
        parts = self._scope.split("/")
        current = ""
        for part in parts:
            if not part:
                continue
            current += f"/{part}"
            if not self._stage.GetPrimAtPath(current).IsValid():
                UsdGeom.Scope.Define(self._stage, current)
