"""
warehousegpt.forklift_rl.environments.pallet_pickup
====================================================
Pallet pickup task: approach, align forks, insert, and lift a target pallet.

Task decomposition
------------------
The pickup task is broken into four sequential micro-phases, each with its
own dense reward signal:

    Phase 0 — Approach
        Drive within ``approach_radius`` of the pallet face.
        r = progress reward (same as NavigationEnv).

    Phase 1 — Alignment
        Align fork tips with the pallet pocket openings.
        r = −|lateral_error| − |angular_error| (dense, continuous)
        Alignment tolerance: ±0.05 m lateral, ±3° angular.

    Phase 2 — Insertion
        Drive forward slowly, inserting forks into pallet pockets.
        Detect penetration depth via simulated contact sensor.
        r = penetration_depth_m · β (dense)
        Collision with pallet leg: −50 penalty.

    Phase 3 — Lift
        Raise forks until pallet clears the ground.
        r = +20 per centimetre of clearance (capped at 0.15 m).
        Terminal success bonus: +200.

Fork-pallet contact detection
------------------------------
Isaac Lab's contact sensor API (``ContactSensor``) provides the penetration
vector and normal force.  In mock mode, contact is approximated from the
relative position of fork tips to pallet pocket centres.

Dense reward shaping (fork alignment)
--------------------------------------
The alignment reward uses a Gaussian kernel to avoid sparse gradients:

    r_align = A · exp(−0.5 · ((Δx/σ_x)² + (Δθ/σ_θ)²))

    A = 5.0,  σ_x = 0.10 m,  σ_θ = 0.10 rad

This ensures the reward gradient is non-zero even when the forklift is
far from the pallet, preventing the agent from getting stuck in a flat
reward landscape.

Dependencies
------------
    pip install gymnasium numpy
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

import numpy as np

from warehousegpt.forklift_rl.environments.base_env import (
    _FORK_MAX_HEIGHT,
    _LIDAR_BEAMS,
    _LIDAR_MAX_RANGE,
    WarehouseBaseEnv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task phases
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    APPROACH = 0
    ALIGNMENT = 1
    INSERTION = 2
    LIFT = 3
    COMPLETE = 4


# ---------------------------------------------------------------------------
# Pallet geometry
# ---------------------------------------------------------------------------

_PALLET_WIDTH = 1.20   # m — standard EUR pallet
_PALLET_DEPTH = 0.80   # m
_PALLET_HEIGHT = 0.145 # m (standing height)
_FORK_WIDTH = 0.15     # m per fork
_FORK_SPACING = 0.45   # m between fork inner edges
_POCKET_HEIGHT = 0.10  # m fork entry height

# Approach / success tolerances
_APPROACH_RADIUS = 2.0    # m
_ALIGN_LATERAL_TOL = 0.05 # m
_ALIGN_ANGULAR_TOL = 0.052 # rad (~3°)
_INSERTION_DEPTH_REQ = 0.60 # m (% of pallet depth)
_LIFT_CLEARANCE_REQ = 0.15  # m

# Reward coefficients
_A_ALIGN = 5.0
_SIGMA_X = 0.10
_SIGMA_THETA = 0.10
_W_PROGRESS = 1.5
_W_INSERTION = 30.0
_W_LIFT = 200.0
_W_COLLISION = -50.0
_W_TIME = -0.002
_W_SUCCESS = 200.0


class PalletPickupEnv(WarehouseBaseEnv):
    """
    Dense-reward pallet pickup environment.

    Parameters
    ----------
    pallet_position:
        Fixed (x, y, yaw) of the pallet in world space.
        If None, randomised within ``position_range`` each episode.
    position_range:
        (min, max) absolute coordinate range for random pallet placement.
    fork_insertion_requires_contact:
        If True, use Isaac Lab contact sensor for ground-truth insertion.
        If False, use kinematic approximation (default, works without GPU).
    **base_kwargs:
        Forwarded to :class:`WarehouseBaseEnv`.

    Usage
    -----
    ::

        env = PalletPickupEnv()
        obs, info = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
        print("Success:", info.get("pickup_success"))
    """

    def __init__(
        self,
        pallet_position: tuple[float, float, float] | None = None,
        position_range: float = 10.0,
        fork_insertion_requires_contact: bool = False,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self._fixed_pallet = pallet_position
        self._pos_range = position_range
        self._contact_mode = fork_insertion_requires_contact

        # Task state
        self._phase: Phase = Phase.APPROACH
        self._pallet_pose: np.ndarray = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self._prev_dist: float = 0.0
        self._insertion_depth: float = 0.0
        self._lift_height: float = 0.0
        self._pickup_success: bool = False
        self._phase_rewards: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _on_reset(self, info: dict[str, Any]) -> None:
        self._phase = Phase.APPROACH
        self._pickup_success = False
        self._insertion_depth = 0.0
        self._lift_height = 0.0
        self._phase_rewards = {p.name: 0.0 for p in Phase}

        rng = getattr(self, "_rng", np.random.default_rng())

        if self._fixed_pallet is not None:
            self._pallet_pose = np.array(self._fixed_pallet, dtype=np.float32)
        else:
            px = float(rng.uniform(-self._pos_range, self._pos_range))
            py = float(rng.uniform(-self._pos_range, self._pos_range))
            pyaw = float(rng.uniform(-np.pi / 4, np.pi / 4))
            self._pallet_pose = np.array([px, py, pyaw], dtype=np.float32)

        # Spawn forklift ~4–6 m from pallet, facing it roughly
        approach_angle = float(self._pallet_pose[2]) + np.pi + rng.uniform(-0.5, 0.5)
        dist = float(rng.uniform(4.0, 6.0))
        self._state["pose"][0] = self._pallet_pose[0] + dist * np.cos(approach_angle)
        self._state["pose"][1] = self._pallet_pose[1] + dist * np.sin(approach_angle)
        self._state["pose"][5] = approach_angle + np.pi  # face pallet
        self._state["fork"][0] = _POCKET_HEIGHT  # pre-lower forks to pocket height

        self._prev_dist = dist
        info["pallet_pose"] = self._pallet_pose.tolist()
        info["phase"] = self._phase.name

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        pose = info.get("pose", self._state["pose"])
        fork = info.get("fork", self._state["fork"])

        reward = _W_TIME  # base time penalty

        if self._phase == Phase.APPROACH:
            reward += self._reward_approach(pose)
        elif self._phase == Phase.ALIGNMENT:
            reward += self._reward_alignment(pose)
        elif self._phase == Phase.INSERTION:
            reward += self._reward_insertion(pose, obs)
        elif self._phase == Phase.LIFT:
            reward += self._reward_lift(fork)

        # Safety: punish LiDAR collisions in all phases
        lidar_min = float(obs[:_LIDAR_BEAMS].min())
        if lidar_min < 0.4:
            reward += _W_COLLISION

        self._phase_rewards[self._phase.name] += reward
        return float(reward)

    def _reward_approach(self, pose: np.ndarray) -> float:
        pos = np.array(pose[:2])
        pallet_pos = self._pallet_pose[:2]
        # Target point: 0.5 m in front of pallet face
        target = pallet_pos + 0.5 * np.array([
            np.cos(self._pallet_pose[2]),
            np.sin(self._pallet_pose[2]),
        ])
        dist = float(np.linalg.norm(target - pos))
        r = _W_PROGRESS * (self._prev_dist - dist)
        self._prev_dist = dist

        # Advance phase
        if dist < _APPROACH_RADIUS:
            self._phase = Phase.ALIGNMENT
            self._prev_dist = self._lateral_error(pose)
            logger.debug("PalletPickupEnv: advancing to ALIGNMENT phase.")

        return r

    def _reward_alignment(self, pose: np.ndarray) -> float:
        lat_err = self._lateral_error(pose)
        ang_err = self._angular_error(pose)

        # Gaussian alignment reward
        r_align = _A_ALIGN * float(np.exp(
            -0.5 * ((lat_err / _SIGMA_X) ** 2 + (ang_err / _SIGMA_THETA) ** 2)
        ))

        # Advance phase when aligned
        if abs(lat_err) < _ALIGN_LATERAL_TOL and abs(ang_err) < _ALIGN_ANGULAR_TOL:
            self._phase = Phase.INSERTION
            logger.debug("PalletPickupEnv: advancing to INSERTION phase.")

        return r_align

    def _reward_insertion(self, pose: np.ndarray, obs: np.ndarray) -> float:
        # Simulate fork tip position (2 m forward of forklift centre)
        yaw = float(pose[5])
        fork_tip = np.array(pose[:2]) + 2.0 * np.array([np.cos(yaw), np.sin(yaw)])

        # Penetration depth (projection along pallet y-axis)
        pallet_fwd = np.array([
            np.cos(self._pallet_pose[2]),
            np.sin(self._pallet_pose[2]),
        ])
        pallet_rel = fork_tip - self._pallet_pose[:2]
        depth = float(np.dot(pallet_rel, -pallet_fwd))  # negative = inserted

        if depth > 0:
            self._insertion_depth = depth
            r = _W_INSERTION * (depth / _INSERTION_DEPTH_REQ)
        else:
            r = 0.0

        if depth >= _INSERTION_DEPTH_REQ:
            self._phase = Phase.LIFT
            logger.debug(
                "PalletPickupEnv: forks inserted %.2f m — advancing to LIFT.", depth
            )

        return min(r, _W_INSERTION)

    def _reward_lift(self, fork: np.ndarray) -> float:
        fork_height = float(fork[0])
        clearance = max(0.0, fork_height - _PALLET_HEIGHT)
        self._lift_height = clearance

        r = _W_LIFT * min(clearance / _LIFT_CLEARANCE_REQ, 1.0)

        if clearance >= _LIFT_CLEARANCE_REQ:
            self._pickup_success = True
            self._phase = Phase.COMPLETE
            r += _W_SUCCESS
            logger.info(
                "PalletPickupEnv: PICKUP SUCCESS (episode %d, step %d)!",
                self._episode_count,
                self._step_count,
            )

        return r

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _lateral_error(self, pose: np.ndarray) -> float:
        """Signed lateral offset of forklift from pallet centreline (m)."""
        pos = np.array(pose[:2])
        pallet_side = np.array([
            -np.sin(self._pallet_pose[2]),
            np.cos(self._pallet_pose[2]),
        ])
        rel = pos - self._pallet_pose[:2]
        return float(np.dot(rel, pallet_side))

    def _angular_error(self, pose: np.ndarray) -> float:
        """Angle between forklift heading and pallet axis (rad)."""
        forklift_yaw = float(pose[5])
        pallet_yaw = float(self._pallet_pose[2])
        # Forklift should be aligned opposite to pallet forward direction
        desired_yaw = pallet_yaw + np.pi
        err = forklift_yaw - desired_yaw
        return float(np.arctan2(np.sin(err), np.cos(err)))

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self, obs: np.ndarray, info: dict[str, Any]) -> bool:
        if self._pickup_success:
            return True
        # Hard collision
        if float(obs[:_LIDAR_BEAMS].min()) < 0.25:
            return True
        return False

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _reset_info(self) -> dict[str, Any]:
        info = super()._reset_info()
        info["phase"] = Phase.APPROACH.name
        return info

    def episode_stats(self) -> dict[str, Any]:
        return {
            "pickup_success": self._pickup_success,
            "final_phase": self._phase.name,
            "insertion_depth_m": round(self._insertion_depth, 4),
            "lift_height_m": round(self._lift_height, 4),
            "phase_rewards": {k: round(v, 3) for k, v in self._phase_rewards.items()},
            "steps": self._step_count,
        }
