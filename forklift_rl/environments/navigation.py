"""
warehousegpt.forklift_rl.environments.navigation
=================================================
Navigation task environment: drive from start to goal while avoiding obstacles.

Task description
----------------
The forklift must navigate from its spawn position to a randomly placed
goal location in the warehouse.  Static and dynamic obstacles (shelving
racks, parked equipment, walking workers) populate the arena.

Reward function
---------------
Dense shaping drives efficient learning:

    r_t = r_progress + r_collision + r_time + r_orientation + r_goal

    r_progress    = β₁ · (d_{t-1} - d_t) / d_max
                    Positive when closing distance to goal.

    r_collision   = −γ · k
                    Fired when LiDAR min range < safety_radius.
                    k scales with speed at collision.

    r_time        = −δ (step penalty, encourages efficiency)

    r_orientation = β₂ · cos(Δθ)
                    Positive when forklift faces goal direction.

    r_goal        = +Ω  (terminal bonus, default 100.0)

Curriculum learning
-------------------
Episode difficulty is controlled by ``obstacle_density`` which increases
linearly over training stages.  The scheduler is driven by the trainer
and updated via ``set_curriculum_level()``.

Stages:
    Level 0: empty arena, fixed goal 5 m ahead.
    Level 1: 5 random static obstacles.
    Level 2: 15 static + 2 moving workers.
    Level 3: 30 static + 5 moving workers + dynamic forklifts.
    Level 4: full warehouse layout from Isaac Lab scene graph.

Dependencies
------------
    pip install gymnasium numpy
"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import numpy as np

from warehousegpt.forklift_rl.environments.base_env import (
    OBS_DIM,
    _LIDAR_BEAMS,
    _LIDAR_MAX_RANGE,
    _MAX_LINEAR_VEL,
    WarehouseBaseEnv,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curriculum parameters
# ---------------------------------------------------------------------------

_CURRICULUM_STAGES: list[dict[str, Any]] = [
    {"n_static_obstacles": 0,  "n_dynamic_agents": 0, "goal_dist_range": (5.0, 5.0)},
    {"n_static_obstacles": 5,  "n_dynamic_agents": 0, "goal_dist_range": (5.0, 15.0)},
    {"n_static_obstacles": 15, "n_dynamic_agents": 2, "goal_dist_range": (10.0, 25.0)},
    {"n_static_obstacles": 30, "n_dynamic_agents": 5, "goal_dist_range": (15.0, 40.0)},
    {"n_static_obstacles": 60, "n_dynamic_agents": 10, "goal_dist_range": (20.0, 50.0)},
]

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

_W_PROGRESS = 2.0
_W_COLLISION = -50.0
_W_TIME = -0.005
_W_ORIENTATION = 0.3
_W_GOAL = 100.0

_GOAL_RADIUS = 1.0          # metres — tolerance for "arrived"
_SAFETY_RADIUS = 0.8        # metres — forklift footprint + margin
_LIDAR_COLLISION_IDX = slice(0, _LIDAR_BEAMS)  # index into obs vector


class NavigationEnv(WarehouseBaseEnv):
    """
    Forklift navigation environment with curriculum learning.

    Parameters
    ----------
    curriculum_level:
        Initial curriculum stage (0–4).  Increases via ``set_curriculum_level()``.
    reward_weights:
        Override default reward weights {progress, collision, time, orientation, goal}.
    goal_position:
        Fixed goal (x, y) in metres.  If None, randomised each episode.
    **base_kwargs:
        Forwarded to :class:`WarehouseBaseEnv`.

    Usage
    -----
    ::

        env = NavigationEnv(curriculum_level=0)
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                obs, info = env.reset()
    """

    def __init__(
        self,
        curriculum_level: int = 0,
        reward_weights: dict[str, float] | None = None,
        goal_position: tuple[float, float] | None = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self._curriculum_level = min(max(curriculum_level, 0), len(_CURRICULUM_STAGES) - 1)
        self._rw = {
            "progress": _W_PROGRESS,
            "collision": _W_COLLISION,
            "time": _W_TIME,
            "orientation": _W_ORIENTATION,
            "goal": _W_GOAL,
        }
        if reward_weights:
            self._rw.update(reward_weights)

        self._fixed_goal = goal_position
        self._goal: np.ndarray = np.zeros(2, dtype=np.float32)
        self._prev_dist: float = 0.0
        self._obstacle_positions: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._episode_collisions: int = 0
        self._goal_reached: bool = False

    # ------------------------------------------------------------------
    # Curriculum control
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int) -> None:
        """Advance or set the curriculum difficulty level."""
        self._curriculum_level = min(max(level, 0), len(_CURRICULUM_STAGES) - 1)
        logger.info("NavigationEnv curriculum level → %d", self._curriculum_level)

    @property
    def curriculum_level(self) -> int:
        return self._curriculum_level

    # ------------------------------------------------------------------
    # Episode lifecycle hooks
    # ------------------------------------------------------------------

    def _on_reset(self, info: dict[str, Any]) -> None:
        self._goal_reached = False
        self._episode_collisions = 0

        stage = _CURRICULUM_STAGES[self._curriculum_level]
        rng = getattr(self, "_rng", np.random.default_rng())

        # Place goal
        if self._fixed_goal is not None:
            self._goal = np.array(self._fixed_goal, dtype=np.float32)
        else:
            lo, hi = stage["goal_dist_range"]
            dist = float(rng.uniform(lo, hi))
            angle = float(rng.uniform(0, 2 * np.pi))
            start_xy = self._state["pose"][:2]
            self._goal = start_xy + np.array([dist * np.cos(angle), dist * np.sin(angle)])

        # Place obstacles in LiDAR map
        n_obs = stage["n_static_obstacles"]
        if n_obs > 0:
            self._obstacle_positions = rng.uniform(-20.0, 20.0, size=(n_obs, 2)).astype(np.float32)
            self._inject_obstacles_into_lidar()
        else:
            self._obstacle_positions = np.empty((0, 2))

        self._prev_dist = float(np.linalg.norm(self._goal - self._state["pose"][:2]))

        info["goal"] = self._goal.tolist()
        info["curriculum_level"] = self._curriculum_level
        info["n_obstacles"] = n_obs

    def _inject_obstacles_into_lidar(self) -> None:
        """Project obstacle positions into LiDAR beams."""
        pose = self._state["pose"]
        pos = pose[:2]
        yaw = float(pose[5])

        for obs_xy in self._obstacle_positions:
            rel = obs_xy - pos
            dist = float(np.linalg.norm(rel))
            if dist > _LIDAR_MAX_RANGE:
                continue
            angle = float(np.arctan2(rel[1], rel[0])) - yaw
            angle = float(np.arctan2(np.sin(angle), np.cos(angle)))
            beam_idx = int((angle + np.pi) / (2 * np.pi) * _LIDAR_BEAMS) % _LIDAR_BEAMS
            # Illuminate 3 adjacent beams
            for delta in (-1, 0, 1):
                idx = (beam_idx + delta) % _LIDAR_BEAMS
                self._state["lidar"][idx] = min(self._state["lidar"][idx], dist)

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        pose = info.get("pose", self._state["pose"])
        pos = np.array(pose[:2], dtype=np.float32)

        # Goal distance
        dist = float(np.linalg.norm(self._goal - pos))

        # Progress reward
        r_progress = self._rw["progress"] * (self._prev_dist - dist) / max(self._prev_dist, 1e-3)
        self._prev_dist = dist

        # Collision penalty
        lidar_segment = obs[:_LIDAR_BEAMS]
        min_range = float(lidar_segment.min())
        r_collision = 0.0
        if min_range < _SAFETY_RADIUS:
            speed = float(abs(self._state["velocity"][0]))
            r_collision = self._rw["collision"] * (1.0 + speed)
            self._episode_collisions += 1

        # Time penalty
        r_time = self._rw["time"]

        # Orientation toward goal
        goal_dir = self._goal - pos
        goal_angle = float(np.arctan2(goal_dir[1], goal_dir[0]))
        yaw = float(pose[5])
        angle_err = float(np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw)))
        r_orientation = self._rw["orientation"] * float(np.cos(angle_err))

        # Goal bonus
        r_goal = 0.0
        if dist < _GOAL_RADIUS:
            r_goal = self._rw["goal"]
            self._goal_reached = True

        total = r_progress + r_collision + r_time + r_orientation + r_goal
        return float(total)

    def _is_terminated(self, obs: np.ndarray, info: dict[str, Any]) -> bool:
        if self._goal_reached:
            logger.debug(
                "NavigationEnv: goal reached in %d steps (episode %d).",
                self._step_count,
                self._episode_count,
            )
            return True
        # Hard collision termination
        lidar_segment = obs[:_LIDAR_BEAMS]
        if float(lidar_segment.min()) < 0.3:
            return True
        return False

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _reset_info(self) -> dict[str, Any]:
        info = super()._reset_info()
        info["goal"] = self._goal.tolist() if hasattr(self, "_goal") else [0.0, 0.0]
        info["curriculum_level"] = self._curriculum_level
        return info

    def episode_stats(self) -> dict[str, Any]:
        """Return statistics for the completed episode."""
        return {
            "goal_reached": self._goal_reached,
            "steps": self._step_count,
            "collisions": self._episode_collisions,
            "curriculum_level": self._curriculum_level,
        }
