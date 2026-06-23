"""
warehousegpt.forklift_rl.environments.multi_agent
==================================================
Multi-agent warehouse environment for N autonomous forklifts.

Architecture: CTDE (Centralized Training, Decentralized Execution)
-------------------------------------------------------------------
Training:
    A centralised critic receives the joint observation o = [o_1, …, o_N]
    and the joint action a = [a_1, …, a_N].  This allows agents to
    coordinate during training without communication constraints.

Execution:
    Each agent i executes using only its local observation o_i.  No
    inter-agent communication is required at deployment time.  This
    satisfies the practical constraint that deployed forklifts may
    operate in areas with limited wireless connectivity.

Observation per agent
---------------------
Same as :class:`WarehouseBaseEnv` (1244-dim) plus:
    - Relative positions / headings of K nearest other agents (K=3)
      appended as (K × 4) = 12 extra dims.
Total per-agent obs dim: 1256.

Action per agent
----------------
Same as :class:`WarehouseBaseEnv`: [linear_vel, angular_vel, fork_height, fork_tilt].

Reward structure
----------------
Individual reward:
    r_i = r_task_i + r_collision_avoidance_i

Shared/cooperative reward:
    r_shared = α · Throughput  (pallets delivered per time unit)
    Throughput is computed as the moving average of pallet deliveries
    across all agents over the last 100 steps.

Total agent reward:
    R_i = (1 - λ) · r_i + λ · r_shared
    λ = 0.3 (team incentive coefficient)

Collision avoidance between agents
-----------------------------------
Collision between two agents is detected when their Euclidean distance
falls below ``agent_collision_radius`` (default 2.0 m).  Both agents
receive a penalty of −100 and the episode continues (non-terminal) to
train the agents to recover.

Throughput metric
-----------------
``throughput_rate`` = number of successful pallet deliveries per
1 000 environment steps across all agents.  This is reported in the
``info`` dict and used for TensorBoard logging.

Note on CTDE implementation
----------------------------
This environment implements the *environment* side of CTDE.  The
centralised critic is implemented in the training algorithm (e.g., MAPPO
in ``RLTrainer``).  The environment provides:
    - ``joint_obs()`` — stacked observations for the critic.
    - ``joint_state()`` — global state dict for state-based CTDE.

Dependencies
------------
    pip install gymnasium numpy
"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import numpy as np

from warehousegpt.forklift_rl.environments.base_env import (
    ACT_DIM,
    OBS_DIM,
    _LIDAR_BEAMS,
    _LIDAR_MAX_RANGE,
    WarehouseBaseEnv,
    _MAX_LINEAR_VEL,
)
from warehousegpt.forklift_rl.environments.navigation import NavigationEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_K_NEAREST = 3          # Number of nearest agents to include in obs
_AGENT_OBS_DIM = OBS_DIM + _K_NEAREST * 4  # 1244 + 12 = 1256

_COLLISION_RADIUS = 2.0  # metres — inter-agent collision distance
_COLLISION_PENALTY = -100.0

_LAMBDA_TEAM = 0.30      # Weight of team reward

_WAREHOUSE_HALF_SIZE = 30.0  # metres from centre


class MultiAgentWarehouseEnv:
    """
    Multi-agent warehouse environment for N autonomous forklifts.

    Implements the standard multi-agent Gymnasium-like interface:
        reset() → list[obs]
        step(actions) → (list[obs], list[reward], list[done], list[trunc], list[info])

    Optionally provides CTDE helpers:
        joint_obs() → np.ndarray  (N × per_agent_obs_dim)
        joint_state() → dict

    Parameters
    ----------
    n_agents:
        Number of forklift agents.
    warehouse_size:
        Half-size of the square warehouse arena (metres).
    agent_collision_radius:
        Centre-to-centre distance (m) triggering an inter-agent collision.
    lambda_team:
        Weight of shared throughput reward in total reward.
    max_episode_steps:
        Truncation threshold.
    throughput_window:
        Number of steps over which throughput rate is averaged.
    base_env_kwargs:
        Additional kwargs forwarded to each sub-environment.

    Usage
    -----
    ::

        env = MultiAgentWarehouseEnv(n_agents=4)
        obs_list = env.reset()
        for _ in range(500):
            actions = [agent.act(o) for agent, o in zip(agents, obs_list)]
            obs_list, rews, terms, truncs, infos = env.step(actions)

        # CTDE critic input:
        joint = env.joint_obs()   # shape (4, 1256)
    """

    def __init__(
        self,
        n_agents: int = 4,
        warehouse_size: float = _WAREHOUSE_HALF_SIZE,
        agent_collision_radius: float = _COLLISION_RADIUS,
        lambda_team: float = _LAMBDA_TEAM,
        max_episode_steps: int = 2000,
        throughput_window: int = 100,
        base_env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._n = n_agents
        self._size = warehouse_size
        self._col_radius = agent_collision_radius
        self._lambda = lambda_team
        self._max_steps = max_episode_steps
        self._step_count = 0
        self._deliveries: list[int] = [0] * throughput_window

        kw = base_env_kwargs or {}
        self._envs: list[NavigationEnv] = [
            NavigationEnv(max_episode_steps=max_episode_steps, **kw)
            for _ in range(n_agents)
        ]

        # Per-agent state tracking
        self._obs_buf: list[np.ndarray] = [np.zeros(OBS_DIM, dtype=np.float32)] * n_agents
        self._agent_poses: np.ndarray = np.zeros((n_agents, 6), dtype=np.float32)
        self._agent_done: list[bool] = [False] * n_agents
        self._episode_deliveries: int = 0
        self._delivery_steps: list[int] = []

        logger.info("MultiAgentWarehouseEnv: %d agents, arena=±%.0f m", n_agents, warehouse_size)

    # ------------------------------------------------------------------
    # Interface properties
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return self._n

    @property
    def obs_dim(self) -> int:
        return _AGENT_OBS_DIM

    @property
    def act_dim(self) -> int:
        return ACT_DIM

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """
        Reset all agent environments.

        Returns
        -------
        (obs_list, info_list)
            obs_list: N observations of shape (_AGENT_OBS_DIM,).
        """
        self._step_count = 0
        self._agent_done = [False] * self._n
        self._episode_deliveries = 0
        self._delivery_steps.clear()

        rng = np.random.default_rng(seed)
        obs_list: list[np.ndarray] = []
        info_list: list[dict[str, Any]] = []

        for i, env in enumerate(self._envs):
            obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
            # Separate spawn positions to avoid initial overlaps
            offset = self._grid_spawn_offset(i)
            env._state["pose"][:2] = offset  # noqa: SLF001
            self._agent_poses[i] = env._state["pose"]  # noqa: SLF001
            self._obs_buf[i] = obs
            obs_list.append(self._augment_obs(i, obs))
            info_list.append(info)

        return obs_list, info_list

    def _grid_spawn_offset(self, agent_idx: int) -> np.ndarray:
        """Place agents in a regular grid to avoid spawn collisions."""
        cols = max(int(np.ceil(np.sqrt(self._n))), 1)
        row = agent_idx // cols
        col = agent_idx % cols
        spacing = 4.0
        x = (col - cols / 2) * spacing
        y = (row - cols / 2) * spacing
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[
        list[np.ndarray],
        list[float],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        """
        Execute one step for all agents.

        Parameters
        ----------
        actions:
            List of N action arrays, each of shape (ACT_DIM,).

        Returns
        -------
        (obs_list, rewards, terminateds, truncateds, infos)
        """
        assert len(actions) == self._n, f"Expected {self._n} actions, got {len(actions)}"
        self._step_count += 1

        raw_obs: list[np.ndarray] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict[str, Any]] = []

        for i, (env, action) in enumerate(zip(self._envs, actions)):
            if self._agent_done[i]:
                # Auto-reset on done
                obs, info = env.reset()
                self._agent_poses[i] = env._state["pose"]  # noqa: SLF001
                raw_obs.append(obs)
                rewards.append(0.0)
                terminateds.append(False)
                truncateds.append(False)
                infos.append(info)
                continue

            obs, rew, term, trunc, info = env.step(action)
            self._agent_poses[i] = np.array(info.get("pose", env._state["pose"][:6]))  # noqa: SLF001
            raw_obs.append(obs)
            rewards.append(float(rew))
            terminateds.append(bool(term))
            truncateds.append(bool(trunc))
            infos.append(info)
            self._agent_done[i] = term or trunc

            if term and info.get("goal_reached", False):
                self._episode_deliveries += 1
                self._delivery_steps.append(self._step_count)

        # Inter-agent collision penalties
        collision_penalties = self._compute_agent_collisions()
        for i in range(self._n):
            rewards[i] += collision_penalties[i]

        # Throughput-based shared reward
        throughput = self._compute_throughput()
        for i in range(self._n):
            rewards[i] = (1 - self._lambda) * rewards[i] + self._lambda * throughput

        # Augment observations with neighbour info
        aug_obs = [self._augment_obs(i, obs) for i, obs in enumerate(raw_obs)]
        self._obs_buf = raw_obs

        # Update info with multi-agent stats
        for i, info in enumerate(infos):
            info["agent_id"] = i
            info["n_collisions_with_agents"] = int(collision_penalties[i] < 0)
            info["episode_deliveries"] = self._episode_deliveries
            info["throughput_rate"] = throughput

        return aug_obs, rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # Inter-agent collision
    # ------------------------------------------------------------------

    def _compute_agent_collisions(self) -> list[float]:
        """Check all agent pairs for proximity collisions."""
        penalties = [0.0] * self._n
        for i in range(self._n):
            for j in range(i + 1, self._n):
                dist = float(np.linalg.norm(
                    self._agent_poses[i, :2] - self._agent_poses[j, :2]
                ))
                if dist < self._col_radius:
                    penalties[i] += _COLLISION_PENALTY
                    penalties[j] += _COLLISION_PENALTY
                    logger.debug(
                        "Agent collision: agents %d↔%d dist=%.2f m", i, j, dist
                    )
        return penalties

    # ------------------------------------------------------------------
    # Throughput reward
    # ------------------------------------------------------------------

    def _compute_throughput(self, window: int = 100) -> float:
        """
        Shared reward based on delivery throughput.

        Returns the number of deliveries completed in the last ``window``
        steps, normalised to [0, 1] per agent.
        """
        recent = sum(
            1 for s in self._delivery_steps
            if self._step_count - s < window
        )
        # Normalise: max expected = n_agents × window / 200 steps per delivery
        max_expected = max(self._n * window / 200.0, 1.0)
        normalised = min(recent / max_expected, 1.0)
        return float(normalised * 10.0)  # scale to reasonable reward magnitude

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def _augment_obs(self, agent_idx: int, base_obs: np.ndarray) -> np.ndarray:
        """
        Append relative positions and headings of K nearest neighbours.

        Extra dims: (K × 4) = [rel_x, rel_y, rel_dist, rel_heading] per neighbour.
        """
        my_pose = self._agent_poses[agent_idx]
        my_pos = my_pose[:2]
        my_yaw = float(my_pose[5])

        other_indices = [i for i in range(self._n) if i != agent_idx]
        distances = [
            (float(np.linalg.norm(self._agent_poses[j, :2] - my_pos)), j)
            for j in other_indices
        ]
        distances.sort()
        k_nearest = distances[:_K_NEAREST]

        extra = np.zeros(_K_NEAREST * 4, dtype=np.float32)
        for slot, (dist, j) in enumerate(k_nearest):
            rel = self._agent_poses[j, :2] - my_pos
            rel_angle = float(np.arctan2(rel[1], rel[0])) - my_yaw
            rel_angle = float(np.arctan2(np.sin(rel_angle), np.cos(rel_angle)))
            extra[slot * 4 : slot * 4 + 4] = [
                float(np.clip(rel[0] / 50.0, -1, 1)),
                float(np.clip(rel[1] / 50.0, -1, 1)),
                float(np.clip(dist / 50.0, 0, 1)),
                float(rel_angle / np.pi),
            ]

        return np.concatenate([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # CTDE helpers
    # ------------------------------------------------------------------

    def joint_obs(self) -> np.ndarray:
        """
        Return joint observation matrix for the centralised critic.

        Returns
        -------
        np.ndarray of shape (N, _AGENT_OBS_DIM)
        """
        return np.stack(
            [self._augment_obs(i, obs) for i, obs in enumerate(self._obs_buf)],
            axis=0,
        )

    def joint_state(self) -> dict[str, np.ndarray]:
        """
        Return the global state dict for state-based centralised critics.

        Includes all agent poses and the warehouse occupancy grid (mock).
        """
        return {
            "agent_poses": self._agent_poses.copy(),  # (N, 6)
            "agent_velocities": np.stack(
                [env._state["velocity"] for env in self._envs], axis=0  # noqa: SLF001
            ),
            "agent_fork_states": np.stack(
                [env._state["fork"] for env in self._envs], axis=0  # noqa: SLF001
            ),
            "episode_deliveries": np.array([self._episode_deliveries], dtype=np.float32),
            "step": np.array([self._step_count], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int) -> None:
        """Broadcast curriculum level to all sub-environments."""
        for env in self._envs:
            env.set_curriculum_level(level)
        logger.info("MultiAgentWarehouseEnv: curriculum level → %d for all agents.", level)

    def render(self) -> np.ndarray:
        """
        Return a top-down occupancy map with agent positions marked.
        """
        size = 400
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 50  # dark grey bg

        def world_to_px(xy: np.ndarray) -> tuple[int, int]:
            px = int((xy[0] + self._size) / (2 * self._size) * size)
            py = int((xy[1] + self._size) / (2 * self._size) * size)
            return (np.clip(px, 0, size - 1), np.clip(py, 0, size - 1))

        colours = [
            (0, 200, 255),
            (255, 100, 0),
            (0, 255, 100),
            (255, 0, 200),
            (200, 200, 0),
            (0, 100, 255),
        ]
        for i, pose in enumerate(self._agent_poses):
            px, py = world_to_px(pose[:2])
            colour = colours[i % len(colours)]
            # Draw agent as a filled circle
            r = 6
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        cx = np.clip(px + dx, 0, size - 1)
                        cy = np.clip(py + dy, 0, size - 1)
                        canvas[cy, cx] = colour
        return canvas

    def close(self) -> None:
        for env in self._envs:
            env.close()
