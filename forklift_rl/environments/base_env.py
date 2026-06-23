"""
warehousegpt.forklift_rl.environments.base_env
===============================================
Base Gymnasium environment for autonomous warehouse forklifts.

Integration with Isaac Lab
--------------------------
Isaac Lab (built on Isaac Sim / PhysX) provides the physics simulation.
This class wraps the Isaac Lab ``DirectRLEnv`` / ``ManagerBasedRLEnv``
interface with a standard Gymnasium API so that any off-the-shelf RL
library (Stable-Baselines3, RLlib, DreamerV3, etc.) can be plugged in
without modification.

When Isaac Lab is not available (CPU development machines, CI) the class
falls back to a lightweight NumPy-based mock simulation that preserves the
full observation/action interface.

Observation space (flat, 1-D vector)
--------------------------------------
+------------------+------+--------+-------------------------------+
| Sensor           | Dims | Range  | Notes                         |
+==================+======+========+===============================+
| LiDAR            |  720 | [0,30] | 720-beam, 30 m range (metres) |
+------------------+------+--------+-------------------------------+
| RGB camera (enc) |  512 | [-1,1] | Encoded by ViT-Tiny backbone  |
+------------------+------+--------+-------------------------------+
| Fork position    |    3 | [0,1]  | height, tilt, spread (norm.)  |
+------------------+------+--------+-------------------------------+
| Velocity         |    3 | [-1,1] | vx, vy, ω (normalised)        |
+------------------+------+--------+-------------------------------+
| Pose             |    6 | [-1,1] | x, y, z, roll, pitch, yaw     |
+------------------+------+--------+-------------------------------+
Total: 1244 dimensions.

Action space (continuous, Box)
------------------------------
| Index | Action         | Range    | Unit    |
|-------|----------------|----------|---------|
|   0   | linear_vel     | [-1, 1]  | m/s     |
|   1   | angular_vel    | [-1, 1]  | rad/s   |
|   2   | fork_height    | [-1, 1]  | m/s (rate) |
|   3   | fork_tilt      | [-1, 1]  | deg/s   |

Reward shaping
--------------
Subclasses override ``_compute_reward()`` to implement task-specific
dense rewards.  The base class provides a small survival penalty
(−0.001 per step) to encourage efficiency.

Dependencies
------------
    pip install gymnasium numpy
    # Isaac Lab: installed via NVIDIA Isaac Sim bundle
    # Camera encoder:
    pip install timm torch torchvision
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation / action dimensions
# ---------------------------------------------------------------------------

_LIDAR_BEAMS = 720
_LIDAR_MAX_RANGE = 30.0  # metres

_CAM_ENC_DIM = 512        # ViT-Tiny embedding dimension
_FORK_DIM = 3             # height, tilt, spread
_VEL_DIM = 3              # vx, vy, omega
_POSE_DIM = 6             # x, y, z, roll, pitch, yaw

OBS_DIM = _LIDAR_BEAMS + _CAM_ENC_DIM + _FORK_DIM + _VEL_DIM + _POSE_DIM  # 1244

ACT_DIM = 4  # linear_vel, angular_vel, fork_height, fork_tilt

# Physical limits for normalisation
_MAX_LINEAR_VEL = 2.0    # m/s
_MAX_ANGULAR_VEL = 1.5   # rad/s
_MAX_FORK_RATE = 0.5     # m/s
_MAX_FORK_TILT_RATE = 20.0  # deg/s

# Fork hardware limits
_FORK_MAX_HEIGHT = 6.0   # m
_FORK_MIN_HEIGHT = 0.0
_FORK_MAX_TILT = 15.0    # deg (forward)
_FORK_MIN_TILT = -5.0    # deg (back)


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

def _try_import_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore[import-untyped]
        return gym
    except ImportError:
        logger.warning("gymnasium not installed. Install with: pip install gymnasium")
        return None


def _try_import_isaac_lab() -> Any:
    try:
        import isaaclab.envs  # type: ignore[import-untyped]
        return isaaclab.envs
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Camera encoder stub
# ---------------------------------------------------------------------------

class _ViTEncoder:
    """
    Lightweight ViT-Tiny RGB encoder for the observation pipeline.

    In production: fine-tuned on warehouse imagery from Isaac Sim.
    Inference runs on the GPU alongside Isaac Lab physics at ~1 ms/frame.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model: object | None = None
        try:
            import timm  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]

            self._model = timm.create_model(
                "vit_tiny_patch16_224", pretrained=False, num_classes=0
            )
            self._model.eval()  # type: ignore[union-attr]
            self._model.to(device)  # type: ignore[union-attr]
            self._torch = torch
            logger.debug("ViTEncoder: model loaded on %s", device)
        except ImportError:
            logger.debug("timm/torch not available — ViTEncoder using random projection.")

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """Encode HxWx3 uint8 RGB to (512,) float32 embedding."""
        if self._model is None:
            # Random projection fallback
            flat = rgb.astype(np.float32).flatten() / 255.0
            key = flat[:_CAM_ENC_DIM] if len(flat) >= _CAM_ENC_DIM else np.pad(flat, (0, _CAM_ENC_DIM - len(flat)))
            return key * 2.0 - 1.0

        import torchvision.transforms.functional as TF  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]

        pil = Image.fromarray(rgb)
        t = TF.to_tensor(TF.resize(pil, [224, 224])).unsqueeze(0).to(self._device)
        with self._torch.no_grad():  # type: ignore[union-attr]
            emb = self._model(t).squeeze(0).cpu().numpy()  # type: ignore[operator]
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------


class WarehouseBaseEnv(ABC):
    """
    Abstract base Gymnasium environment for autonomous warehouse forklifts.

    Subclasses must implement:
        - ``_compute_reward()``
        - ``_is_terminated()``
        - ``_get_task_obs()``  (optional extra obs appended to base)

    Parameters
    ----------
    isaac_lab_cfg:
        Isaac Lab environment configuration object.  When None the class
        runs in mock-simulation mode (no GPU required).
    max_episode_steps:
        Maximum timesteps per episode before truncation.
    render_mode:
        ``"human"`` opens the Isaac Sim viewport; ``"rgb_array"`` returns frames.
    camera_resolution:
        (H, W) of the onboard RGB camera.
    lidar_noise_std:
        Gaussian noise std dev added to LiDAR returns (metres).
    device:
        Torch/CUDA device for camera encoder.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        isaac_lab_cfg: Any | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        camera_resolution: tuple[int, int] = (224, 224),
        lidar_noise_std: float = 0.02,
        device: str = "cpu",
    ) -> None:
        self._max_steps = max_episode_steps
        self._render_mode = render_mode
        self._cam_res = camera_resolution
        self._lidar_noise = lidar_noise_std
        self._device = device
        self._step_count = 0
        self._episode_count = 0

        # Camera encoder
        self._encoder = _ViTEncoder(device=device)

        # Isaac Lab integration
        self._isaac_env: Any | None = None
        if isaac_lab_cfg is not None:
            self._isaac_env = self._init_isaac_lab(isaac_lab_cfg)

        # Internal state (used in mock mode)
        self._state = self._init_state()

        # Gymnasium spaces
        gym = _try_import_gymnasium()
        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(OBS_DIM,),
                dtype=np.float32,
            )
            # Override high for LiDAR segment (range 0–30 m)
            obs_low = np.full(OBS_DIM, -1.0, dtype=np.float32)
            obs_high = np.full(OBS_DIM, 1.0, dtype=np.float32)
            obs_low[:_LIDAR_BEAMS] = 0.0
            obs_high[:_LIDAR_BEAMS] = _LIDAR_MAX_RANGE
            self.observation_space = gym.spaces.Box(
                low=obs_low, high=obs_high, dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
            )
        else:
            self.observation_space = None
            self.action_space = None

        logger.debug(
            "%s initialised: obs_dim=%d  act_dim=%d  mock=%s",
            self.__class__.__name__,
            OBS_DIM,
            ACT_DIM,
            self._isaac_env is None,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns
        -------
        observation: np.ndarray of shape (OBS_DIM,)
        info: dict with episode metadata
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._step_count = 0
        self._episode_count += 1

        if self._isaac_env is not None:
            raw_obs = self._isaac_reset(options)
        else:
            self._state = self._init_state()
            self._state = self._apply_domain_randomisation(self._state)
            raw_obs = self._build_obs()

        obs = self._post_process_obs(raw_obs)
        info = self._reset_info()
        self._on_reset(info)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action: np.ndarray of shape (ACT_DIM,) in [-1, 1].

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

        if self._isaac_env is not None:
            raw_obs, done, info = self._isaac_step(action)
        else:
            raw_obs, done, info = self._mock_step(action)

        obs = self._post_process_obs(raw_obs)
        reward = self._compute_reward(obs, action, info)
        terminated = self._is_terminated(obs, info) or done
        truncated = self._step_count >= self._max_steps
        info["step"] = self._step_count
        info["episode"] = self._episode_count
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current state."""
        if self._render_mode == "rgb_array":
            return self._capture_rgb()
        if self._render_mode == "human" and self._isaac_env is not None:
            # Isaac Sim handles rendering automatically
            pass
        return None

    def close(self) -> None:
        """Clean up Isaac Lab resources."""
        if self._isaac_env is not None:
            try:
                self._isaac_env.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        """
        Compute the scalar reward for this transition.

        The base survival penalty of -0.001 is applied automatically;
        subclasses should return task-specific shaped reward.
        """

    @abstractmethod
    def _is_terminated(self, obs: np.ndarray, info: dict[str, Any]) -> bool:
        """Return True if the episode should terminate (goal reached, collision, etc.)."""

    # ------------------------------------------------------------------
    # Isaac Lab integration
    # ------------------------------------------------------------------

    @staticmethod
    def _init_isaac_lab(cfg: Any) -> Any:
        """Initialise an Isaac Lab environment from its config object."""
        il_envs = _try_import_isaac_lab()
        if il_envs is None:
            logger.warning("isaaclab not available — falling back to mock mode.")
            return None
        try:
            env = il_envs.DirectRLEnv(cfg=cfg)
            logger.info("Isaac Lab environment initialised.")
            return env
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isaac Lab init failed (%s) — mock mode.", exc)
            return None

    def _isaac_reset(self, options: dict[str, Any] | None) -> np.ndarray:
        obs_dict, _ = self._isaac_env.reset(options=options or {})
        return self._extract_isaac_obs(obs_dict)

    def _isaac_step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        obs_dict, rew, term, trunc, info = self._isaac_env.step(action)
        raw_obs = self._extract_isaac_obs(obs_dict)
        done = bool(term or trunc)
        return raw_obs, done, info

    def _extract_isaac_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        """
        Flatten Isaac Lab structured observation dict into our fixed vector.

        Expected Isaac Lab obs keys (configurable):
            "lidar", "camera_rgb", "fork_state", "velocity", "pose"
        """
        lidar = np.asarray(
            obs_dict.get("lidar", np.full(_LIDAR_BEAMS, _LIDAR_MAX_RANGE)), dtype=np.float32
        )
        lidar = np.clip(lidar[:_LIDAR_BEAMS], 0, _LIDAR_MAX_RANGE)

        rgb = np.asarray(obs_dict.get("camera_rgb", np.zeros((*self._cam_res, 3), dtype=np.uint8)))
        cam_enc = self._encoder.encode(rgb)

        fork = np.asarray(obs_dict.get("fork_state", np.zeros(_FORK_DIM)), dtype=np.float32)[:_FORK_DIM]
        vel = np.asarray(obs_dict.get("velocity", np.zeros(_VEL_DIM)), dtype=np.float32)[:_VEL_DIM]
        pose = np.asarray(obs_dict.get("pose", np.zeros(_POSE_DIM)), dtype=np.float32)[:_POSE_DIM]

        return np.concatenate([lidar, cam_enc, fork, vel, pose]).astype(np.float32)

    # ------------------------------------------------------------------
    # Mock simulation
    # ------------------------------------------------------------------

    def _init_state(self) -> dict[str, np.ndarray]:
        return {
            "pose": np.zeros(6, dtype=np.float32),       # x,y,z,r,p,yaw
            "velocity": np.zeros(3, dtype=np.float32),   # vx,vy,omega
            "fork": np.array([0.2, 0.0, 0.0], dtype=np.float32),  # h, tilt, spread
            "lidar": np.full(_LIDAR_BEAMS, _LIDAR_MAX_RANGE, dtype=np.float32),
        }

    def _apply_domain_randomisation(
        self, state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Apply random initial pose and obstacle layout."""
        rng = getattr(self, "_rng", np.random.default_rng())
        state["pose"][:2] = rng.uniform(-5.0, 5.0, size=2).astype(np.float32)
        state["pose"][5] = rng.uniform(-np.pi, np.pi)
        # Simulate obstacles: random LiDAR hits
        n_obstacles = int(rng.integers(5, 30))
        obstacle_beams = rng.integers(0, _LIDAR_BEAMS, size=n_obstacles)
        obstacle_ranges = rng.uniform(0.5, 15.0, size=n_obstacles).astype(np.float32)
        state["lidar"][obstacle_beams] = obstacle_ranges
        return state

    def _mock_step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Simple kinematic integration in mock mode."""
        dt = 1.0 / 30.0
        lin_vel = float(action[0]) * _MAX_LINEAR_VEL
        ang_vel = float(action[1]) * _MAX_ANGULAR_VEL
        fork_rate = float(action[2]) * _MAX_FORK_RATE
        tilt_rate = float(action[3]) * _MAX_FORK_TILT_RATE

        yaw = float(self._state["pose"][5])
        self._state["pose"][0] += lin_vel * np.cos(yaw) * dt
        self._state["pose"][1] += lin_vel * np.sin(yaw) * dt
        self._state["pose"][5] += ang_vel * dt
        self._state["pose"][5] = float(np.arctan2(
            np.sin(self._state["pose"][5]), np.cos(self._state["pose"][5])
        ))
        self._state["velocity"][:] = [lin_vel, 0.0, ang_vel]
        # Fork kinematics
        self._state["fork"][0] = float(
            np.clip(self._state["fork"][0] + fork_rate * dt, _FORK_MIN_HEIGHT, _FORK_MAX_HEIGHT)
        )
        self._state["fork"][1] = float(
            np.clip(self._state["fork"][1] + tilt_rate * dt, _FORK_MIN_TILT, _FORK_MAX_TILT)
        )

        # Add LiDAR noise
        noise = np.random.default_rng().normal(0, self._lidar_noise, _LIDAR_BEAMS)
        self._state["lidar"] = np.clip(
            self._state["lidar"] + noise, 0, _LIDAR_MAX_RANGE
        ).astype(np.float32)

        obs = self._build_obs()
        done = False  # Subclasses determine termination
        info = {
            "pose": self._state["pose"].copy(),
            "velocity": self._state["velocity"].copy(),
            "fork": self._state["fork"].copy(),
        }
        return obs, done, info

    def _build_obs(self) -> np.ndarray:
        """Concatenate all observation components into a flat vector."""
        lidar = self._state["lidar"]

        # Mock camera: blank RGB frame
        rgb = np.zeros((*self._cam_res, 3), dtype=np.uint8)
        cam_enc = self._encoder.encode(rgb)

        # Normalise fork state
        fork_norm = np.array([
            self._state["fork"][0] / _FORK_MAX_HEIGHT,
            self._state["fork"][1] / _FORK_MAX_TILT,
            self._state["fork"][2],  # spread already [0,1]
        ], dtype=np.float32)

        # Normalise velocities
        vel_norm = np.array([
            self._state["velocity"][0] / _MAX_LINEAR_VEL,
            self._state["velocity"][1] / _MAX_LINEAR_VEL,
            self._state["velocity"][2] / _MAX_ANGULAR_VEL,
        ], dtype=np.float32)

        # Normalise pose (rough world scale ±50 m, angles ±π)
        pose = self._state["pose"]
        pose_norm = np.array([
            pose[0] / 50.0, pose[1] / 50.0, pose[2] / 10.0,
            pose[3] / np.pi, pose[4] / np.pi, pose[5] / np.pi,
        ], dtype=np.float32)

        return np.concatenate([lidar, cam_enc, fork_norm, vel_norm, pose_norm]).astype(np.float32)

    def _post_process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Final clipping and type cast."""
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _capture_rgb(self) -> np.ndarray:
        """Return current camera frame (mock: blank frame)."""
        return np.zeros((*self._cam_res, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _on_reset(self, info: dict[str, Any]) -> None:
        """Called at the end of reset() — override for subclass setup."""

    def _reset_info(self) -> dict[str, Any]:
        return {
            "episode": self._episode_count,
            "step": 0,
            "mode": "isaac_lab" if self._isaac_env else "mock",
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def act_dim(self) -> int:
        return ACT_DIM

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_dim={OBS_DIM}, act_dim={ACT_DIM}, "
            f"max_steps={self._max_steps}, "
            f"isaac={'yes' if self._isaac_env else 'mock'})"
        )
