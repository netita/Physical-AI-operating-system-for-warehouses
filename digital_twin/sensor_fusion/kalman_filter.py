"""
kalman_filter.py — Extended Kalman Filter for warehouse object tracking.

State vector (8D)
-----------------
  x[0]  = x          world-frame position (m)
  x[1]  = y
  x[2]  = z
  x[3]  = vx         velocity (m/s)
  x[4]  = vy
  x[5]  = vz
  x[6]  = heading     yaw angle (rad)   -π … π
  x[7]  = w           angular velocity (rad/s)

Supported measurement models
-----------------------------
  camera_bbox   — (cx, cy, width, height) bounding box in image → position XY
  lidar         — (x, y, z) 3-D point from LiDAR
  odometry      — (x, y, z, vx, vy, heading, w) direct state subset

All matrices are pure numpy.  No external filtering library dependency.
"""

from __future__ import annotations

import math
import time

import numpy as np

# State dimension
_N = 8


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class WarehouseKalmanFilter:
    """
    Extended Kalman Filter for a single tracked agent.

    Parameters
    ----------
    dt : float
        Nominal prediction step (seconds).  Can be overridden per call.
    process_noise_std : float
        Standard deviation of process noise (isotropic acceleration noise).
    initial_state : np.ndarray | None
        8-element initial state vector.  Defaults to zero.
    initial_covariance : float
        Initial diagonal covariance value (uncertainty in all state dims).
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_std: float = 0.5,
        initial_state: np.ndarray | None = None,
        initial_covariance: float = 10.0,
    ) -> None:
        self.dt = dt
        self._process_noise_std = process_noise_std

        # State estimate
        self.x: np.ndarray = (
            np.zeros(_N, dtype=np.float64)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).copy()
        )

        # Covariance matrix
        self.P: np.ndarray = np.eye(_N, dtype=np.float64) * initial_covariance

        self._last_update_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # State transition (constant-velocity + constant-heading-rate)
    # ------------------------------------------------------------------

    def _F(self, dt: float) -> np.ndarray:
        """Jacobian of state transition (linearised around current state)."""
        F = np.eye(_N, dtype=np.float64)
        # position ← velocity * dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # heading ← w * dt
        F[6, 7] = dt
        return F

    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Non-linear state transition function."""
        x_new = x.copy()
        x_new[0] = x[0] + x[3] * dt
        x_new[1] = x[1] + x[4] * dt
        x_new[2] = x[2] + x[5] * dt
        # heading wraps
        x_new[6] = _wrap_angle(x[6] + x[7] * dt)
        return x_new

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete process noise covariance (piecewise constant acceleration)."""
        q = self._process_noise_std ** 2
        Q = np.zeros((_N, _N), dtype=np.float64)
        # Position noise (integrated from velocity noise)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q * dt ** 4 / 4
        Q[0, 3] = Q[3, 0] = q * dt ** 3 / 2
        Q[1, 4] = Q[4, 1] = q * dt ** 3 / 2
        Q[2, 5] = Q[5, 2] = q * dt ** 3 / 2
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q * dt ** 2
        # Heading noise
        Q[6, 6] = q * dt ** 2
        Q[7, 7] = q
        return Q

    # ------------------------------------------------------------------
    # Measurement models
    # ------------------------------------------------------------------

    # --- Camera bounding box (monocular depth estimate) ----------------

    _R_CAMERA = np.diag([0.5, 0.5]).astype(np.float64)   # (x, y) σ = 0.5 m

    def _h_camera(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: camera → (x, y) world coords."""
        return np.array([x[0], x[1]], dtype=np.float64)

    def _H_camera(self) -> np.ndarray:
        """Jacobian of camera measurement function w.r.t. state."""
        H = np.zeros((2, _N), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # --- LiDAR 3-D point -----------------------------------------------

    _R_LIDAR = np.diag([0.05, 0.05, 0.1]).astype(np.float64)  # σ = 5 cm XY, 10 cm Z

    def _h_lidar(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1], x[2]], dtype=np.float64)

    def _H_lidar(self) -> np.ndarray:
        H = np.zeros((3, _N), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    # --- Odometry (direct state subset) ---------------------------------

    _R_ODOM = np.diag([0.02, 0.02, 0.05, 0.05, 0.05, 0.02, 0.01]).astype(np.float64)

    def _h_odom(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [x[0], x[1], x[2], x[3], x[4], x[6], x[7]], dtype=np.float64
        )

    def _H_odom(self) -> np.ndarray:
        # rows: x, y, z, vx, vy, heading, w
        H = np.zeros((7, _N), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0   # position
        H[3, 3] = H[4, 4] = 1.0               # velocity XY
        H[5, 6] = 1.0                          # heading
        H[6, 7] = 1.0                          # angular velocity
        return H

    # ------------------------------------------------------------------
    # EKF steps
    # ------------------------------------------------------------------

    def predict(self, dt: float | None = None) -> "WarehouseKalmanFilter":
        """
        Propagate state and covariance forward by dt seconds.

        Returns self for method chaining.
        """
        dt = dt if dt is not None else self.dt
        F = self._F(dt)
        Q = self._Q(dt)

        self.x = self._f(self.x, dt)
        self.P = F @ self.P @ F.T + Q
        self._last_update_time = time.monotonic()
        return self

    def _ekf_update(
        self,
        z: np.ndarray,
        h_fn: "Callable[[np.ndarray], np.ndarray]",
        H: np.ndarray,
        R: np.ndarray,
        angle_idx: int | None = None,
    ) -> "WarehouseKalmanFilter":
        """Generic EKF measurement update step."""
        y = z - h_fn(self.x)

        # If there is an angle dimension, wrap the innovation
        if angle_idx is not None:
            y[angle_idx] = _wrap_angle(y[angle_idx])

        S = H @ self.P @ H.T + R          # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.x[6] = _wrap_angle(self.x[6])  # keep heading normalised

        I = np.eye(_N, dtype=np.float64)
        self.P = (I - K @ H) @ self.P     # Joseph form omitted for performance
        return self

    def update_camera(
        self,
        world_x: float,
        world_y: float,
        measurement_noise: np.ndarray | None = None,
    ) -> "WarehouseKalmanFilter":
        """
        Incorporate a camera-derived world-coordinate observation (x, y).

        Parameters
        ----------
        world_x, world_y : float
            Estimated world-frame position from mono-depth or stereo.
        measurement_noise : (2, 2) ndarray | None
            Override default measurement noise matrix.
        """
        z = np.array([world_x, world_y], dtype=np.float64)
        R = measurement_noise if measurement_noise is not None else self._R_CAMERA
        return self._ekf_update(z, self._h_camera, self._H_camera(), R)

    def update_lidar(
        self,
        x: float,
        y: float,
        z: float,
        measurement_noise: np.ndarray | None = None,
    ) -> "WarehouseKalmanFilter":
        """Incorporate a LiDAR 3-D point observation."""
        meas = np.array([x, y, z], dtype=np.float64)
        R = measurement_noise if measurement_noise is not None else self._R_LIDAR
        return self._ekf_update(meas, self._h_lidar, self._H_lidar(), R)

    def update_odometry(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        heading: float,
        angular_vel: float,
        measurement_noise: np.ndarray | None = None,
    ) -> "WarehouseKalmanFilter":
        """Incorporate an odometry observation."""
        meas = np.array([x, y, z, vx, vy, heading, angular_vel], dtype=np.float64)
        R = measurement_noise if measurement_noise is not None else self._R_ODOM
        return self._ekf_update(meas, self._h_odom, self._H_odom(), R, angle_idx=5)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state vector."""
        return self.x.copy()

    def get_position(self) -> tuple[float, float, float]:
        """Return (x, y, z) world-frame position."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_velocity(self) -> tuple[float, float, float]:
        """Return (vx, vy, vz) velocity."""
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    def get_heading(self) -> float:
        """Return heading angle (rad)."""
        return float(self.x[6])

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the covariance matrix."""
        return self.P.copy()

    def mahalanobis_distance(
        self, z: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> float:
        """
        Mahalanobis distance between measurement z and predicted measurement.
        Used for data association in the multi-object tracker.
        """
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.sqrt(y @ S_inv @ y))

    def __repr__(self) -> str:
        pos = self.get_position()
        return (
            f"WarehouseKalmanFilter("
            f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
            f"heading={math.degrees(self.get_heading()):.1f}°)"
        )
