"""
ros2_bridge.py — ROS 2 ↔ Digital Twin bridge.

Subscribes to standard ROS 2 topics, converts messages to numpy arrays,
and publishes digital-twin state back to /warehouse/twin/* topics.

Topics consumed
---------------
/camera/<id>/image_raw       — sensor_msgs/Image
/scan                        — sensor_msgs/LaserScan
/odom                        — nav_msgs/Odometry
/tf                          — tf2_msgs/TFMessage

Topics published
----------------
/warehouse/twin/state        — std_msgs/String  (JSON)
/warehouse/twin/bev          — sensor_msgs/Image (BEV map)
/warehouse/twin/incidents    — std_msgs/String  (JSON array)

The bridge runs as an rclpy node inside a dedicated asyncio-compatible
executor.  When rclpy is not available (e.g. dev/test environment) the
module still imports cleanly — the bridge will raise at instantiation
time with a clear error message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rclpy import — graceful degradation in non-ROS environments
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False
    logger.warning(
        "rclpy not found — ROS2Bridge will raise RuntimeError at instantiation. "
        "Install ROS 2 Humble and source /opt/ros/humble/setup.bash to enable."
    )

# Try to import message types
try:
    from sensor_msgs.msg import Image, LaserScan
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import String
    _HAS_ROS_MSGS = True
except ImportError:
    _HAS_ROS_MSGS = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LidarScan:
    """Converted LaserScan in polar + Cartesian."""
    timestamp: float
    ranges: np.ndarray          # (N,) float32
    angles: np.ndarray          # (N,) float32  radians
    points_xy: np.ndarray       # (N, 2) float32 Cartesian XY in sensor frame


@dataclass
class OdomPose:
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    vx: float
    vy: float
    vyaw: float


@dataclass
class CameraImage:
    timestamp: float
    camera_id: str
    image: np.ndarray           # HxWxC uint8 (BGR)


@dataclass
class TFTransform:
    timestamp: float
    parent_frame: str
    child_frame: str
    translation: np.ndarray     # (3,) float64
    rotation: np.ndarray        # (4,) float64 quaternion xyzw


# ---------------------------------------------------------------------------
# Conversion utilities (standalone, usable without rclpy)
# ---------------------------------------------------------------------------

def ros_image_to_numpy(msg: Any) -> np.ndarray:
    """Convert sensor_msgs/Image to HxWxC uint8 numpy array."""
    encoding = msg.encoding.lower()
    dtype = np.uint8

    raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)

    if encoding in ("rgb8", "rgb"):
        # Flip to BGR for OpenCV convention
        return raw[:, :, ::-1]
    if encoding in ("bgr8", "bgr"):
        return raw
    if encoding == "mono8":
        return raw
    if encoding == "32fc1":
        raw_f = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        # Normalise to 0–255 for display
        norm = cv2_normalise(raw_f)
        return norm

    # Fallback: return raw bytes reshaped
    return raw


def cv2_normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise float array to uint8."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def ros_scan_to_lidar(msg: Any) -> LidarScan:
    """Convert sensor_msgs/LaserScan to LidarScan."""
    n = len(msg.ranges)
    angles = np.linspace(msg.angle_min, msg.angle_max, n, dtype=np.float32)
    ranges = np.array(msg.ranges, dtype=np.float32)

    # Mask invalid readings
    valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
    ranges[~valid] = np.nan

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points_xy = np.stack([x, y], axis=1)  # (N, 2)

    return LidarScan(
        timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        ranges=ranges,
        angles=angles,
        points_xy=points_xy,
    )


def ros_odom_to_pose(msg: Any) -> OdomPose:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    t = msg.twist.twist
    return OdomPose(
        timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        x=p.x, y=p.y, z=p.z,
        qx=q.x, qy=q.y, qz=q.z, qw=q.w,
        vx=t.linear.x,
        vy=t.linear.y,
        vyaw=t.angular.z,
    )


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

_SENSOR_QOS = None  # initialised below when rclpy is available


def _make_sensor_qos() -> "QoSProfile":
    return QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class _WarehouseTwinNode:
    """
    Wraps an rclpy Node to provide typed callbacks and publishing helpers.

    This class is instantiated on a background thread that spins the ROS
    executor.  Results are delivered to the asyncio side via thread-safe
    callbacks registered through on_camera / on_scan / on_odom / on_tf.
    """

    def __init__(
        self,
        node_name: str,
        camera_topic_prefix: str,
        scan_topic: str,
        odom_topic: str,
        tf_topic: str,
    ) -> None:
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy is not available. Source a ROS 2 workspace.")
        if not _HAS_ROS_MSGS:
            raise RuntimeError("ROS 2 message packages are not installed.")

        self._node: Node = rclpy.create_node(node_name)
        qos = _make_sensor_qos()

        # Subscribers
        self._node.create_subscription(
            LaserScan, scan_topic, self._on_scan, qos
        )
        self._node.create_subscription(
            Odometry, odom_topic, self._on_odom, qos
        )
        self._node.create_subscription(
            TFMessage, tf_topic, self._on_tf, 100
        )

        # We subscribe to /camera/<id>/image_raw by discovering topics at runtime
        self._camera_prefix = camera_topic_prefix
        self._camera_subs: list[Any] = []

        # Publishers
        self._pub_state = self._node.create_publisher(
            String, "/warehouse/twin/state", 10
        )
        self._pub_incidents = self._node.create_publisher(
            String, "/warehouse/twin/incidents", 10
        )
        self._pub_bev = self._node.create_publisher(
            Image, "/warehouse/twin/bev", qos
        )

        # User-registered callbacks
        self._camera_cbs: list[Callable[[CameraImage], None]] = []
        self._scan_cbs: list[Callable[[LidarScan], None]] = []
        self._odom_cbs: list[Callable[[OdomPose], None]] = []
        self._tf_cbs: list[Callable[[TFTransform], None]] = []

        self._node.get_logger().info(f"WarehouseTwinNode '{node_name}' ready.")

    # ------------------------------------------------------------------
    # ROS callbacks (execute on executor thread)
    # ------------------------------------------------------------------

    def _on_scan(self, msg: Any) -> None:
        scan = ros_scan_to_lidar(msg)
        for cb in self._scan_cbs:
            cb(scan)

    def _on_odom(self, msg: Any) -> None:
        pose = ros_odom_to_pose(msg)
        for cb in self._odom_cbs:
            cb(pose)

    def _on_tf(self, msg: Any) -> None:
        for transform in msg.transforms:
            t = transform.transform
            tf = TFTransform(
                timestamp=transform.header.stamp.sec
                + transform.header.stamp.nanosec * 1e-9,
                parent_frame=transform.header.frame_id,
                child_frame=transform.child_frame_id,
                translation=np.array(
                    [t.translation.x, t.translation.y, t.translation.z],
                    dtype=np.float64,
                ),
                rotation=np.array(
                    [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                    dtype=np.float64,
                ),
            )
            for cb in self._tf_cbs:
                cb(tf)

    def _make_camera_cb(self, camera_id: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            img = CameraImage(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                camera_id=camera_id,
                image=ros_image_to_numpy(msg),
            )
            for user_cb in self._camera_cbs:
                user_cb(img)
        return _cb

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_camera(self, cb: Callable[[CameraImage], None]) -> None:
        self._camera_cbs.append(cb)

    def on_scan(self, cb: Callable[[LidarScan], None]) -> None:
        self._scan_cbs.append(cb)

    def on_odom(self, cb: Callable[[OdomPose], None]) -> None:
        self._odom_cbs.append(cb)

    def on_tf(self, cb: Callable[[TFTransform], None]) -> None:
        self._tf_cbs.append(cb)

    # ------------------------------------------------------------------
    # Camera topic discovery
    # ------------------------------------------------------------------

    def discover_cameras(self) -> list[str]:
        """Return topic names matching the camera prefix."""
        names_and_types = self._node.get_topic_names_and_types()
        camera_topics = [
            name for name, _ in names_and_types
            if name.startswith(self._camera_prefix)
            and name.endswith("/image_raw")
        ]
        return camera_topics

    def subscribe_camera(self, topic: str, camera_id: str) -> None:
        qos = _make_sensor_qos()
        sub = self._node.create_subscription(
            Image, topic, self._make_camera_cb(camera_id), qos
        )
        self._camera_subs.append(sub)
        logger.info("Subscribed to camera topic '%s' as '%s'", topic, camera_id)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_state(self, state_json: str) -> None:
        msg = String()
        msg.data = state_json
        self._pub_state.publish(msg)

    def publish_incidents(self, incidents_json: str) -> None:
        msg = String()
        msg.data = incidents_json
        self._pub_incidents.publish(msg)

    def publish_bev(self, bev_image: np.ndarray) -> None:
        """Publish a BGR numpy array as sensor_msgs/Image."""
        msg = Image()
        h, w, c = bev_image.shape
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.step = w * c
        msg.data = bev_image.tobytes()
        self._pub_bev.publish(msg)

    @property
    def node(self) -> "Node":
        return self._node


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ROS2Bridge:
    """
    High-level bridge between ROS 2 and the digital twin.

    Spins an rclpy executor on a background daemon thread so the ROS
    callbacks are non-blocking from asyncio's perspective.  Delivers
    data to asyncio coroutines via asyncio.Queue.

    Example
    -------
    bridge = ROS2Bridge()
    bridge.start()

    async def consume():
        async for scan in bridge.scan_stream():
            process(scan)
    """

    def __init__(
        self,
        node_name: str = "warehouse_twin_node",
        camera_topic_prefix: str = "/camera",
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        tf_topic: str = "/tf",
        queue_depth: int = 20,
    ) -> None:
        if not _HAS_RCLPY:
            raise RuntimeError(
                "rclpy is not installed. "
                "Source /opt/ros/humble/setup.bash and ensure rclpy is on PYTHONPATH."
            )

        rclpy.init()
        self._ros_node = _WarehouseTwinNode(
            node_name=node_name,
            camera_topic_prefix=camera_topic_prefix,
            scan_topic=scan_topic,
            odom_topic=odom_topic,
            tf_topic=tf_topic,
        )

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._ros_node.node)

        # asyncio queues (filled from ROS callbacks via thread-safe put_nowait)
        self._scan_q: asyncio.Queue[LidarScan] = asyncio.Queue(maxsize=queue_depth)
        self._odom_q: asyncio.Queue[OdomPose] = asyncio.Queue(maxsize=queue_depth)
        self._tf_q: asyncio.Queue[TFTransform] = asyncio.Queue(maxsize=queue_depth)
        self._camera_q: asyncio.Queue[CameraImage] = asyncio.Queue(maxsize=queue_depth)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start spinning the ROS executor on a background thread."""
        self._loop = asyncio.get_event_loop()

        # Register callbacks that forward to asyncio queues
        def _safe_put(q: asyncio.Queue[Any], item: Any) -> None:
            if self._loop and not q.full():
                self._loop.call_soon_threadsafe(q.put_nowait, item)

        self._ros_node.on_scan(lambda s: _safe_put(self._scan_q, s))
        self._ros_node.on_odom(lambda p: _safe_put(self._odom_q, p))
        self._ros_node.on_tf(lambda t: _safe_put(self._tf_q, t))
        self._ros_node.on_camera(lambda c: _safe_put(self._camera_q, c))

        # Auto-discover and subscribe to all camera topics
        for topic in self._ros_node.discover_cameras():
            cam_id = topic.split("/")[2] if topic.count("/") >= 2 else topic
            self._ros_node.subscribe_camera(topic, cam_id)

        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin, daemon=True, name="ros2-spin"
        )
        self._spin_thread.start()
        logger.info("ROS2Bridge started — spinning executor on background thread.")

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:
            logger.error("ROS executor error: %s", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown()
        if self._spin_thread:
            self._spin_thread.join(timeout=5.0)
        rclpy.shutdown()
        logger.info("ROS2Bridge stopped.")

    # ------------------------------------------------------------------
    # Async stream generators
    # ------------------------------------------------------------------

    async def scan_stream(
        self, timeout: float = 2.0
    ) -> "AsyncGenerator[LidarScan, None]":
        while self._running:
            try:
                scan = await asyncio.wait_for(self._scan_q.get(), timeout=timeout)
                yield scan
            except TimeoutError:
                pass

    async def odom_stream(
        self, timeout: float = 2.0
    ) -> "AsyncGenerator[OdomPose, None]":
        while self._running:
            try:
                pose = await asyncio.wait_for(self._odom_q.get(), timeout=timeout)
                yield pose
            except TimeoutError:
                pass

    async def tf_stream(
        self, timeout: float = 2.0
    ) -> "AsyncGenerator[TFTransform, None]":
        while self._running:
            try:
                tf = await asyncio.wait_for(self._tf_q.get(), timeout=timeout)
                yield tf
            except TimeoutError:
                pass

    async def camera_stream(
        self, timeout: float = 2.0
    ) -> "AsyncGenerator[CameraImage, None]":
        while self._running:
            try:
                img = await asyncio.wait_for(self._camera_q.get(), timeout=timeout)
                yield img
            except TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Publishing helpers (call from asyncio; thread-safe)
    # ------------------------------------------------------------------

    def publish_state(self, state_dict: dict[str, Any]) -> None:
        self._ros_node.publish_state(json.dumps(state_dict))

    def publish_incidents(self, incidents: list[dict[str, Any]]) -> None:
        self._ros_node.publish_incidents(json.dumps(incidents))

    def publish_bev(self, bev: np.ndarray) -> None:
        self._ros_node.publish_bev(bev)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ROS2Bridge":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
