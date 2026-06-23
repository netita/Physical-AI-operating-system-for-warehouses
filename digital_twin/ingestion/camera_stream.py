"""
camera_stream.py — Real-time multi-camera frame ingestion.

Supports two back-ends:
  1. OpenCV VideoCapture (RTSP or file)
  2. GStreamer pipeline string (high-throughput, e.g. NVDEC decode)

Frames are pushed into a per-camera asyncio.Queue with configurable
backpressure.  A fan-out helper gathers multiple cameras concurrently
using asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum frames queued before the producer blocks (backpressure)
_DEFAULT_QUEUE_DEPTH = 4


@dataclass
class CameraFrame:
    """A single decoded frame from one camera."""

    camera_id: str
    frame_index: int
    timestamp: float          # wall-clock seconds (time.monotonic)
    image: np.ndarray         # HxWxC uint8 BGR
    source_url: str = ""


@dataclass
class CameraConfig:
    camera_id: str
    source_url: str           # rtsp://… or gstreamer pipeline string
    use_gstreamer: bool = False
    target_fps: float = 30.0
    queue_depth: int = _DEFAULT_QUEUE_DEPTH
    reconnect_delay: float = 2.0   # seconds between reconnect attempts


class _CameraWorker:
    """Owns a single VideoCapture and pushes frames onto a queue."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.queue: asyncio.Queue[CameraFrame] = asyncio.Queue(
            maxsize=config.queue_depth
        )
        self._stop_event = asyncio.Event()
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        cfg = self.config
        if cfg.use_gstreamer:
            # pipeline_str ends with "! appsink" — OpenCV reads it natively
            cap = cv2.VideoCapture(cfg.source_url, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(cfg.source_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError(
                f"[{cfg.camera_id}] Failed to open capture: {cfg.source_url}"
            )

        logger.info("[%s] Capture opened: %s", cfg.camera_id, cfg.source_url)
        return cap

    async def _read_loop(self, cap: cv2.VideoCapture) -> None:
        """Read frames from cap, push onto self.queue (with backpressure)."""
        loop = asyncio.get_running_loop()
        frame_interval = 1.0 / self.config.target_fps

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            # Offload blocking read to thread pool so the event loop is free
            ret, bgr = await loop.run_in_executor(None, cap.read)

            if not ret or bgr is None:
                logger.warning("[%s] Read failed — reconnecting", self.config.camera_id)
                break

            frame = CameraFrame(
                camera_id=self.config.camera_id,
                frame_index=self._frame_index,
                timestamp=time.monotonic(),
                image=bgr,
                source_url=self.config.source_url,
            )
            self._frame_index += 1

            # Backpressure: if queue is full, drop the oldest frame
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    logger.debug("[%s] Dropped stale frame (queue full)", self.config.camera_id)
                except asyncio.QueueEmpty:
                    pass

            await self.queue.put(frame)

            elapsed = time.monotonic() - t0
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def run(self) -> None:
        """Main loop: open capture, read frames, reconnect on failure."""
        while not self._stop_event.is_set():
            cap: cv2.VideoCapture | None = None
            try:
                cap = self._open_capture()
                await self._read_loop(cap)
            except Exception as exc:
                logger.error("[%s] Capture error: %s", self.config.camera_id, exc)
            finally:
                if cap is not None:
                    cap.release()

            if not self._stop_event.is_set():
                logger.info(
                    "[%s] Reconnecting in %.1fs …",
                    self.config.camera_id,
                    self.config.reconnect_delay,
                )
                await asyncio.sleep(self.config.reconnect_delay)

    def stop(self) -> None:
        self._stop_event.set()


class CameraStreamIngestion:
    """
    Manages multiple camera workers and exposes a unified async frame stream.

    Usage
    -----
    ingest = CameraStreamIngestion()
    ingest.connect_rtsp("rtsp://192.168.1.10/stream1", "dock_cam_1")
    ingest.connect_rtsp("rtsp://192.168.1.11/stream1", "aisle_cam_2")

    async for frame in ingest.stream_frames():
        process(frame)
    """

    def __init__(self) -> None:
        self._workers: dict[str, _CameraWorker] = {}
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect_rtsp(
        self,
        url: str,
        camera_id: str,
        target_fps: float = 30.0,
        queue_depth: int = _DEFAULT_QUEUE_DEPTH,
        reconnect_delay: float = 2.0,
    ) -> None:
        """Register an RTSP stream (decoded via FFmpeg inside OpenCV)."""
        if camera_id in self._workers:
            raise ValueError(f"Camera '{camera_id}' is already registered.")

        cfg = CameraConfig(
            camera_id=camera_id,
            source_url=url,
            use_gstreamer=False,
            target_fps=target_fps,
            queue_depth=queue_depth,
            reconnect_delay=reconnect_delay,
        )
        self._workers[camera_id] = _CameraWorker(cfg)
        logger.info("Registered RTSP camera '%s' → %s", camera_id, url)

    def connect_gstreamer(
        self,
        pipeline_str: str,
        camera_id: str | None = None,
        target_fps: float = 30.0,
        queue_depth: int = _DEFAULT_QUEUE_DEPTH,
        reconnect_delay: float = 2.0,
    ) -> None:
        """
        Register a GStreamer pipeline string (high-throughput, NVDEC-capable).

        Example pipeline_str::

            "rtspsrc location=rtsp://cam/stream ! rtph264depay ! \
             nvh264dec ! videoconvert ! video/x-raw,format=BGR ! appsink"
        """
        _id = camera_id or f"gst_cam_{len(self._workers)}"
        if _id in self._workers:
            raise ValueError(f"Camera '{_id}' is already registered.")

        cfg = CameraConfig(
            camera_id=_id,
            source_url=pipeline_str,
            use_gstreamer=True,
            target_fps=target_fps,
            queue_depth=queue_depth,
            reconnect_delay=reconnect_delay,
        )
        self._workers[_id] = _CameraWorker(cfg)
        logger.info("Registered GStreamer camera '%s'", _id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all camera workers as background tasks."""
        for camera_id, worker in self._workers.items():
            task = asyncio.create_task(worker.run(), name=f"cam_{camera_id}")
            self._tasks.append(task)
        logger.info("Started %d camera worker(s)", len(self._tasks))

    async def stop(self) -> None:
        """Signal all workers to stop and await task completion."""
        for worker in self._workers.values():
            worker.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("All camera workers stopped.")

    # ------------------------------------------------------------------
    # Frame streaming
    # ------------------------------------------------------------------

    async def stream_frames(
        self,
        camera_ids: list[str] | None = None,
        timeout: float = 5.0,
    ) -> AsyncGenerator[CameraFrame, None]:
        """
        Async generator that yields frames from one or all cameras,
        round-robined via asyncio.gather on per-camera queues.

        Parameters
        ----------
        camera_ids : list[str] | None
            Subset of camera IDs to merge.  None = all cameras.
        timeout : float
            Seconds to wait for the next frame before raising StopAsyncIteration.
        """
        ids = camera_ids or list(self._workers.keys())
        if not ids:
            raise RuntimeError("No cameras registered. Call connect_rtsp() first.")

        queues = [self._workers[cid].queue for cid in ids]

        async def _pull(q: asyncio.Queue[CameraFrame]) -> CameraFrame:
            return await asyncio.wait_for(q.get(), timeout=timeout)

        while True:
            # Fan-out: pull one frame from EVERY camera concurrently,
            # then yield them all before going round again.
            results = await asyncio.gather(
                *[_pull(q) for q in queues], return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Frame pull exception: %s", result)
                    continue
                yield result

    async def stream_camera(
        self,
        camera_id: str,
        timeout: float = 5.0,
    ) -> AsyncGenerator[CameraFrame, None]:
        """Yield frames from a single named camera."""
        if camera_id not in self._workers:
            raise KeyError(f"Unknown camera '{camera_id}'.")
        q = self._workers[camera_id].queue
        while True:
            frame = await asyncio.wait_for(q.get(), timeout=timeout)
            yield frame

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CameraStreamIngestion":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
