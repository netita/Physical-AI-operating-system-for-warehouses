"""
Camera Manager — NVIDIA Isaac Sim

Places and configures cameras of multiple types across the warehouse stage:
  - Overhead fisheye (ceiling grid)
  - End-of-aisle rack-mounted cameras
  - Forklift ego-perspective (mast-mounted)
  - Structured-light depth sensor

All cameras are registered with omni.isaac.sensor and can yield synchronised
RGB, depth and semantic segmentation outputs.

Usage:
    from isaac_sim.camera_placement import CameraManager
    mgr = CameraManager(stage, warehouse_cfg)
    mgr.place_overhead_cameras(grid_spacing=10.0)
    mgr.place_rack_cameras()
    mgr.configure_depth_sensor()
    frames = mgr.get_camera_data()   # dict[camera_id] → CameraFrame
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physx import RotatingLidarPhysX
from pxr import Gf, Usd, UsdGeom, UsdLux

# Replicator annotators for segmentation + depth
import omni.replicator.core as rep


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraFrame:
    """Single captured frame from one camera."""
    camera_id: str
    rgb: np.ndarray            # shape (H, W, 3) uint8
    depth: np.ndarray          # shape (H, W)   float32, metres
    segmentation: np.ndarray   # shape (H, W)   int32 semantic IDs
    intrinsics: np.ndarray     # 3×3 camera matrix
    extrinsics: np.ndarray     # 4×4 world-to-camera transform
    timestamp: float           # simulation time in seconds


@dataclass
class CameraSpec:
    """Specification for a single camera."""
    camera_id: str
    prim_path: str
    resolution: Tuple[int, int] = (1920, 1080)
    focal_length_mm: float = 3.5
    horizontal_aperture_mm: float = 6.0
    clipping_range: Tuple[float, float] = (0.01, 200.0)
    camera_type: str = "rgb"       # "rgb" | "fisheye" | "depth" | "structured_light"
    fisheye_fov_deg: float = 180.0


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Manages the full camera rig for a warehouse scene.

    Parameters
    ----------
    stage : Usd.Stage
        The active USD stage.
    warehouse_path : str
        Root prim of the warehouse (e.g. ``/World/Warehouse``).
    floor_length : float
        Warehouse floor length in metres (X).
    floor_width : float
        Warehouse floor width in metres (Y).
    ceiling_height : float
        Height of the ceiling in metres (Z).
    """

    CAMERA_ROOT = "/World/Cameras"

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        warehouse_path: str = "/World/Warehouse",
        floor_length: float = 120.0,
        floor_width: float = 60.0,
        ceiling_height: float = 9.0,
    ) -> None:
        self._stage = stage or get_current_stage()
        self._warehouse_path = warehouse_path
        self._L = floor_length
        self._W = floor_width
        self._H = ceiling_height

        self._camera_specs: Dict[str, CameraSpec] = {}
        self._isaac_cameras: Dict[str, Camera] = {}
        self._rep_render_products: Dict[str, object] = {}

        self._ensure_camera_root()

    # ------------------------------------------------------------------
    # Camera placement methods
    # ------------------------------------------------------------------

    def place_overhead_cameras(
        self,
        grid_spacing: float = 10.0,
        resolution: Tuple[int, int] = (1280, 720),
        mount_height_offset: float = 0.3,
        max_cameras: int = 8,
    ) -> List[str]:
        """
        Place fisheye cameras on the ceiling in a regular grid.

        Parameters
        ----------
        grid_spacing : float
            Centre-to-centre distance between cameras in metres.
        resolution : tuple
            Output image resolution (W, H) pixels.
        mount_height_offset : float
            Gap between ceiling surface and camera centre (m).

        Returns
        -------
        List[str]
            Camera IDs of all placed overhead cameras.
        """
        placed: List[str] = []
        z = self._H - mount_height_offset

        x = grid_spacing / 2.0
        col = 0
        while x < self._L and len(placed) < max_cameras:
            y = grid_spacing / 2.0
            row = 0
            while y < self._W and len(placed) < max_cameras:
                cam_id = f"overhead_{col:02d}_{row:02d}"
                prim_path = f"{self.CAMERA_ROOT}/Overhead/{cam_id}"

                self._create_camera_prim(
                    cam_id=cam_id,
                    prim_path=prim_path,
                    position=(x, y, z),
                    orientation_xyz_deg=(180.0, 0.0, 0.0),   # point downward
                    resolution=resolution,
                    focal_length_mm=1.8,       # ultra-wide fisheye
                    horizontal_aperture_mm=5.6,
                    camera_type="fisheye",
                    fisheye_fov_deg=185.0,
                )
                placed.append(cam_id)
                y += grid_spacing
                row += 1
            x += grid_spacing
            col += 1

        return placed

    def place_rack_cameras(
        self,
        aisle_x_positions: Optional[List[float]] = None,
        resolution: Tuple[int, int] = (1280, 720),
        mount_height: float = 3.5,
        max_cameras: int = 4,
    ) -> List[str]:
        """
        Place wide-angle cameras at rack ends facing down each aisle.

        Parameters
        ----------
        aisle_x_positions : list[float], optional
            X positions of aisle centre lines. If None, auto-infers from
            warehouse config (every 6 m from edge).
        resolution : tuple
            Output resolution (W, H) pixels.
        mount_height : float
            Camera height in metres.

        Returns
        -------
        List[str]
            Camera IDs.
        """
        if aisle_x_positions is None:
            # Auto-place at every 6 m — rough single/double aisle heuristic
            aisle_x_positions = list(
                np.arange(6.0, self._L, 6.0)
            )

        placed: List[str] = []
        for i, ax in enumerate(aisle_x_positions):
            if len(placed) >= max_cameras:
                break
            for end_idx, (y_pos, yaw) in enumerate(
                [(2.0, 90.0), (self._W - 2.0, 270.0)]
            ):
                if len(placed) >= max_cameras:
                    break
                cam_id = f"rack_aisle{i:02d}_end{end_idx}"
                prim_path = f"{self.CAMERA_ROOT}/Rack/{cam_id}"
                self._create_camera_prim(
                    cam_id=cam_id,
                    prim_path=prim_path,
                    position=(ax, y_pos, mount_height),
                    orientation_xyz_deg=(0.0, 0.0, yaw),
                    resolution=resolution,
                    focal_length_mm=2.8,
                    horizontal_aperture_mm=6.35,
                    camera_type="rgb",
                )
                placed.append(cam_id)
        return placed

    def place_forklift_cameras(
        self,
        forklift_prim_paths: Optional[List[str]] = None,
        resolution: Tuple[int, int] = (1280, 720),
    ) -> List[str]:
        """
        Attach ego-perspective cameras to the mast of each forklift prim.

        Parameters
        ----------
        forklift_prim_paths : list[str], optional
            USD paths to forklift root prims. Auto-discovers from
            /World/Actors/Forklifts if not provided.
        resolution : tuple
            Output resolution.

        Returns
        -------
        List[str]
            Camera IDs.
        """
        if forklift_prim_paths is None:
            forklift_prim_paths = self._discover_forklifts()

        placed: List[str] = []
        for i, fl_path in enumerate(forklift_prim_paths):
            fl_prim = self._stage.GetPrimAtPath(fl_path)
            if not fl_prim.IsValid():
                continue

            cam_id = f"forklift_{i:03d}_ego"
            prim_path = f"{fl_path}/MastCamera"

            # Mount at top of mast: approximately 3.2 m high, 0.5 m forward
            self._create_camera_prim(
                cam_id=cam_id,
                prim_path=prim_path,
                position=(0.5, 0.0, 3.2),     # local to forklift
                orientation_xyz_deg=(-10.0, 0.0, 0.0),
                resolution=resolution,
                focal_length_mm=4.0,
                horizontal_aperture_mm=6.35,
                camera_type="rgb",
            )
            placed.append(cam_id)
        return placed

    def configure_depth_sensor(
        self,
        positions: Optional[List[Tuple[float, float, float]]] = None,
        pattern: str = "structured_light",
        resolution: Tuple[int, int] = (848, 480),
        range_metres: Tuple[float, float] = (0.15, 10.0),
    ) -> List[str]:
        """
        Place depth sensors (structured-light style) at given positions.

        Parameters
        ----------
        positions : list[tuple], optional
            World positions (x, y, z). Defaults to dock area placements.
        pattern : str
            Sensor pattern: ``"structured_light"`` or ``"tof"``.
        resolution : tuple
            Depth map resolution.
        range_metres : tuple
            (min, max) depth range in metres.

        Returns
        -------
        List[str]
            Camera IDs for depth sensors.
        """
        if positions is None:
            # Default: place at loading dock area
            positions = [
                (self._L - 2.0, y, 2.5)
                for y in np.linspace(5.0, self._W - 5.0, 4)
            ]

        placed: List[str] = []
        for i, pos in enumerate(positions):
            cam_id = f"depth_sensor_{i:03d}"
            prim_path = f"{self.CAMERA_ROOT}/Depth/{cam_id}"

            spec = self._create_camera_prim(
                cam_id=cam_id,
                prim_path=prim_path,
                position=pos,
                orientation_xyz_deg=(0.0, -30.0, 0.0),
                resolution=resolution,
                focal_length_mm=3.5,
                horizontal_aperture_mm=6.35,
                camera_type=pattern,
                clipping_range=range_metres,
            )
            placed.append(cam_id)
        return placed

    # ------------------------------------------------------------------
    # Data capture
    # ------------------------------------------------------------------

    def get_camera_data(
        self, camera_ids: Optional[List[str]] = None
    ) -> Dict[str, CameraFrame]:
        """
        Read current RGB, depth, and semantic segmentation from all (or
        selected) cameras.  Assumes the simulation has been stepped and
        Replicator render products are ready.

        Parameters
        ----------
        camera_ids : list[str], optional
            Subset of cameras to read. Reads all if None.

        Returns
        -------
        dict[str, CameraFrame]
        """
        ids = camera_ids if camera_ids is not None else list(self._camera_specs.keys())
        frames: Dict[str, CameraFrame] = {}

        for cam_id in ids:
            spec = self._camera_specs.get(cam_id)
            if spec is None:
                continue

            rgb, depth, seg = self._read_annotators(cam_id, spec)
            K = self._compute_intrinsics(spec)
            E = self._read_extrinsics(spec.prim_path)

            frames[cam_id] = CameraFrame(
                camera_id=cam_id,
                rgb=rgb,
                depth=depth,
                segmentation=seg,
                intrinsics=K,
                extrinsics=E,
                timestamp=self._get_sim_time(),
            )

        return frames

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_camera_prim(
        self,
        cam_id: str,
        prim_path: str,
        position: Tuple[float, float, float],
        orientation_xyz_deg: Tuple[float, float, float],
        resolution: Tuple[int, int],
        focal_length_mm: float,
        horizontal_aperture_mm: float,
        camera_type: str = "rgb",
        clipping_range: Tuple[float, float] = (0.01, 200.0),
        fisheye_fov_deg: float = 180.0,
    ) -> CameraSpec:
        stage = self._stage

        # Ensure parent scope exists
        parent = "/".join(prim_path.split("/")[:-1])
        if not stage.GetPrimAtPath(parent).IsValid():
            UsdGeom.Scope.Define(stage, parent)

        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.CreateFocalLengthAttr(focal_length_mm)
        camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range))

        if camera_type == "fisheye":
            # OmniKit fisheye projection token
            camera.CreateProjectionAttr("fisheye")
            # Some Kit builds use a custom attribute for FOV
            prim = stage.GetPrimAtPath(prim_path)
            prim.CreateAttribute(
                "omni:camera:fisheyeFov", omni.usd.get_context().get_stage().GetAttributeDefinition(None) if False else None
            ) if False else None
            # Set via direct attribute (supported in Isaac Sim 4.x+)
            cam_prim = stage.GetPrimAtPath(prim_path)
            if cam_prim.IsValid():
                fov_attr = cam_prim.GetAttribute("omni:camera:fisheyeFov")
                if fov_attr:
                    fov_attr.Set(fisheye_fov_deg)

        cam_prim = stage.GetPrimAtPath(prim_path)
        if cam_prim.IsValid():
            xf = UsdGeom.Xformable(cam_prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            rx, ry, rz = orientation_xyz_deg
            xf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

        spec = CameraSpec(
            camera_id=cam_id,
            prim_path=prim_path,
            resolution=resolution,
            focal_length_mm=focal_length_mm,
            horizontal_aperture_mm=horizontal_aperture_mm,
            clipping_range=clipping_range,
            camera_type=camera_type,
            fisheye_fov_deg=fisheye_fov_deg,
        )
        self._camera_specs[cam_id] = spec

        # Register Replicator render product
        try:
            render_product = rep.create.render_product(
                prim_path, resolution=list(resolution)
            )
            self._rep_render_products[cam_id] = render_product
        except Exception:
            pass  # Replicator may not be available in non-interactive mode

        return spec

    def _read_annotators(
        self, cam_id: str, spec: CameraSpec
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read RGB, depth and semantic segmentation from Replicator annotators.
        Returns zeros arrays on failure (graceful degradation).
        """
        W, H = spec.resolution
        rgb_default = np.zeros((H, W, 3), dtype=np.uint8)
        depth_default = np.zeros((H, W), dtype=np.float32)
        seg_default = np.zeros((H, W), dtype=np.int32)

        rp = self._rep_render_products.get(cam_id)
        if rp is None:
            return rgb_default, depth_default, seg_default

        try:
            # RGB
            rgb_ann = rep.annotators.get("rgb")
            rgb_ann.attach([rp])
            rgb_data = rgb_ann.get_data()
            rgb_arr = np.frombuffer(rgb_data["data"], dtype=np.uint8).reshape(H, W, 4)
            rgb = rgb_arr[:, :, :3]

            # Linear depth
            depth_ann = rep.annotators.get("distance_to_image_plane")
            depth_ann.attach([rp])
            depth_data = depth_ann.get_data()
            depth = np.frombuffer(depth_data["data"], dtype=np.float32).reshape(H, W)

            # Semantic segmentation
            seg_ann = rep.annotators.get("semantic_segmentation")
            seg_ann.attach([rp])
            seg_data = seg_ann.get_data()
            seg = seg_data["data"].reshape(H, W).astype(np.int32)

            return rgb, depth, seg
        except Exception:
            return rgb_default, depth_default, seg_default

    def _compute_intrinsics(self, spec: CameraSpec) -> np.ndarray:
        """Build 3×3 camera intrinsics matrix from USD camera attributes."""
        W, H = spec.resolution
        f_px = spec.focal_length_mm * W / spec.horizontal_aperture_mm
        cx = W / 2.0
        cy = H / 2.0
        K = np.array([
            [f_px, 0.0,  cx],
            [0.0,  f_px, cy],
            [0.0,  0.0,  1.0],
        ], dtype=np.float64)
        return K

    def _read_extrinsics(self, prim_path: str) -> np.ndarray:
        """Read 4×4 world-to-camera transform from USD xform stack."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return np.eye(4, dtype=np.float64)
        xf = UsdGeom.Xformable(prim)
        transform = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Convert GfMatrix4d to numpy
        mat = np.array([[transform[r][c] for c in range(4)] for r in range(4)],
                       dtype=np.float64)
        return np.linalg.inv(mat)   # world-to-camera

    def _get_sim_time(self) -> float:
        try:
            from isaacsim.core.api import World
            return World.instance().current_time
        except Exception:
            return 0.0

    def _discover_forklifts(self) -> List[str]:
        """Auto-discover forklift prims under /World/Actors/Forklifts."""
        scope_prim = self._stage.GetPrimAtPath("/World/Actors/Forklifts")
        if not scope_prim.IsValid():
            return []
        return [str(child.GetPath()) for child in scope_prim.GetChildren()]

    def _ensure_camera_root(self) -> None:
        prim = self._stage.GetPrimAtPath(self.CAMERA_ROOT)
        if not prim.IsValid():
            UsdGeom.Scope.Define(self._stage, self.CAMERA_ROOT)
        for sub in ("Overhead", "Rack", "Depth"):
            sub_path = f"{self.CAMERA_ROOT}/{sub}"
            if not self._stage.GetPrimAtPath(sub_path).IsValid():
                UsdGeom.Scope.Define(self._stage, sub_path)
