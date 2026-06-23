"""
Fire Simulation — NVIDIA Omniverse Particles

Simulates warehouse fire scenarios for safety-incident synthetic data
generation. Produces volumetric fire and smoke via Omniverse particle
systems and flow (FleX / PhysX Particles) and returns per-frame semantic
segmentation masks that label fire / smoke pixels.

Usage:
    from isaac_sim.fire_simulation import FireSimulation
    fs = FireSimulation(stage)
    fire_id = fs.ignite(location=(30.0, 15.0, 0.0), spread_rate=0.05)
    for step in range(300):
        fs.simulate_spread(fire_id, dt=1/30)
        fs.generate_smoke(fire_id, density=0.6)
        mask = fs.get_fire_mask(fire_id)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# Omniverse particle / flow APIs
try:
    import omni.physx.scripts.particleUtils as particle_utils
    _HAS_PARTICLE_UTILS = True
except ImportError:
    _HAS_PARTICLE_UTILS = False

try:
    import omni.replicator.core as rep
    _HAS_REPLICATOR = True
except ImportError:
    _HAS_REPLICATOR = False


# ---------------------------------------------------------------------------
# Semantic label IDs (must match dataset label map)
# ---------------------------------------------------------------------------

LABEL_FIRE = 10
LABEL_SMOKE = 11
LABEL_BACKGROUND = 0


# ---------------------------------------------------------------------------
# Fire state
# ---------------------------------------------------------------------------

@dataclass
class _FireState:
    fire_id: int
    location: np.ndarray          # initial ignition point (x, y, z)
    spread_rate: float            # metres per second radial spread
    radius: float = 0.2           # current fire radius (m)
    intensity: float = 1.0        # normalised intensity 0–1
    age: float = 0.0              # seconds since ignition

    # Particle system prim paths
    fire_prim_path: str = ""
    smoke_prim_path: str = ""
    emitter_paths: List[str] = field(default_factory=list)

    # Spread front: list of (x, y, radius) emitter discs
    spread_nodes: List[Tuple[float, float, float]] = field(default_factory=list)

    smoke_density: float = 0.3
    smoke_height: float = 3.0     # metres above fire base
    is_extinguished: bool = False


# ---------------------------------------------------------------------------
# FireSimulation
# ---------------------------------------------------------------------------

class FireSimulation:
    """
    Manages one or more fire events in the warehouse stage.

    Each fire has:
      - A particle-based flame emitter (colour from orange to white-hot)
      - A volumetric smoke plume rising above the flame
      - A spreading radial footprint modelled as additional emitters
      - A semantic segmentation overlay method

    Parameters
    ----------
    stage : Usd.Stage, optional
        Active stage. Uses current stage if None.
    fire_scope : str
        Root USD path for fire prims.
    """

    FIRE_SCOPE = "/World/Effects/Fire"
    SMOKE_SCOPE = "/World/Effects/Smoke"

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        fire_scope: str = FIRE_SCOPE,
        smoke_scope: str = SMOKE_SCOPE,
    ) -> None:
        self._stage = stage or get_current_stage()
        self._fire_scope = fire_scope
        self._smoke_scope = smoke_scope
        self._fires: Dict[int, _FireState] = {}
        self._next_id = 0

        self._ensure_scopes()

    # ------------------------------------------------------------------
    # Ignition
    # ------------------------------------------------------------------

    def ignite(
        self,
        location: Tuple[float, float, float],
        spread_rate: float = 0.03,
        initial_radius: float = 0.3,
        intensity: float = 1.0,
        seed: Optional[int] = None,
    ) -> int:
        """
        Start a fire at the given world position.

        Parameters
        ----------
        location : (x, y, z)
            Ignition point in world space.
        spread_rate : float
            Radial spread speed in metres per second.
        initial_radius : float
            Starting fire disc radius in metres.
        intensity : float
            Initial flame intensity (0–1 maps to particle rate scaling).
        seed : int, optional
            RNG seed for particle variation.

        Returns
        -------
        int
            Fire ID used in subsequent method calls.
        """
        rng = random.Random(seed)
        fid = self._next_id
        self._next_id += 1

        loc = np.array(location, dtype=np.float64)
        state = _FireState(
            fire_id=fid,
            location=loc,
            spread_rate=spread_rate,
            radius=initial_radius,
            intensity=intensity,
        )

        # Create USD prims for fire and smoke
        state.fire_prim_path = self._create_fire_emitter(
            fid, loc, initial_radius, intensity, rng
        )
        state.smoke_prim_path = self._create_smoke_emitter(
            fid, loc, smoke_density=0.3, rng=rng
        )
        state.spread_nodes = [(float(loc[0]), float(loc[1]), initial_radius)]

        self._fires[fid] = state
        return fid

    # ------------------------------------------------------------------
    # Spread simulation
    # ------------------------------------------------------------------

    def simulate_spread(
        self,
        fire_id: int,
        dt: float = 1.0 / 30.0,
        max_radius: float = 15.0,
        spread_probability: float = 0.05,
    ) -> float:
        """
        Advance fire spread by one timestep.

        Fire expands radially; new emitter nodes are added at the perimeter
        to model irregular spread.  Smoke height and density increase with
        radius.

        Parameters
        ----------
        fire_id : int
        dt : float
            Simulation timestep (seconds).
        max_radius : float
            Maximum spread radius in metres before fire auto-extinguishes.
        spread_probability : float
            Per-frame probability of spawning a new emitter at the perimeter.

        Returns
        -------
        float
            Current fire radius in metres.
        """
        state = self._get_state(fire_id)
        if state.is_extinguished:
            return state.radius

        state.age += dt
        state.radius = min(max_radius, state.radius + state.spread_rate * dt)

        # Smoke grows with radius
        state.smoke_height = 2.0 + state.radius * 0.8
        state.smoke_density = min(1.0, 0.2 + state.radius * 0.05)

        # Stochastic new emitter at perimeter
        rng = random.Random(int(state.age * 1000) + fire_id)
        if rng.random() < spread_probability and state.radius < max_radius:
            angle = rng.uniform(0.0, 2 * math.pi)
            nx = state.location[0] + math.cos(angle) * state.radius
            ny = state.location[1] + math.sin(angle) * state.radius
            nr = rng.uniform(0.1, 0.4)
            state.spread_nodes.append((nx, ny, nr))

            # Create new USD emitter for this node
            new_loc = np.array([nx, ny, float(state.location[2])])
            new_path = self._create_fire_emitter(
                fire_id, new_loc, nr, state.intensity * 0.6, rng,
                suffix=f"_node_{len(state.spread_nodes):03d}"
            )
            state.emitter_paths.append(new_path)

        # Update main emitter size
        self._update_emitter_radius(state.fire_prim_path, state.radius)

        if state.radius >= max_radius:
            state.is_extinguished = True  # Marks for suppression animation

        return state.radius

    # ------------------------------------------------------------------
    # Smoke generation
    # ------------------------------------------------------------------

    def generate_smoke(
        self,
        fire_id: int,
        density: float = 0.5,
        dispersion_radius: float = 5.0,
        wind_vector: Tuple[float, float, float] = (0.2, 0.0, 0.0),
    ) -> str:
        """
        Update volumetric smoke parameters above the fire.

        Parameters
        ----------
        fire_id : int
        density : float
            Smoke optical density (0 = transparent, 1 = opaque).
        dispersion_radius : float
            Horizontal spread of smoke plume at ceiling level.
        wind_vector : (vx, vy, vz)
            Drift velocity in m/s applied to smoke particles.

        Returns
        -------
        str
            USD path of the smoke prim.
        """
        state = self._get_state(fire_id)
        state.smoke_density = density
        smoke_prim = self._stage.GetPrimAtPath(state.smoke_prim_path)

        if smoke_prim.IsValid():
            # Update position to follow fire centre + drift
            xf = UsdGeom.Xformable(smoke_prim)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            drift_x = state.location[0] + wind_vector[0] * state.age * 0.1
            drift_y = state.location[1] + wind_vector[1] * state.age * 0.1
            smoke_z = float(state.location[2]) + state.smoke_height

            if "xformOp:translate" in ops:
                ops["xformOp:translate"].Set(Gf.Vec3d(drift_x, drift_y, smoke_z))

            # Update density via display colour alpha proxy
            smoke_geo = UsdGeom.Sphere(smoke_prim)
            if smoke_geo:
                alpha = min(1.0, density)
                smoke_geo.CreateDisplayColorAttr(
                    [Gf.Vec3f(0.25, 0.25, 0.25)]
                )
                # Radius scales with dispersion + wind drift
                smoke_prim.GetAttribute("radius").Set(
                    max(0.5, dispersion_radius + state.age * 0.02)
                )

        return state.smoke_prim_path

    # ------------------------------------------------------------------
    # Semantic segmentation mask
    # ------------------------------------------------------------------

    def get_fire_mask(
        self,
        fire_id: int,
        image_shape: Tuple[int, int] = (1080, 1920),
        camera_matrix: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a per-pixel semantic label mask for the fire and smoke.

        When Replicator annotators are available, reads the semantic
        segmentation buffer and remaps fire/smoke object IDs to LABEL_FIRE
        and LABEL_SMOKE. Otherwise returns a projected 2-D ellipse mask.

        Parameters
        ----------
        fire_id : int
        image_shape : (H, W)
            Output mask resolution.
        camera_matrix : np.ndarray (3,3), optional
            Camera intrinsics for projection fallback.
        camera_pose : np.ndarray (4,4), optional
            Camera world transform (inverse of extrinsics).

        Returns
        -------
        np.ndarray, shape (H, W), dtype int32
            Pixel-wise label map: 0=background, 10=fire, 11=smoke.
        """
        state = self._get_state(fire_id)
        H, W = image_shape
        mask = np.zeros((H, W), dtype=np.int32)

        if _HAS_REPLICATOR:
            # Read from Replicator semantic annotator
            try:
                mask = self._read_replicator_seg(state, H, W)
                return mask
            except Exception:
                pass  # Fall through to projection

        # ---- Projection fallback ----
        if camera_matrix is None:
            # Default intrinsics (1920×1080 @ 60° HFOV)
            fx = fy = W / (2.0 * math.tan(math.radians(30.0)))
            camera_matrix = np.array([
                [fx, 0, W / 2],
                [0, fy, H / 2],
                [0,  0,   1],
            ], dtype=np.float64)

        if camera_pose is None:
            # Default: overhead camera at (60, 30, 9), pointing down
            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[2, 3] = 9.0

        mask = self._project_fire_ellipse(state, mask, camera_matrix, camera_pose)
        return mask

    # ------------------------------------------------------------------
    # Extinguish
    # ------------------------------------------------------------------

    def extinguish(self, fire_id: int) -> None:
        """Remove all fire and smoke prims for a fire event."""
        state = self._get_state(fire_id)
        paths_to_remove = (
            [state.fire_prim_path, state.smoke_prim_path] + state.emitter_paths
        )
        for path in paths_to_remove:
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid():
                self._stage.RemovePrim(prim.GetPath())
        state.is_extinguished = True

    # ------------------------------------------------------------------
    # USD particle / smoke prim construction
    # ------------------------------------------------------------------

    def _create_fire_emitter(
        self,
        fire_id: int,
        location: np.ndarray,
        radius: float,
        intensity: float,
        rng: random.Random,
        suffix: str = "",
    ) -> str:
        """Create a particle-based fire emitter prim."""
        prim_path = (
            f"{self._fire_scope}/Fire_{fire_id:04d}{suffix}"
        )

        if _HAS_PARTICLE_UTILS:
            try:
                particle_utils.create_particle_system(
                    stage=self._stage,
                    path=prim_path,
                    simulation_owner="/World/PhysicsScene",
                )
                # Configure particle emitter
                emitter_path = f"{prim_path}/Emitter"
                particle_utils.create_particle_emitter(
                    stage=self._stage,
                    path=emitter_path,
                    particle_system_path=prim_path,
                )
                emitter_prim = self._stage.GetPrimAtPath(emitter_path)
                if emitter_prim.IsValid():
                    xf = UsdGeom.Xformable(emitter_prim)
                    xf.AddTranslateOp().Set(Gf.Vec3d(*location.tolist()))
                return prim_path
            except Exception:
                pass

        # Fallback: animated cone geometry as proxy
        prim_path_proxy = f"{self._fire_scope}/FireProxy_{fire_id:04d}{suffix}"
        cone = UsdGeom.Cone.Define(self._stage, prim_path_proxy)
        prim = self._stage.GetPrimAtPath(prim_path_proxy)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*location.tolist()))
        # Flame height proportional to intensity, radius from spread
        cone.CreateHeightAttr(2.0 * intensity)
        cone.CreateRadiusAttr(radius)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 0.4 * rng.random(), 0.0)]  # orange-yellow
        )
        return prim_path_proxy

    def _create_smoke_emitter(
        self,
        fire_id: int,
        location: np.ndarray,
        smoke_density: float,
        rng: random.Random,
    ) -> str:
        """Create a volumetric smoke sphere proxy prim."""
        prim_path = f"{self._smoke_scope}/Smoke_{fire_id:04d}"
        sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
        prim = self._stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(
            Gf.Vec3d(float(location[0]), float(location[1]), float(location[2]) + 3.0)
        )
        sphere.CreateRadiusAttr(2.0)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr(
            [Gf.Vec3f(0.25, 0.25, 0.25)]
        )
        # Tag for semantic segmentation
        prim.CreateAttribute("semantic:labels", Sdf.ValueTypeNames.StringArray).Set(
            ["smoke"]
        )
        return prim_path

    def _update_emitter_radius(self, prim_path: str, radius: float) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        cone = UsdGeom.Cone(prim)
        if cone:
            cone.CreateRadiusAttr(radius)
            cone.CreateHeightAttr(min(6.0, 1.5 + radius * 0.5))

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------

    def _read_replicator_seg(
        self, state: _FireState, H: int, W: int
    ) -> np.ndarray:
        """Read semantic segmentation from Replicator and remap fire labels."""
        seg_ann = rep.annotators.get("semantic_segmentation")
        # Assume render product is already attached externally
        data = seg_ann.get_data()
        raw = data["data"].reshape(H, W).astype(np.int32)
        id_map = data.get("info", {}).get("idToLabels", {})

        fire_ids = {
            int(k) for k, v in id_map.items() if "fire" in str(v).lower()
        }
        smoke_ids = {
            int(k) for k, v in id_map.items() if "smoke" in str(v).lower()
        }

        mask = np.zeros((H, W), dtype=np.int32)
        for fid in fire_ids:
            mask[raw == fid] = LABEL_FIRE
        for sid in smoke_ids:
            mask[raw == sid] = LABEL_SMOKE

        return mask

    def _project_fire_ellipse(
        self,
        state: _FireState,
        mask: np.ndarray,
        K: np.ndarray,
        cam_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Project fire disc and smoke sphere onto the image plane and paint
        the corresponding pixels with semantic labels.
        """
        H, W = mask.shape
        cam_inv = np.linalg.inv(cam_pose)

        def project_point(world_pt: np.ndarray) -> Optional[Tuple[int, int]]:
            p_cam = cam_inv @ np.append(world_pt, 1.0)
            if p_cam[2] <= 0.0:
                return None
            p_img = K @ p_cam[:3]
            u, v = int(p_img[0] / p_img[2]), int(p_img[1] / p_img[2])
            if 0 <= u < W and 0 <= v < H:
                return (u, v)
            return None

        # Paint fire nodes
        for nx, ny, nr in state.spread_nodes:
            centre_3d = np.array([nx, ny, float(state.location[2]) + 0.5])
            projected = project_point(centre_3d)
            if projected is None:
                continue
            u0, v0 = projected
            # Radius in pixels: approximate via focal length
            fx = K[0, 0]
            dist = float(np.linalg.norm(cam_pose[:3, 3] - centre_3d))
            r_px = max(1, int(fx * nr / max(dist, 0.01)))

            rr, cc = _draw_filled_ellipse(v0, u0, r_px, r_px, H, W)
            mask[rr, cc] = LABEL_FIRE

        # Paint smoke
        smoke_3d = np.array([
            float(state.location[0]),
            float(state.location[1]),
            float(state.location[2]) + state.smoke_height,
        ])
        projected_smoke = project_point(smoke_3d)
        if projected_smoke is not None:
            u0, v0 = projected_smoke
            fx = K[0, 0]
            dist = float(np.linalg.norm(cam_pose[:3, 3] - smoke_3d))
            r_px = max(1, int(fx * state.radius * 1.5 / max(dist, 0.01)))
            rr, cc = _draw_filled_ellipse(v0, u0, r_px, int(r_px * 0.6), H, W)
            # Only set smoke where not already fire
            smoke_region = mask[rr, cc]
            mask[rr, cc] = np.where(smoke_region == LABEL_FIRE, LABEL_FIRE, LABEL_SMOKE)

        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self, fire_id: int) -> _FireState:
        state = self._fires.get(fire_id)
        if state is None:
            raise KeyError(f"Fire {fire_id} not found.")
        return state

    def _ensure_scopes(self) -> None:
        for path in (
            "/World/Effects",
            self._fire_scope,
            self._smoke_scope,
        ):
            if not self._stage.GetPrimAtPath(path).IsValid():
                UsdGeom.Scope.Define(self._stage, path)


# ---------------------------------------------------------------------------
# Rasterisation utility (no OpenCV dependency)
# ---------------------------------------------------------------------------

def _draw_filled_ellipse(
    cy: int, cx: int, ry: int, rx: int, H: int, W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return row and column indices of pixels inside an axis-aligned ellipse.
    Pure numpy implementation — no OpenCV required.
    """
    y_min = max(0, cy - ry)
    y_max = min(H, cy + ry + 1)
    x_min = max(0, cx - rx)
    x_max = min(W, cx + rx + 1)

    ys = np.arange(y_min, y_max)
    xs = np.arange(x_min, x_max)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    if ry == 0 or rx == 0:
        inside = np.zeros_like(yy, dtype=bool)
    else:
        inside = ((yy - cy) ** 2 / ry ** 2 + (xx - cx) ** 2 / rx ** 2) <= 1.0

    rr = yy[inside].ravel()
    cc = xx[inside].ravel()
    return rr, cc
