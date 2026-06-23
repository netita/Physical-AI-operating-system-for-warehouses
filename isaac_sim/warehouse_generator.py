"""
Warehouse Generator — NVIDIA Isaac Sim / Omniverse

Procedurally generates photorealistic warehouse USD stages for synthetic
data collection. Supports single-aisle, double-aisle and cross-dock layouts.

Usage (inside an Isaac Sim Python environment):
    from isaac_sim.warehouse_generator import WarehouseGenerator, WarehouseConfig
    cfg = WarehouseConfig()
    gen = WarehouseGenerator(cfg)
    stage = gen.generate_scene()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Omniverse / Isaac Sim runtime imports
# ---------------------------------------------------------------------------
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    get_current_stage,
    open_stage,
)
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Usd


# ---------------------------------------------------------------------------
# Enumerations and Configuration
# ---------------------------------------------------------------------------

class LayoutType(str, Enum):
    SINGLE_AISLE = "single_aisle"
    DOUBLE_AISLE = "double_aisle"
    CROSS_DOCK = "cross_dock"


@dataclass
class RackConfig:
    """Dimensions of a single rack unit."""
    width: float = 2.7          # metres — standard pallet-rack bay width
    depth: float = 1.1          # metres
    height: float = 5.5         # metres
    beam_levels: int = 4        # number of load-bearing beam pairs
    uprights_per_bay: int = 3


@dataclass
class LoadingDockConfig:
    num_doors: int = 4
    door_width: float = 4.0     # metres
    door_height: float = 4.5
    dock_depth: float = 12.0    # metres of exterior apron


@dataclass
class LightingConfig:
    num_overhead_area_lights: int = 12
    overhead_intensity: float = 5000.0   # lux-equivalent nits
    overhead_colour: Tuple[float, float, float] = (1.0, 0.97, 0.88)  # 4000 K CCT
    num_skylights: int = 4
    skylight_intensity: float = 2000.0
    enable_point_lights: bool = True
    point_light_count: int = 8
    point_light_intensity: float = 3000.0


@dataclass
class WarehouseConfig:
    # --- overall footprint ---
    floor_length: float = 120.0         # metres (X axis)
    floor_width: float = 60.0           # metres (Y axis)
    ceiling_height: float = 9.0         # metres (Z axis)
    wall_thickness: float = 0.3

    # --- layout ---
    layout: LayoutType = LayoutType.DOUBLE_AISLE
    aisle_width: float = 3.5            # metres between rack faces
    cross_aisle_width: float = 5.0
    num_rack_rows: int = 8
    num_rack_bays_per_row: int = 20

    # --- sub-configs ---
    rack: RackConfig = field(default_factory=RackConfig)
    dock: LoadingDockConfig = field(default_factory=LoadingDockConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)

    # --- asset library paths (Nucleus or local) ---
    rack_usd_path: str = "omniverse://localhost/NVIDIA/Assets/Warehouse/Racks/PalletRack_A.usd"
    floor_mdl: str = "omniverse://localhost/NVIDIA/Assets/Warehouse/Materials/ConcreteFloor_A.mdl"
    wall_mdl: str = "omniverse://localhost/NVIDIA/Assets/Warehouse/Materials/CinderBlock_A.mdl"
    ceiling_mdl: str = "omniverse://localhost/NVIDIA/Assets/Warehouse/Materials/Metal_Ceiling_A.mdl"

    # --- physics ---
    enable_physics: bool = True
    gravity: float = -9.81

    # --- random seed ---
    seed: int = 42


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _define_xform(stage: Usd.Stage, path: str) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    return xform


def _set_translate(prim: Usd.Prim, x: float, y: float, z: float) -> None:
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    # Try to reuse existing translate op
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(x, y, z))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))


def _set_scale(prim: Usd.Prim, sx: float, sy: float, sz: float) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))


def _bind_material(prim: Usd.Prim, mat_path: str, stage: Usd.Stage) -> None:
    """Bind an MDL material already defined on the stage to a prim."""
    mat_prim = stage.GetPrimAtPath(mat_path)
    if mat_prim.IsValid():
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material = UsdShade.Material(mat_prim)
        binding_api.Bind(material)


def _create_mdl_material(
    stage: Usd.Stage, mat_scope: str, name: str, mdl_path: str
) -> str:
    """Create an MDL-backed material prim, return its stage path."""
    mat_path = f"{mat_scope}/{name}"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("mdlMaterial")
    shader.CreateInput("module", Sdf.ValueTypeNames.Asset).Set(mdl_path)
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out"
    )
    return mat_path


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class WarehouseGenerator:
    """
    Procedurally builds a warehouse USD stage.

    All geometry is placed under /World/Warehouse.  Physics colliders are
    applied to floor, walls, and racks so Isaac Sim robots can interact with
    the environment immediately.
    """

    ROOT_PATH = "/World"
    WAREHOUSE_PATH = "/World/Warehouse"
    MATERIAL_SCOPE = "/World/Looks"

    def __init__(self, config: Optional[WarehouseConfig] = None) -> None:
        self.cfg = config or WarehouseConfig()
        random.seed(self.cfg.seed)
        self._stage: Optional[Usd.Stage] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_scene(self) -> Usd.Stage:
        """
        Generate the full warehouse environment.

        Returns
        -------
        Usd.Stage
            The populated USD stage handle.
        """
        self._stage = self._init_stage()
        self._setup_physics()
        self._create_material_scope()
        self._build_shell()           # floor, walls, ceiling
        self._build_racks()
        self._build_loading_docks()
        self._build_lighting()
        self._stage.Save()
        return self._stage

    # ------------------------------------------------------------------
    # Stage initialisation
    # ------------------------------------------------------------------

    def _init_stage(self) -> Usd.Stage:
        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()

        # Set up stage defaults
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Create root xform hierarchy
        _define_xform(stage, self.ROOT_PATH)
        _define_xform(stage, self.WAREHOUSE_PATH)
        return stage

    def _setup_physics(self) -> None:
        if not self.cfg.enable_physics:
            return
        scene = UsdPhysics.Scene.Define(self._stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(abs(self.cfg.gravity))

    def _create_material_scope(self) -> None:
        UsdGeom.Scope.Define(self._stage, self.MATERIAL_SCOPE)
        # Pre-create materials that will be bound later
        _create_mdl_material(
            self._stage, self.MATERIAL_SCOPE, "FloorMat", self.cfg.floor_mdl
        )
        _create_mdl_material(
            self._stage, self.MATERIAL_SCOPE, "WallMat", self.cfg.wall_mdl
        )
        _create_mdl_material(
            self._stage, self.MATERIAL_SCOPE, "CeilingMat", self.cfg.ceiling_mdl
        )

    # ------------------------------------------------------------------
    # Shell: floor, walls, ceiling
    # ------------------------------------------------------------------

    def _build_shell(self) -> None:
        cfg = self.cfg
        wh = self.WAREHOUSE_PATH
        stage = self._stage

        L = cfg.floor_length
        W = cfg.floor_width
        H = cfg.ceiling_height
        T = cfg.wall_thickness

        # ---- Floor ----
        floor_path = f"{wh}/Floor"
        floor = UsdGeom.Cube.Define(stage, floor_path)
        floor_prim = stage.GetPrimAtPath(floor_path)
        _set_translate(floor_prim, L / 2, W / 2, -T / 2)
        _set_scale(floor_prim, L / 2, W / 2, T / 2)
        self._add_collider(floor_prim)
        _bind_material(floor_prim, f"{self.MATERIAL_SCOPE}/FloorMat", stage)

        # ---- Ceiling ----
        ceiling_path = f"{wh}/Ceiling"
        ceiling = UsdGeom.Cube.Define(stage, ceiling_path)
        ceiling_prim = stage.GetPrimAtPath(ceiling_path)
        _set_translate(ceiling_prim, L / 2, W / 2, H + T / 2)
        _set_scale(ceiling_prim, L / 2, W / 2, T / 2)
        _bind_material(ceiling_prim, f"{self.MATERIAL_SCOPE}/CeilingMat", stage)

        # ---- Walls (4 sides) ----
        walls = [
            # (name, cx, cy, cz, sx, sy, sz)
            ("Wall_North", L / 2, W + T / 2, H / 2, L / 2, T / 2, H / 2),
            ("Wall_South", L / 2, -T / 2,    H / 2, L / 2, T / 2, H / 2),
            ("Wall_East",  L + T / 2, W / 2, H / 2, T / 2, W / 2, H / 2),
            ("Wall_West",  -T / 2,    W / 2, H / 2, T / 2, W / 2, H / 2),
        ]
        for name, cx, cy, cz, sx, sy, sz in walls:
            path = f"{wh}/{name}"
            UsdGeom.Cube.Define(stage, path)
            prim = stage.GetPrimAtPath(path)
            _set_translate(prim, cx, cy, cz)
            _set_scale(prim, sx, sy, sz)
            self._add_collider(prim)
            _bind_material(prim, f"{self.MATERIAL_SCOPE}/WallMat", stage)

    def _add_collider(self, prim: Usd.Prim) -> None:
        if self.cfg.enable_physics:
            UsdPhysics.CollisionAPI.Apply(prim)

    # ------------------------------------------------------------------
    # Rack placement
    # ------------------------------------------------------------------

    def _build_racks(self) -> None:
        layout = self.cfg.layout
        if layout == LayoutType.SINGLE_AISLE:
            self._place_racks_single_aisle()
        elif layout == LayoutType.DOUBLE_AISLE:
            self._place_racks_double_aisle()
        elif layout == LayoutType.CROSS_DOCK:
            self._place_racks_cross_dock()

    def _rack_positions_for_row(
        self, row_x: float, row_y_start: float, num_bays: int
    ) -> List[Tuple[float, float]]:
        """Return (x, y) positions for a single rack row."""
        positions = []
        rack_w = self.cfg.rack.width
        for bay in range(num_bays):
            y = row_y_start + bay * rack_w
            positions.append((row_x, y))
        return positions

    def _place_rack_instance(
        self, rack_idx: int, x: float, y: float, angle_deg: float = 0.0
    ) -> None:
        stage = self._stage
        path = f"{self.WAREHOUSE_PATH}/Racks/Rack_{rack_idx:04d}"

        # Reference external rack USD asset
        rack_prim = add_reference_to_stage(
            usd_path=self.cfg.rack_usd_path, prim_path=path
        )
        if rack_prim is None:
            # Fallback: build a simple box proxy when asset is unavailable
            rack_prim = self._build_rack_proxy(path, rack_idx)

        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return

        _set_translate(prim, x, y, 0.0)

        if angle_deg != 0.0:
            xformable = UsdGeom.Xformable(prim)
            xformable.AddRotateZOp().Set(angle_deg)

        if self.cfg.enable_physics:
            UsdPhysics.CollisionAPI.Apply(prim)

    def _build_rack_proxy(self, path: str, idx: int) -> Usd.Prim:
        """Simple box proxy when the Nucleus rack USD is unavailable."""
        stage = self._stage
        rc = self.cfg.rack
        rack_group = _define_xform(stage, path)

        # Upright columns
        for col in range(rc.uprights_per_bay):
            col_x = col * (rc.width / (rc.uprights_per_bay - 1))
            col_path = f"{path}/Upright_{col}"
            upright = UsdGeom.Cube.Define(stage, col_path)
            prim = stage.GetPrimAtPath(col_path)
            _set_translate(prim, col_x, 0.05, rc.height / 2)
            _set_scale(prim, 0.05, rc.depth / 2, rc.height / 2)

        # Horizontal beams
        for level in range(rc.beam_levels):
            z = (level + 1) * (rc.height / (rc.beam_levels + 1))
            beam_path = f"{path}/Beam_{level}"
            UsdGeom.Cube.Define(stage, beam_path)
            prim = stage.GetPrimAtPath(beam_path)
            _set_translate(prim, rc.width / 2, 0.05, z)
            _set_scale(prim, rc.width / 2, 0.04, 0.04)

        return stage.GetPrimAtPath(path)

    def _place_racks_single_aisle(self) -> None:
        """All racks in parallel rows with one shared central aisle."""
        cfg = self.cfg
        rack_depth = cfg.rack.depth
        aisle = cfg.aisle_width
        row_spacing = rack_depth + aisle
        rack_idx = 0

        x_start = 5.0
        for row in range(cfg.num_rack_rows):
            x = x_start + row * row_spacing
            positions = self._rack_positions_for_row(
                x, 5.0, cfg.num_rack_bays_per_row
            )
            for (rx, ry) in positions:
                self._place_rack_instance(rack_idx, rx, ry)
                rack_idx += 1

    def _place_racks_double_aisle(self) -> None:
        """Back-to-back rack pairs separated by two aisles."""
        cfg = self.cfg
        rd = cfg.rack.depth
        aisle = cfg.aisle_width
        pair_spacing = rd * 2 + aisle * 2 + 0.5   # 0.5 m back-to-back gap
        rack_idx = 0

        x_start = 5.0
        num_pairs = cfg.num_rack_rows // 2
        for pair in range(num_pairs):
            x_front = x_start + pair * pair_spacing
            x_back = x_front + rd + 0.5

            for x, angle in [(x_front, 0.0), (x_back, 180.0)]:
                positions = self._rack_positions_for_row(
                    x, 5.0, cfg.num_rack_bays_per_row
                )
                for (rx, ry) in positions:
                    self._place_rack_instance(rack_idx, rx, ry, angle_deg=angle)
                    rack_idx += 1

    def _place_racks_cross_dock(self) -> None:
        """Cross-dock: receiving racks on one side, shipping on the other,
        with a wide transfer aisle through the middle."""
        cfg = self.cfg
        L = cfg.floor_length
        W = cfg.floor_width
        cross_aisle_y = W / 2
        rd = cfg.rack.depth
        aisle = cfg.aisle_width
        rack_idx = 0

        # South half — receiving
        x_start = 5.0
        num_rows_half = cfg.num_rack_rows // 2
        for row in range(num_rows_half):
            x = x_start + row * (rd + aisle)
            y_positions = self._rack_positions_for_row(
                x, 5.0, cfg.num_rack_bays_per_row
            )
            for (rx, ry) in y_positions:
                if ry + rd < cross_aisle_y - cfg.cross_aisle_width / 2:
                    self._place_rack_instance(rack_idx, rx, ry)
                    rack_idx += 1

        # North half — shipping
        for row in range(num_rows_half):
            x = x_start + row * (rd + aisle)
            y_positions = self._rack_positions_for_row(
                x, cross_aisle_y + cfg.cross_aisle_width / 2,
                cfg.num_rack_bays_per_row
            )
            for (rx, ry) in y_positions:
                if ry + rd < W - 5.0:
                    self._place_rack_instance(rack_idx, rx, ry)
                    rack_idx += 1

    # ------------------------------------------------------------------
    # Loading docks
    # ------------------------------------------------------------------

    def _build_loading_docks(self) -> None:
        cfg = self.cfg
        dock_cfg = cfg.dock
        stage = self._stage
        dock_root = f"{self.WAREHOUSE_PATH}/LoadingDocks"
        _define_xform(stage, dock_root)

        door_w = dock_cfg.door_width
        door_h = dock_cfg.door_height
        spacing = cfg.floor_width / (dock_cfg.num_doors + 1)

        for i in range(dock_cfg.num_doors):
            door_y = spacing * (i + 1)
            dock_path = f"{dock_root}/Dock_{i:02d}"
            _define_xform(stage, dock_path)

            # Door aperture (open box cut — represented as a thin frame)
            frame_path = f"{dock_path}/DoorFrame"
            UsdGeom.Cube.Define(stage, frame_path)
            frame_prim = stage.GetPrimAtPath(frame_path)
            _set_translate(frame_prim, cfg.floor_length, door_y, door_h / 2)
            _set_scale(frame_prim, 0.15, door_w / 2 + 0.3, door_h / 2 + 0.3)

            # Dock leveler platform
            leveler_path = f"{dock_path}/Leveler"
            UsdGeom.Cube.Define(stage, leveler_path)
            lev_prim = stage.GetPrimAtPath(leveler_path)
            _set_translate(
                lev_prim,
                cfg.floor_length + dock_cfg.dock_depth / 2,
                door_y,
                -0.15,
            )
            _set_scale(lev_prim, dock_cfg.dock_depth / 2, door_w / 2, 0.15)
            self._add_collider(lev_prim)

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _build_lighting(self) -> None:
        lc = self.cfg.lighting
        stage = self._stage
        light_root = f"{self.WAREHOUSE_PATH}/Lighting"
        _define_xform(stage, light_root)

        L = self.cfg.floor_length
        W = self.cfg.floor_width
        H = self.cfg.ceiling_height

        # ---- Overhead area lights (fluorescent fixture emulation) ----
        rows = max(1, int(math.sqrt(lc.num_overhead_area_lights)))
        cols = max(1, lc.num_overhead_area_lights // rows)
        x_spacing = L / (cols + 1)
        y_spacing = W / (rows + 1)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                path = f"{light_root}/AreaLight_{idx:03d}"
                area = UsdLux.RectLight.Define(stage, path)
                prim = stage.GetPrimAtPath(path)
                _set_translate(prim, x_spacing * (c + 1), y_spacing * (r + 1), H - 0.05)
                area.CreateWidthAttr(4.0)
                area.CreateHeightAttr(1.0)
                area.CreateIntensityAttr(lc.overhead_intensity)
                r_val, g_val, b_val = lc.overhead_colour
                area.CreateColorAttr(Gf.Vec3f(r_val, g_val, b_val))
                idx += 1
                if idx >= lc.num_overhead_area_lights:
                    break

        # ---- Skylights (dome light segments) ----
        sky_x_step = L / (lc.num_skylights + 1)
        for s in range(lc.num_skylights):
            path = f"{light_root}/Skylight_{s:02d}"
            disk = UsdLux.DiskLight.Define(stage, path)
            prim = stage.GetPrimAtPath(path)
            _set_translate(prim, sky_x_step * (s + 1), W / 2, H)
            disk.CreateRadiusAttr(2.5)
            disk.CreateIntensityAttr(lc.skylight_intensity)
            disk.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))  # cool daylight

        # ---- Point lights (accent / emergency) ----
        if lc.enable_point_lights:
            for p in range(lc.point_light_count):
                path = f"{light_root}/PointLight_{p:02d}"
                pt = UsdLux.SphereLight.Define(stage, path)
                prim = stage.GetPrimAtPath(path)
                px = random.uniform(5.0, L - 5.0)
                py = random.uniform(5.0, W - 5.0)
                _set_translate(prim, px, py, H - 1.5)
                pt.CreateRadiusAttr(0.15)
                pt.CreateIntensityAttr(lc.point_light_intensity)
                pt.CreateColorAttr(Gf.Vec3f(1.0, 0.9, 0.7))

        # ---- Distant sun light (for skylight bleed) ----
        sun_path = f"{light_root}/Sun"
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(500.0)
        sun.CreateAngleAttr(0.53)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
        sun_prim = stage.GetPrimAtPath(sun_path)
        UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(
            Gf.Vec3f(45.0, 0.0, 135.0)
        )
