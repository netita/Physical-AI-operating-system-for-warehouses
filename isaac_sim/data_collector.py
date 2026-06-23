"""
Data Collector — Synthetic Dataset Orchestrator

Orchestrates a complete synthetic data generation run:
  1. Build a warehouse scene (WarehouseGenerator)
  2. Randomise appearance (DomainRandomizer)
  3. Spawn and animate actors (WorkerBehavior, ForkiftBehavior)
  4. Optionally ignite fire events (FireSimulation)
  5. Capture multi-modal frames (CameraManager)
  6. Export in HuggingFace Datasets format

CLI usage:
    python -m isaac_sim.data_collector \\
        --config configs/warehouse_default.yaml \\
        --output-dir /data/warehouse_synthetic_v1 \\
        --num-episodes 100 \\
        --frames-per-episode 200

Programmatic usage:
    from isaac_sim.data_collector import DataCollector, EpisodeConfig
    collector = DataCollector()
    collector.run_episode(EpisodeConfig(episode_id=0))
    collector.export_dataset("/data/out")
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure the warehousegpt package root is on sys.path so sibling imports work
# regardless of working directory or how Isaac Sim's python.sh sets PYTHONPATH.
_THIS_DIR = Path(__file__).resolve().parent          # warehousegpt/isaac_sim/
_PKG_ROOT  = _THIS_DIR.parent                        # warehousegpt/
_REPO_ROOT = _PKG_ROOT.parent                        # Physical AI operating system for warehouses/
for _p in (_PKG_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# SimulationApp MUST be instantiated before any omni.* / isaacsim.* imports.
# When running headless (e.g. on a render farm) pass headless=True.
try:
    from isaacsim import SimulationApp
    _simulation_app = SimulationApp({"headless": True})
    import omni.kit.app
    import omni.usd
    from isaacsim.core.api import World
    _HAS_ISAAC = True
except ImportError:
    _simulation_app = None
    _HAS_ISAAC = False

# Internal modules
from isaac_sim.warehouse_generator import WarehouseConfig, WarehouseGenerator, LayoutType
from isaac_sim.domain_randomizer import DomainRandomizer
from isaac_sim.camera_placement import CameraManager, CameraFrame
from isaac_sim.worker_behavior import WorkerBehavior
from isaac_sim.forklift_behavior import ForkiftBehavior
from isaac_sim.fire_simulation import FireSimulation


# ---------------------------------------------------------------------------
# Dataset format helpers
# ---------------------------------------------------------------------------

try:
    import datasets as hf_datasets
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EpisodeConfig:
    """Configuration for a single synthetic data episode."""

    episode_id: int = 0
    seed: int = 0

    # Scenario flags
    include_workers: bool = True
    include_forklifts: bool = True
    include_fire: bool = False
    incident_type: Optional[str] = None   # "near_miss" | "collision" | "tip_over"

    # Actor counts
    min_workers: int = 2
    max_workers: int = 8
    min_forklifts: int = 1
    max_forklifts: int = 4

    # Simulation
    num_frames: int = 200
    dt: float = 1.0 / 30.0     # 30 Hz

    # Warehouse layout
    layout: str = "double_aisle"
    floor_length: float = 120.0
    floor_width: float = 60.0
    ceiling_height: float = 9.0

    # Weather / time of day
    randomize_weather: bool = True
    randomize_time_of_day: bool = True
    randomize_lighting: bool = True
    randomize_textures: bool = True

    # Camera config
    overhead_camera_spacing: float = 30.0
    image_width: int = 1920
    image_height: int = 1080

    # Output
    save_rgb: bool = True
    save_depth: bool = True
    save_segmentation: bool = True
    save_bboxes: bool = True
    jpeg_quality: int = 90


@dataclass
class FrameRecord:
    """Metadata for a single captured frame."""
    episode_id: int
    frame_id: int
    camera_id: str
    timestamp: float
    image_path: str
    depth_path: Optional[str]
    seg_path: Optional[str]
    intrinsics: List[List[float]]           # 3×3
    extrinsics: List[List[float]]           # 4×4
    worker_poses: Dict[str, List[List[float]]]
    forklift_poses: Dict[str, List[List[float]]]
    bounding_boxes: List[Dict[str, Any]]
    weather: str
    time_of_day: float
    fire_active: bool
    fire_radius: float


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------

class DataCollector:
    """
    Orchestrates full synthetic data generation runs.

    Parameters
    ----------
    output_root : str | Path
        Root directory for all generated data.
    warehouse_config : WarehouseConfig, optional
        Shared warehouse config. Episodes can override per-episode fields.
    """

    DATASET_VERSION = "1.0.0"

    def __init__(
        self,
        output_root: str = "/tmp/warehouse_synthetic",
        warehouse_config: Optional[WarehouseConfig] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self._base_wh_cfg = warehouse_config or WarehouseConfig()
        self._frame_records: List[FrameRecord] = []
        self._current_episode: Optional[EpisodeConfig] = None

        # Subsystems (initialised per episode)
        self._world: Optional[Any] = None
        self._generator: Optional[WarehouseGenerator] = None
        self._randomizer: Optional[DomainRandomizer] = None
        self._cameras: Optional[CameraManager] = None
        self._workers: Optional[WorkerBehavior] = None
        self._forklifts: Optional[ForkiftBehavior] = None
        self._fire: Optional[FireSimulation] = None
        self._active_fire_id: Optional[int] = None
        self._current_weather: str = "clear"
        self._current_time_of_day: float = 12.0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def run_episode(self, config: EpisodeConfig) -> List[FrameRecord]:
        """
        Execute one complete synthetic data episode.

        Generates the environment, randomises appearance, spawns actors,
        runs the simulation loop, and captures frames.

        Parameters
        ----------
        config : EpisodeConfig

        Returns
        -------
        list[FrameRecord]
            All frame records captured in this episode.
        """
        self._current_episode = config
        episode_records: List[FrameRecord] = []

        print(f"[DataCollector] Starting episode {config.episode_id} "
              f"(seed={config.seed})")

        # 1. Set up stage and World
        self._setup_world(config)

        # 2. Generate warehouse geometry
        self._setup_warehouse(config)

        # 3. Randomise domain
        self._randomise_domain(config)

        # 4. Place cameras
        self._setup_cameras(config)

        # 5. Spawn actors
        self._spawn_actors(config)

        # 6. Optionally ignite fire
        if config.include_fire:
            self._start_fire(config)

        # 7. Simulation + capture loop
        print(f"[DataCollector] Running {config.num_frames} frames...")
        for frame_idx in range(config.num_frames):
            try:
                self._simulation_step(config.dt, config)
                records = self.capture_frame(frame_idx, config)
                episode_records.extend(records)
            except Exception as exc:
                print(f"  [WARNING] frame {frame_idx} failed: {exc}")
                import traceback; traceback.print_exc()
                continue

            if frame_idx % 10 == 0:
                print(f"  frame {frame_idx}/{config.num_frames}")

        self._frame_records.extend(episode_records)
        print(f"[DataCollector] Episode {config.episode_id} complete: "
              f"{len(episode_records)} frames captured.")
        return episode_records

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture_frame(
        self,
        frame_idx: int,
        config: Optional[EpisodeConfig] = None,
    ) -> List[FrameRecord]:
        """
        Capture one time-step from all cameras and write files to disk.

        Returns one FrameRecord per active camera.
        """
        if config is None:
            config = self._current_episode
        assert config is not None, "No active episode config."

        episode_dir = self._episode_dir(config.episode_id)
        records: List[FrameRecord] = []

        if self._cameras is None:
            return records

        cam_frames: Dict[str, CameraFrame] = self._cameras.get_camera_data()

        for cam_id, frame in cam_frames.items():
            # --- File paths ---
            frame_stem = f"ep{config.episode_id:06d}_f{frame_idx:06d}_{cam_id}"
            rgb_rel = f"rgb/{frame_stem}.jpg"
            depth_rel = f"depth/{frame_stem}.npy"
            seg_rel = f"seg/{frame_stem}.npy"

            rgb_path = episode_dir / rgb_rel
            depth_path = episode_dir / depth_rel
            seg_path = episode_dir / seg_rel

            # --- Save RGB ---
            if config.save_rgb:
                self._save_rgb(frame.rgb, rgb_path, quality=config.jpeg_quality)

            # --- Save Depth ---
            if config.save_depth:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(depth_path), frame.depth)

            # --- Save Segmentation ---
            if config.save_segmentation:
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(seg_path), frame.segmentation)

            # --- Bounding boxes from seg map ---
            bboxes: List[Dict[str, Any]] = []
            if config.save_bboxes:
                bboxes = self._extract_bboxes(frame.segmentation)

            # --- Worker / forklift poses ---
            worker_poses = {}
            if self._workers:
                for wid, mat in self._workers.get_worker_poses().items():
                    worker_poses[str(wid)] = mat.tolist()

            forklift_poses: Dict[str, Any] = {}
            # (ForkiftBehavior does not expose get_poses() directly; use state)
            # We store as empty dict for now — subclasses can extend.

            # --- Build record ---
            record = FrameRecord(
                episode_id=config.episode_id,
                frame_id=frame_idx,
                camera_id=cam_id,
                timestamp=frame.timestamp,
                image_path=str(rgb_path),
                depth_path=str(depth_path) if config.save_depth else None,
                seg_path=str(seg_path) if config.save_segmentation else None,
                intrinsics=frame.intrinsics.tolist(),
                extrinsics=frame.extrinsics.tolist(),
                worker_poses=worker_poses,
                forklift_poses=forklift_poses,
                bounding_boxes=bboxes,
                weather=self._current_weather,
                time_of_day=self._current_time_of_day,
                fire_active=self._active_fire_id is not None,
                fire_radius=(
                    self._get_fire_radius()
                    if self._active_fire_id is not None
                    else 0.0
                ),
            )
            records.append(record)

            # Write per-frame JSON sidecar
            meta_path = episode_dir / f"meta/{frame_stem}.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(asdict(record), indent=2))

        return records

    # ------------------------------------------------------------------
    # Dataset export
    # ------------------------------------------------------------------

    def export_dataset(
        self,
        output_dir: str,
        format: str = "huggingface",
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
    ) -> Path:
        """
        Export all collected frame records as a HuggingFace Dataset.

        Parameters
        ----------
        output_dir : str
            Directory to save the dataset.
        format : str
            ``"huggingface"`` (default) or ``"json"`` (flat JSON lines).
        push_to_hub : bool
            Whether to push to HuggingFace Hub.
        hub_repo_id : str, optional
            HuggingFace Hub repo ID (required if push_to_hub=True).

        Returns
        -------
        Path
            Path to the saved dataset.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"[DataCollector] Exporting {len(self._frame_records)} records "
              f"to {out_path} ...")

        if format == "huggingface" and _HAS_HF:
            return self._export_huggingface(out_path, push_to_hub, hub_repo_id)
        else:
            return self._export_jsonl(out_path)

    # ------------------------------------------------------------------
    # Private: scene setup
    # ------------------------------------------------------------------

    def _setup_world(self, config: EpisodeConfig) -> None:
        # Don't use World here — WarehouseGenerator.generate_scene() calls
        # ctx.new_stage() which would invalidate any World created first.
        # Physics is configured directly in the USD stage by WarehouseGenerator.
        if _HAS_ISAAC:
            import omni.replicator.core as rep
            import carb.settings
            carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
            carb.settings.get_settings().set_bool(
                "/app/omni.graph.scriptnode/opt_in", True
            )
            rep.orchestrator.set_capture_on_play(False)

    def _setup_warehouse(self, config: EpisodeConfig) -> None:
        # Use the pre-built NVIDIA warehouse stage (LOD-optimised, GPU-friendly).
        # Our WarehouseGenerator creates 160+ separate USD references which
        # exhausts the Vulkan descriptor pool on first render. The NVIDIA stage
        # uses instancing and already passed GPU validation in test_warehouse_scene.py.
        if _HAS_ISAAC:
            from isaacsim.storage.native import get_assets_root_path
            import omni.usd
            assets_root = get_assets_root_path()
            if assets_root:
                stage_url = (
                    assets_root
                    + "/Isaac/Samples/Replicator/Stage/"
                    "full_warehouse_worker_and_anim_cameras.usd"
                )
                print(f"[DataCollector] Loading warehouse stage: {stage_url}")
                omni.usd.get_context().open_stage(stage_url)
                _simulation_app.update()
            else:
                # Fallback: generate procedurally (small config to stay within VRAM)
                wh_cfg = WarehouseConfig(
                    floor_length=config.floor_length,
                    floor_width=config.floor_width,
                    ceiling_height=config.ceiling_height,
                    layout=LayoutType(config.layout),
                    num_rack_rows=4,
                    num_rack_bays_per_row=8,
                    seed=config.seed,
                )
                self._generator = WarehouseGenerator(wh_cfg)
                self._generator.generate_scene()
                _simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        self._randomizer = DomainRandomizer(stage)
        self._workers = WorkerBehavior(stage)
        self._forklifts = ForkiftBehavior(
            stage,
            floor_length=config.floor_length,
            floor_width=config.floor_width,
        )
        self._fire = FireSimulation(stage)

    def _randomise_domain(self, config: EpisodeConfig) -> None:
        dr = self._randomizer
        if dr is None:
            return
        if config.randomize_lighting:
            dr.randomize_lighting(seed=config.seed)
        if config.randomize_textures:
            dr.randomize_textures(seed=config.seed + 1)
        if config.randomize_time_of_day:
            self._current_time_of_day = dr.randomize_time_of_day(
                seed=config.seed + 2
            )
        if config.randomize_weather:
            self._current_weather = dr.randomize_weather(
                seed=config.seed + 3
            )

    def _setup_cameras(self, config: EpisodeConfig) -> None:
        import omni.usd as _ousd
        stage = (
            self._generator._stage
            if self._generator
            else (_ousd.get_context().get_stage() if _HAS_ISAAC else None)
        )
        self._cameras = CameraManager(
            stage=stage,
            floor_length=config.floor_length,
            floor_width=config.floor_width,
            ceiling_height=config.ceiling_height,
        )
        self._cameras.place_overhead_cameras(
            grid_spacing=config.overhead_camera_spacing,
            resolution=(config.image_width, config.image_height),
            max_cameras=6,
        )
        self._cameras.place_rack_cameras(
            resolution=(config.image_width, config.image_height),
            max_cameras=2,
        )
        self._cameras.configure_depth_sensor()

    def _spawn_actors(self, config: EpisodeConfig) -> None:
        rng = random.Random(config.seed + 10)

        if config.include_workers and self._workers:
            dr = self._randomizer
            if dr:
                dr.randomize_worker_count(
                    seed=config.seed + 10,
                    min_workers=config.min_workers,
                    max_workers=config.max_workers,
                )
            else:
                count = rng.randint(config.min_workers, config.max_workers)
                for _ in range(count):
                    x = rng.uniform(5.0, config.floor_length - 5.0)
                    y = rng.uniform(5.0, config.floor_width - 5.0)
                    self._workers.spawn_worker((x, y, 0.0))

        if config.include_forklifts and self._forklifts:
            dr = self._randomizer
            if dr:
                dr.randomize_forklift_count(
                    seed=config.seed + 20,
                    min_forklifts=config.min_forklifts,
                    max_forklifts=config.max_forklifts,
                )
            else:
                count = rng.randint(config.min_forklifts, config.max_forklifts)
                for _ in range(count):
                    x = rng.uniform(5.0, config.floor_length - 5.0)
                    y = rng.uniform(5.0, config.floor_width - 5.0)
                    self._forklifts.spawn_forklift((x, y, 0.0))

    def _start_fire(self, config: EpisodeConfig) -> None:
        if self._fire is None:
            return
        rng = random.Random(config.seed + 99)
        fx = rng.uniform(10.0, config.floor_length - 10.0)
        fy = rng.uniform(10.0, config.floor_width - 10.0)
        self._active_fire_id = self._fire.ignite(
            location=(fx, fy, 0.0),
            spread_rate=rng.uniform(0.02, 0.08),
            seed=config.seed,
        )

    def _simulation_step(self, dt: float, config: EpisodeConfig) -> None:
        if self._workers:
            self._workers.simulate_step(dt)
        if self._forklifts:
            self._forklifts.simulate_step(dt)
        if self._fire and self._active_fire_id is not None:
            self._fire.simulate_spread(self._active_fire_id, dt=dt)
            self._fire.generate_smoke(self._active_fire_id)
        if _HAS_ISAAC and _simulation_app is not None:
            _simulation_app.update()

    def _get_fire_radius(self) -> float:
        if self._fire is None or self._active_fire_id is None:
            return 0.0
        state = self._fire._fires.get(self._active_fire_id)
        return state.radius if state else 0.0

    # ------------------------------------------------------------------
    # Private: file I/O
    # ------------------------------------------------------------------

    def _episode_dir(self, episode_id: int) -> Path:
        d = self.output_root / f"episode_{episode_id:06d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_rgb(
        self, rgb: np.ndarray, path: Path, quality: int = 90
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_PIL:
            img = PILImage.fromarray(rgb.astype(np.uint8))
            img.save(str(path), "JPEG", quality=quality)
        else:
            # Fallback: save as raw npy
            np.save(str(path.with_suffix(".npy")), rgb)

    def _extract_bboxes(
        self, seg: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extract axis-aligned bounding boxes from a segmentation mask.
        Returns a list of dicts with keys: label_id, x1, y1, x2, y2, area.
        """
        bboxes: List[Dict[str, Any]] = []
        for label_id in np.unique(seg):
            if label_id == 0:
                continue  # background
            mask = seg == label_id
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any():
                continue
            y1, y2 = int(np.where(rows)[0][[0, -1]])
            x1, x2 = int(np.where(cols)[0][[0, -1]])
            bboxes.append({
                "label_id": int(label_id),
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "area": int((y2 - y1) * (x2 - x1)),
            })
        return bboxes

    # ------------------------------------------------------------------
    # Private: dataset export
    # ------------------------------------------------------------------

    def _export_huggingface(
        self,
        out_path: Path,
        push_to_hub: bool,
        hub_repo_id: Optional[str],
    ) -> Path:
        rows = [asdict(r) for r in self._frame_records]

        # Write dataset_info.json
        dataset_info = {
            "version": self.DATASET_VERSION,
            "num_records": len(rows),
            "features": {
                "episode_id": "int32",
                "frame_id": "int32",
                "camera_id": "string",
                "timestamp": "float64",
                "image_path": "string",
                "depth_path": "string",
                "seg_path": "string",
                "weather": "string",
                "time_of_day": "float32",
                "fire_active": "bool",
                "fire_radius": "float32",
                "bounding_boxes": "sequence<dict>",
                "worker_poses": "dict<string, array(4,4)>",
                "intrinsics": "array(3,3)",
                "extrinsics": "array(4,4)",
            },
        }
        (out_path / "dataset_info.json").write_text(
            json.dumps(dataset_info, indent=2)
        )

        if _HAS_HF:
            ds = hf_datasets.Dataset.from_list(rows)
            ds.save_to_disk(str(out_path / "data"))

            if push_to_hub and hub_repo_id:
                ds.push_to_hub(hub_repo_id)
                print(f"[DataCollector] Pushed to Hub: {hub_repo_id}")
        else:
            # Fallback to JSONL
            self._export_jsonl(out_path)

        print(f"[DataCollector] Dataset saved to {out_path}")
        return out_path

    def _export_jsonl(self, out_path: Path) -> Path:
        jsonl_path = out_path / "dataset.jsonl"
        with jsonl_path.open("w") as f:
            for record in self._frame_records:
                f.write(json.dumps(asdict(record)) + "\n")
        print(f"[DataCollector] JSONL saved to {jsonl_path}")
        return jsonl_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WarehouseGPT Synthetic Data Generator (Phase 1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file (optional, overrides defaults)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/warehouse_synthetic",
        help="Root output directory for generated data",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to generate",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=200,
        help="Number of simulation frames to capture per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (each episode uses seed + episode_id)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="double_aisle",
        choices=["single_aisle", "double_aisle", "cross_dock"],
        help="Warehouse rack layout",
    )
    parser.add_argument(
        "--include-fire",
        action="store_true",
        help="Include fire simulation in episodes",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push exported dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g. your-org/warehouse-synth-v1)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
    )
    args, _ = parser.parse_known_args()  # ignore Isaac Sim flags like --headless

    # Optionally load YAML config
    base_wh_cfg = WarehouseConfig()
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            yaml_data = yaml.safe_load(f)
        # Map top-level YAML keys to WarehouseConfig
        wh_keys = {
            "floor_length", "floor_width", "ceiling_height",
            "num_rack_rows", "num_rack_bays_per_row", "aisle_width",
        }
        for k, v in yaml_data.get("warehouse", {}).items():
            if k in wh_keys:
                setattr(base_wh_cfg, k, v)

    collector = DataCollector(
        output_root=args.output_dir,
        warehouse_config=base_wh_cfg,
    )

    start_t = time.time()

    for ep_idx in range(args.num_episodes):
        ep_cfg = EpisodeConfig(
            episode_id=ep_idx,
            seed=args.seed + ep_idx,
            layout=args.layout,
            num_frames=args.frames_per_episode,
            include_fire=args.include_fire,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        collector.run_episode(ep_cfg)

    elapsed = time.time() - start_t
    print(f"\n[DataCollector] All {args.num_episodes} episodes done "
          f"in {elapsed:.1f}s ({elapsed / args.num_episodes:.1f}s/ep)")

    collector.export_dataset(
        output_dir=os.path.join(args.output_dir, "dataset"),
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if _simulation_app is not None:
            _simulation_app.close()
