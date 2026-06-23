"""
Export WarehouseGPT World Model to TorchScript + TensorRT for Triton Inference Server.

Workflow:
  1. Load trained PyTorch checkpoint
  2. Trace/script to TorchScript (.pt)
  3. Export to ONNX with dynamic axes
  4. Validate ONNX model
  5. Build TensorRT FP16 engine via Polygraphy
  6. Organise model repository structure for Triton

Usage:
  python export_world_model.py \\
      --checkpoint /models/world_model_v2.ckpt \\
      --output-dir /model-repository/world_model \\
      --version 2 \\
      --precision fp16 \\
      --batch-sizes 1 4 8 16 32

Requirements:
  pip install torch onnx onnxruntime-gpu tensorrt polygraphy tritonclient
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("export_world_model")


# ---------------------------------------------------------------------------
# Model Architecture Stubs
# (In production, import from warehousegpt.world_model.transformer.model)
# ---------------------------------------------------------------------------

class WorldModelForExport(nn.Module):
    """
    Thin wrapper around the WarehouseGPT world model that exposes
    the exact tensor interfaces expected by Triton.

    In production, instantiate the real model:
        from warehousegpt.world_model.transformer.model import WarehouseWorldModel
        model = WarehouseWorldModel.load_from_checkpoint(checkpoint_path)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 24,
        max_seq_len: int = 3136,
        num_agents: int = 64,
        prediction_horizon: int = 30,
        occupancy_height: int = 200,
        occupancy_width: int = 200,
        future_seq_len: int = 3136,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        self.prediction_horizon = prediction_horizon
        self.occupancy_height = occupancy_height
        self.occupancy_width = occupancy_width
        self.future_seq_len = future_seq_len

        # Video token transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.video_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)

        # Agent state encoder
        self.agent_encoder = nn.Linear(12, embed_dim)

        # Occupancy encoder (lightweight CNN)
        self.occ_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((14, 14)),
        )

        # Cross-modal fusion transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)

        # Output heads
        self.future_token_head = nn.Linear(embed_dim, embed_dim)
        self.risk_score_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, prediction_horizon),
            nn.Sigmoid(),
        )
        self.occupancy_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 10 * occupancy_height // 8 * occupancy_width // 8),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, prediction_horizon * 3),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        video_tokens: torch.Tensor,    # [B, S, E]
        agent_states: torch.Tensor,    # [B, A, 12]
        occupancy_map: torch.Tensor,   # [B, H, W, 4]
        attention_mask: torch.Tensor,  # [B, S] bool
    ) -> tuple[
        torch.Tensor,  # future_tokens  [B, S, E]
        torch.Tensor,  # risk_scores    [B, A, T]
        torch.Tensor,  # occupancy      [B, 10, H, W]
        torch.Tensor,  # trajectories   [B, A, T, 3]
        torch.Tensor,  # anomaly_scores [B, A]
    ]:
        B, S, E = video_tokens.shape
        A = agent_states.shape[1]

        # --- Video token encoding ---
        key_padding_mask = ~attention_mask  # True = ignore
        video_features = self.video_encoder(
            video_tokens.to(torch.float32),
            src_key_padding_mask=key_padding_mask,
        )  # [B, S, E]

        # --- Agent state encoding ---
        agent_features = self.agent_encoder(agent_states)  # [B, A, E]

        # --- Occupancy encoding ---
        occ_input = occupancy_map.permute(0, 3, 1, 2).float()  # [B, 4, H, W]
        occ_features = self.occ_encoder(occ_input)              # [B, E, 14, 14]
        occ_features = occ_features.flatten(2).transpose(1, 2)  # [B, 196, E]

        # --- Fusion: agents attend to video + occupancy ---
        memory = torch.cat([video_features[:, :256, :], occ_features], dim=1)
        fused = self.fusion_decoder(agent_features, memory)  # [B, A, E]

        # --- Output heads ---
        future_tokens = self.future_token_head(video_features)  # [B, S, E]

        risk_scores = self.risk_score_head(fused)                      # [B, A, T]

        occ_logits = self.occupancy_head(fused.mean(dim=1))            # [B, 10*H'*W']
        H_out = self.occupancy_height // 8
        W_out = self.occupancy_width // 8
        occ_prob = torch.sigmoid(occ_logits).view(B, 10, H_out, W_out)
        # Upsample to full resolution
        occupancy = torch.nn.functional.interpolate(
            occ_prob.view(B * 10, 1, H_out, W_out),
            size=(self.occupancy_height, self.occupancy_width),
            mode="bilinear",
            align_corners=False,
        ).view(B, 10, self.occupancy_height, self.occupancy_width)

        traj_flat = self.trajectory_head(fused)                        # [B, A, T*3]
        trajectories = traj_flat.view(B, A, self.prediction_horizon, 3)

        anomaly_scores = self.anomaly_head(fused).squeeze(-1)          # [B, A]

        # Cast outputs back to fp16 where appropriate
        return (
            future_tokens.half(),
            risk_scores,
            occupancy,
            trajectories,
            anomaly_scores,
        )


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device) -> WorldModelForExport:
    """Load model from PyTorch Lightning checkpoint."""
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint_path_obj = Path(checkpoint_path)

    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = WorldModelForExport()

    if checkpoint_path_obj.suffix in (".ckpt", ".pt", ".pth"):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle PyTorch Lightning checkpoint format
        if "state_dict" in state:
            # Strip 'model.' prefix if present (Lightning convention)
            sd = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning("Missing keys in checkpoint: %s", missing[:10])
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected[:10])
        else:
            model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path_obj.suffix}")

    model = model.to(device).eval()
    logger.info("Checkpoint loaded successfully.")
    return model


def get_dummy_inputs(
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, ...]:
    """Generate dummy inputs matching Triton config tensor shapes."""
    video_tokens  = torch.randn(batch_size, 3136, 512,  device=device, dtype=dtype)
    agent_states  = torch.randn(batch_size, 64,   12,   device=device, dtype=torch.float32)
    occupancy_map = torch.randn(batch_size, 200,  200,  4, device=device, dtype=torch.float32)
    # Random boolean mask — ~80% unmasked
    attention_mask = torch.rand(batch_size, 3136, device=device) > 0.2
    return video_tokens, agent_states, occupancy_map, attention_mask


def export_torchscript(
    model: WorldModelForExport,
    output_dir: Path,
    version: int,
    device: torch.device,
) -> Path:
    """Trace model to TorchScript and save."""
    ts_dir = output_dir / str(version)
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_path = ts_dir / "model.pt"

    logger.info("Tracing model to TorchScript...")
    dummy_inputs = get_dummy_inputs(batch_size=1, device=device)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_inputs, strict=False)
        # Optimise for inference
        traced = torch.jit.optimize_for_inference(traced)

    torch.jit.save(traced, str(ts_path))
    logger.info("TorchScript saved: %s (%.1f MB)", ts_path, ts_path.stat().st_size / 1e6)
    return ts_path


def export_onnx(
    model: WorldModelForExport,
    output_path: Path,
    opset_version: int = 17,
    device: torch.device = torch.device("cuda"),
) -> Path:
    """Export model to ONNX with dynamic batch axis."""
    import onnx
    from onnx import checker

    logger.info("Exporting to ONNX (opset %d)...", opset_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_inputs = get_dummy_inputs(batch_size=4, device=device)
    input_names  = ["video_tokens", "agent_states", "occupancy_map", "attention_mask"]
    output_names = [
        "future_tokens", "risk_scores", "occupancy",
        "trajectory_predictions", "anomaly_scores",
    ]

    dynamic_axes = {
        "video_tokens":           {0: "batch_size"},
        "agent_states":           {0: "batch_size"},
        "occupancy_map":          {0: "batch_size"},
        "attention_mask":         {0: "batch_size"},
        "future_tokens":          {0: "batch_size"},
        "risk_scores":            {0: "batch_size"},
        "occupancy":              {0: "batch_size"},
        "trajectory_predictions": {0: "batch_size"},
        "anomaly_scores":         {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

    # Validate
    onnx_model = onnx.load(str(output_path))
    checker.check_model(onnx_model)
    logger.info("ONNX export validated: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def validate_onnx(onnx_path: Path, device: str = "cuda") -> None:
    """Run ONNX Runtime inference to validate correctness."""
    import onnxruntime as ort

    logger.info("Validating ONNX with ORT...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
        else ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)

    # Generate numpy inputs
    batch = 2
    inputs = {
        "video_tokens":   np.random.randn(batch, 3136, 512).astype(np.float16),
        "agent_states":   np.random.randn(batch, 64, 12).astype(np.float32),
        "occupancy_map":  np.random.randn(batch, 200, 200, 4).astype(np.float32),
        "attention_mask": (np.random.rand(batch, 3136) > 0.2),
    }

    t0 = time.perf_counter()
    outputs = sess.run(None, inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "ORT inference OK | latency=%.1fms | outputs: %s",
        latency_ms,
        [o.shape for o in outputs],
    )


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
    max_batch_sizes: list[int] = None,
    workspace_gb: float = 4.0,
) -> Path:
    """Build TensorRT engine using Polygraphy."""
    try:
        from polygraphy.backend.onnx import OnnxFromPath
        from polygraphy.backend.trt import (
            CreateConfig,
            Profile,
            TrtFromNetwork,
            engine_from_network,
            network_from_onnx_path,
            save_engine,
        )
    except ImportError:
        logger.error("Polygraphy not available. Install: pip install polygraphy tensorrt")
        raise

    if max_batch_sizes is None:
        max_batch_sizes = [1, 4, 8, 16, 32]

    logger.info("Building TensorRT engine | precision=%s | max_batch=%d", precision, max(max_batch_sizes))
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt

    # Create optimization profiles for each batch size
    profiles = []
    for bs in max_batch_sizes:
        p = Profile()
        p.add("video_tokens",   min=(1, 3136, 512),       opt=(bs//2 or 1, 3136, 512),       max=(bs, 3136, 512))
        p.add("agent_states",   min=(1, 64, 12),           opt=(bs//2 or 1, 64, 12),           max=(bs, 64, 12))
        p.add("occupancy_map",  min=(1, 200, 200, 4),      opt=(bs//2 or 1, 200, 200, 4),      max=(bs, 200, 200, 4))
        p.add("attention_mask", min=(1, 3136),              opt=(bs//2 or 1, 3136),              max=(bs, 3136))
        profiles.append(p)

    config_kwargs: dict = {
        "tf32": True,
        "profiles": profiles,
        "memory_pool_limits": {
            trt.MemoryPoolType.WORKSPACE: int(workspace_gb * 1024**3),
        },
    }
    if precision == "fp16":
        config_kwargs["fp16"] = True
    elif precision == "int8":
        config_kwargs["int8"] = True
        logger.warning("INT8 requires calibration data — using PTQ with random data (accuracy may degrade).")

    build_config = CreateConfig(**config_kwargs)
    network_loader = network_from_onnx_path(str(onnx_path))
    engine = engine_from_network(network_loader, config=build_config)
    save_engine(engine, path=str(engine_path))

    logger.info("TensorRT engine saved: %s (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path


def benchmark_engine(
    engine_path: Path,
    batch_sizes: list[int] = None,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark TRT engine throughput and latency."""
    try:
        import tensorrt as trt
        from polygraphy.backend.trt import TrtRunner, engine_from_bytes
    except ImportError:
        logger.warning("Polygraphy not available — skipping benchmark.")
        return {}

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    results = {}

    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    logger.info("Benchmarking TRT engine...")
    with TrtRunner(engine_from_bytes(engine_bytes)) as runner:
        for bs in batch_sizes:
            inputs = {
                "video_tokens":   np.random.randn(bs, 3136, 512).astype(np.float16),
                "agent_states":   np.random.randn(bs, 64, 12).astype(np.float32),
                "occupancy_map":  np.random.randn(bs, 200, 200, 4).astype(np.float32),
                "attention_mask": (np.random.rand(bs, 3136) > 0.2),
            }

            # Warmup
            for _ in range(warmup_runs):
                runner.infer(feed_dict=inputs)

            # Benchmark
            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                runner.infer(feed_dict=inputs)
                latencies.append((time.perf_counter() - t0) * 1000)

            p50 = float(np.percentile(latencies, 50))
            p99 = float(np.percentile(latencies, 99))
            throughput = bs / (p50 / 1000)

            results[bs] = {"p50_ms": p50, "p99_ms": p99, "throughput_fps": throughput}
            logger.info(
                "  batch=%2d | p50=%.1fms | p99=%.1fms | throughput=%.0f FPS",
                bs, p50, p99, throughput,
            )

    return results


def create_triton_model_dir(
    base_output_dir: Path,
    version: int,
    ts_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
    config_src: Optional[Path] = None,
) -> None:
    """Create Triton-compatible model repository structure."""
    version_dir = base_output_dir / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)

    if ts_path and ts_path.exists():
        dst = version_dir / "model.pt"
        if not dst.exists():
            import shutil
            shutil.copy2(ts_path, dst)
        logger.info("TorchScript model placed: %s", dst)

    if engine_path and engine_path.exists():
        dst = version_dir / "model.plan"
        if not dst.exists():
            import shutil
            shutil.copy2(engine_path, dst)
        logger.info("TRT engine placed: %s", dst)

    if config_src and config_src.exists():
        import shutil
        shutil.copy2(config_src, base_output_dir / "config.pbtxt")
        logger.info("Config copied: %s", base_output_dir / "config.pbtxt")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export WarehouseGPT World Model to Triton-compatible format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to PyTorch Lightning checkpoint (.ckpt)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/model-repository/world_model",
        help="Triton model repository base directory",
    )
    p.add_argument(
        "--version",
        type=int,
        default=1,
        help="Model version number (creates versioned subdirectory)",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32],
        help="Optimization profile batch sizes for TensorRT",
    )
    p.add_argument(
        "--export-torchscript",
        action="store_true",
        default=True,
        help="Export TorchScript model",
    )
    p.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export ONNX model",
    )
    p.add_argument(
        "--export-trt",
        action="store_true",
        default=True,
        help="Build TensorRT engine",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Benchmark TRT engine after export",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for tracing and validation",
    )
    p.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    p.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GB",
    )
    p.add_argument(
        "--config-pbtxt",
        type=str,
        default="",
        help="Path to Triton config.pbtxt to copy into model repository",
    )
    p.add_argument(
        "--validate-onnx",
        action="store_true",
        default=True,
        help="Validate ONNX model with OnnxRuntime before TRT build",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available — using CPU (TRT export will fail).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_export"
    tmp_dir.mkdir(exist_ok=True)

    # Step 1: Load model
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, device)
    else:
        logger.warning("No checkpoint provided — using randomly-initialized model for testing.")
        model = WorldModelForExport().to(device).eval()

    if args.precision == "fp16":
        model = model.half()

    ts_path: Optional[Path] = None
    onnx_path: Optional[Path] = None
    engine_path: Optional[Path] = None

    # Step 2: TorchScript export
    if args.export_torchscript:
        ts_path = export_torchscript(model, output_dir, args.version, device)

    # Step 3: ONNX export
    if args.export_onnx or args.export_trt:
        onnx_path = tmp_dir / "world_model.onnx"
        export_onnx(model.float(), onnx_path, opset_version=args.opset_version, device=device)

        if args.validate_onnx:
            validate_onnx(onnx_path, device=args.device)

    # Step 4: TensorRT engine
    if args.export_trt and onnx_path:
        engine_path = tmp_dir / f"world_model_{args.precision}.plan"
        build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=args.precision,
            max_batch_sizes=args.batch_sizes,
            workspace_gb=args.workspace_gb,
        )

        # Place engine in versioned Triton directory
        version_dir = output_dir / str(args.version)
        version_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(engine_path, version_dir / "model.plan")
        logger.info("TRT engine placed in model repository: %s", version_dir / "model.plan")

    # Step 5: Place config
    config_src = Path(args.config_pbtxt) if args.config_pbtxt else None
    create_triton_model_dir(output_dir, args.version, ts_path, engine_path, config_src)

    # Step 6: Benchmark
    benchmark_results: dict = {}
    if args.benchmark and engine_path and engine_path.exists():
        benchmark_results = benchmark_engine(
            engine_path=version_dir / "model.plan" if engine_path else engine_path,  # type: ignore[possibly-undefined]
            batch_sizes=args.batch_sizes,
        )

    # Step 7: Write export manifest
    manifest = {
        "model_name": "world_model",
        "version": args.version,
        "precision": args.precision,
        "checkpoint": args.checkpoint,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pytorch_version": torch.__version__,
        "batch_sizes": args.batch_sizes,
        "artifacts": {
            "torchscript": str(ts_path) if ts_path else None,
            "onnx": str(onnx_path) if onnx_path else None,
            "tensorrt": str(engine_path) if engine_path else None,
        },
        "benchmark": benchmark_results,
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Export manifest written: %s", manifest_path)
    logger.info("Export complete. Model repository: %s", output_dir)

    # Print summary
    print("\n" + "="*60)
    print(f"  WarehouseGPT World Model Export Summary")
    print("="*60)
    print(f"  Output directory : {output_dir}")
    print(f"  Version          : {args.version}")
    print(f"  Precision        : {args.precision}")
    print(f"  TorchScript      : {'OK' if ts_path else 'SKIPPED'}")
    print(f"  ONNX             : {'OK' if onnx_path else 'SKIPPED'}")
    print(f"  TensorRT engine  : {'OK' if engine_path else 'SKIPPED'}")
    if benchmark_results:
        print(f"\n  Benchmark results:")
        for bs, r in benchmark_results.items():
            print(f"    batch={bs:2d} | p50={r['p50_ms']:.1f}ms | "
                  f"p99={r['p99_ms']:.1f}ms | {r['throughput_fps']:.0f} FPS")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
