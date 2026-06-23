#!/usr/bin/env python3
"""
scripts/download_dataset.py
============================
Download the NVIDIA PhysicalAI warehouse operations dataset from Hugging Face
and organise it into train / val / test splits.

Dataset
-------
    nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes

Reference: https://huggingface.co/datasets/nvidia/
           PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes

Usage
-----
::

    # Full download (default output: ./data/warehouse_scenes/)
    python scripts/download_dataset.py

    # Custom output directory
    python scripts/download_dataset.py --output-dir /mnt/nvme/data/warehouse

    # Download specific split only
    python scripts/download_dataset.py --splits train

    # Dry-run (show what would be downloaded)
    python scripts/download_dataset.py --dry-run

    # Skip checksum validation (faster for re-runs)
    python scripts/download_dataset.py --no-verify

Environment variables
---------------------
    HF_TOKEN     : Hugging Face access token (required for gated datasets)
    HF_ENDPOINT  : Override the Hugging Face endpoint (e.g. for mirrors)

Requirements
------------
    pip install huggingface_hub>=0.23 tqdm rich
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

DATASET_REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes"

# Official split ratios (applied when the dataset ships as a single archive)
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

# Expected subdirectories / file patterns per split
SPLIT_PATTERNS: dict[str, list[str]] = {
    "train": ["train/", "training/", "*_train_*"],
    "val": ["val/", "validation/", "*_val_*", "*_valid_*"],
    "test": ["test/", "testing/", "*_test_*"],
}


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _try_import_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


class _TqdmFallback:
    """Very simple progress display when tqdm is not available."""

    def __init__(self, total: int = 0, desc: str = "", unit: str = "it") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._start = time.time()
        print(f"{desc}: 0/{total} {unit}", end="\r", flush=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        elapsed = time.time() - self._start
        rate = self.n / max(elapsed, 1e-9)
        pct = 100 * self.n / max(self.total, 1)
        print(
            f"\r{self.desc}: {self.n}/{self.total} {self.unit} "
            f"[{pct:.1f}%]  {rate:.1f} {self.unit}/s   ",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()  # newline


def _make_progress_bar(total: int, desc: str, unit: str = "file"):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    except ImportError:
        return _TqdmFallback(total=total, desc=desc, unit=unit)


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(
    output_dir: Path,
    checksum_file: Path,
) -> tuple[int, int]:
    """
    Verify file checksums against a JSON manifest.

    Expected manifest format::

        {
            "files": [
                {"path": "relative/path/file.mp4", "sha256": "abc123..."}
            ]
        }

    Returns
    -------
    (passed, failed) counts.
    """
    if not checksum_file.exists():
        logger.warning("Checksum manifest not found: %s — skipping verification.", checksum_file)
        return 0, 0

    with checksum_file.open() as fh:
        manifest: dict[str, Any] = json.load(fh)

    file_records = manifest.get("files", [])
    if not file_records:
        logger.warning("Checksum manifest is empty.")
        return 0, 0

    passed = failed = 0
    bar = _make_progress_bar(len(file_records), "Verifying checksums")

    for record in file_records:
        rel_path: str = record["path"]
        expected_sha: str = record["sha256"].lower()
        file_path = output_dir / rel_path

        if not file_path.exists():
            logger.error("MISSING: %s", rel_path)
            failed += 1
        else:
            actual_sha = _sha256_file(file_path)
            if actual_sha == expected_sha:
                passed += 1
            else:
                logger.error(
                    "CHECKSUM MISMATCH: %s\n  expected: %s\n  actual:   %s",
                    rel_path,
                    expected_sha,
                    actual_sha,
                )
                failed += 1

        bar.update(1)

    bar.close()
    return passed, failed


# ---------------------------------------------------------------------------
# Dataset organiser
# ---------------------------------------------------------------------------

def _organise_splits(
    raw_dir: Path,
    output_dir: Path,
    splits: list[str],
) -> dict[str, int]:
    """
    Organise downloaded files into train / val / test sub-directories.

    The dataset may ship with pre-defined splits (look for
    ``train/``, ``val/``, ``test/`` sub-directories) or as a flat
    collection (fall back to ratio-based assignment by file sort order).

    Returns a dict of {split_name: file_count}.
    """
    split_counts: dict[str, int] = {s: 0 for s in splits}

    # Check whether the dataset already has split directories
    predefined = {}
    for split in ["train", "val", "test"]:
        for pattern in SPLIT_PATTERNS[split]:
            candidate = raw_dir / pattern.rstrip("/").rstrip("*")
            if candidate.is_dir():
                predefined[split] = candidate
                break

    if predefined:
        logger.info("Dataset has pre-defined split directories.")
        for split in splits:
            src = predefined.get(split)
            if src is None:
                logger.warning("No pre-defined directory for split %r — skipping.", split)
                continue
            dst = output_dir / split
            dst.mkdir(parents=True, exist_ok=True)
            # Use symlinks when source is within output to avoid duplication
            if src.parent == output_dir:
                pass  # already in place
            else:
                _copy_or_link(src, dst)
            split_counts[split] = sum(1 for _ in dst.rglob("*") if _.is_file())
    else:
        # Flat collection — split by ratio
        logger.info(
            "No pre-defined splits found — splitting by ratio %s.",
            SPLIT_RATIOS,
        )
        all_files = sorted(raw_dir.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        n = len(all_files)

        boundaries = {
            "train": int(n * SPLIT_RATIOS["train"]),
            "val": int(n * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])),
            "test": n,
        }

        split_file_lists: dict[str, list[Path]] = {
            "train": all_files[: boundaries["train"]],
            "val": all_files[boundaries["train"] : boundaries["val"]],
            "test": all_files[boundaries["val"] :],
        }

        for split in splits:
            dst = output_dir / split
            dst.mkdir(parents=True, exist_ok=True)
            files = split_file_lists.get(split, [])
            bar = _make_progress_bar(len(files), f"Organising {split}")
            for src_file in files:
                rel = src_file.relative_to(raw_dir)
                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                bar.update(1)
            bar.close()
            split_counts[split] = len(files)

    return split_counts


def _copy_or_link(src: Path, dst: Path) -> None:
    """Copy src directory tree into dst."""
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(item, target)


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

def download_dataset(
    output_dir: Path,
    splits: list[str],
    verify: bool = True,
    dry_run: bool = False,
    hf_token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    """
    Download the PhysicalAI warehouse dataset from Hugging Face.

    Parameters
    ----------
    output_dir:   Root directory for organised data.
    splits:       Which splits to download / organise.
    verify:       Run SHA-256 checksum validation after download.
    dry_run:      Print what would be downloaded without actually downloading.
    hf_token:     Hugging Face access token.
    cache_dir:    Directory for raw HuggingFace cache (default: ~/.cache/huggingface).
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub>=0.23"
        )
        sys.exit(1)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset   : %s", DATASET_REPO_ID)
    logger.info("Output    : %s", output_dir)
    logger.info("Splits    : %s", splits)
    logger.info("Verify    : %s", verify)
    logger.info("Dry run   : %s", dry_run)

    if dry_run:
        logger.info("[DRY RUN] Would download from: https://huggingface.co/datasets/%s", DATASET_REPO_ID)
        logger.info("[DRY RUN] Splits to organise: %s", splits)
        logger.info("[DRY RUN] No files will be written.")
        return

    # --- Download via snapshot_download (downloads all files in the repo) ---
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset snapshot…")
    t_start = time.time()

    try:
        snapshot_path = snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            local_dir=str(raw_dir),
            token=hf_token or os.environ.get("HF_TOKEN"),
            cache_dir=str(cache_dir) if cache_dir else None,
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )
        logger.info("Snapshot downloaded to: %s (%.1fs)", snapshot_path, time.time() - t_start)
    except RepositoryNotFoundError:
        logger.error(
            "Repository not found: %s\n"
            "The dataset may be gated — set HF_TOKEN or use --token.",
            DATASET_REPO_ID,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        raise

    # --- Checksum verification ---
    if verify:
        logger.info("Verifying checksums…")
        checksum_file = raw_dir / "checksums.json"
        # Also look for dataset_infos.json (HuggingFace standard)
        if not checksum_file.exists():
            checksum_file = raw_dir / "dataset_infos.json"
        passed, failed = _verify_checksums(raw_dir, checksum_file)
        if failed > 0:
            logger.warning(
                "Checksum verification: %d passed, %d FAILED. "
                "Re-run download or use --no-verify to skip.",
                passed,
                failed,
            )
        else:
            logger.info(
                "Checksum verification: %d/%d files OK.", passed, passed + failed
            )

    # --- Organise into splits ---
    logger.info("Organising into splits: %s", splits)
    split_counts = _organise_splits(raw_dir, output_dir, splits)

    # --- Summary ---
    total_files = sum(split_counts.values())
    total_size_gb = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    ) / (1024 ** 3)

    logger.info("=" * 60)
    logger.info("Download complete in %.1fs", time.time() - t_start)
    logger.info("Output directory: %s", output_dir)
    logger.info("Total files     : %d", total_files)
    logger.info("Total size      : %.2f GB", total_size_gb)
    for split, count in split_counts.items():
        logger.info("  %-10s : %d files", split, count)
    logger.info("=" * 60)

    # Write a dataset manifest
    manifest = {
        "dataset": DATASET_REPO_ID,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "splits": split_counts,
        "total_files": total_files,
        "total_size_gb": round(total_size_gb, 3),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to: %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the NVIDIA PhysicalAI warehouse dataset from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/warehouse_scenes"),
        help="Root output directory (default: ./data/warehouse_scenes)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Splits to download and organise (default: all)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 checksum validation after download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (overrides HF_TOKEN env var)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for Hugging Face cache (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    download_dataset(
        output_dir=args.output_dir,
        splits=args.splits,
        verify=not args.no_verify,
        dry_run=args.dry_run,
        hf_token=args.token,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
