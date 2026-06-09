"""
ADB Benchmark Script (V1.0)

Measures pipeline throughput and DQS quality on a synthetic dataset.
Produces a machine-readable JSON report suitable for paper Table I.

Usage:
    python3 tools/benchmark.py [--images N] [--outfile results.json]

Output metrics:
    - Images processed per second (dedup, clean, annotate, export)
    - DQS scores across N synthetic datasets
    - Export zip sizes
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ─── Synthetic dataset generator ──────────────────────────────────────────────

def generate_synthetic_dataset(root: str, n: int, seed: int = 42):
    imgd = os.path.join(root, "images")
    lbld = os.path.join(root, "labels")
    os.makedirs(imgd, exist_ok=True)
    os.makedirs(lbld, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(f"{imgd}/{i:05d}.jpg", img)
        with open(f"{lbld}/{i:05d}.txt", "w") as f:
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            w  = rng.uniform(0.05, 0.40)
            h  = rng.uniform(0.05, 0.30)
            conf = rng.uniform(0.50, 1.00)
            n_cls = rng.integers(1, 4)  # vary class id for class balance metric
            f.write(f"{int(n_cls)} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {conf:.4f}\n")
    return imgd, lbld


# ─── Individual benchmarks ────────────────────────────────────────────────────

def bench_dedup(imgd: str, threshold: int = 8) -> dict:
    from workers.collector.deduplicator import remove_duplicates
    t0 = time.perf_counter()
    result = remove_duplicates(imgd, threshold=threshold)
    elapsed = time.perf_counter() - t0
    n = result["kept"] + result["removed"]
    return {
        "stage": "deduplication",
        "images": n,
        "kept": result["kept"],
        "removed": result["removed"],
        "elapsed_s": round(elapsed, 3),
        "images_per_s": round(n / elapsed, 1) if elapsed > 0 else 0,
    }


def bench_clean(imgd: str, lbld: str) -> dict:
    from workers.cleaner.cleaning_pipeline import clean_dataset
    t0 = time.perf_counter()
    report = clean_dataset(imgd, lbld, dry_run=True)   # dry_run — don't delete
    elapsed = time.perf_counter() - t0
    return {
        "stage": "cleaning",
        "images": report.total,
        "would_remove": report.removed,
        "elapsed_s": round(elapsed, 3),
        "images_per_s": round(report.total / elapsed, 1) if elapsed > 0 else 0,
    }


def bench_dqs(imgd: str, lbld: str) -> dict:
    from models.dqs.feature_extractor import extract_features
    from models.dqs.neural_dqs import predict
    t0 = time.perf_counter()
    feats = extract_features(imgd, lbld)
    score = predict(feats.to_vector())
    elapsed = time.perf_counter() - t0
    n = len(list(Path(imgd).glob("*.jpg")))
    return {
        "stage": "dqs_evaluation",
        "images": n,
        "dqs_score": round(score, 4),
        "features": {
            "annotation_quality": round(feats.annotation_quality, 4),
            "diversity": round(feats.diversity, 4),
            "lighting_diversity": round(feats.lighting_diversity, 4),
            "pose_diversity": round(feats.pose_diversity, 4),
            "class_balance": round(feats.class_balance, 4),
        },
        "elapsed_s": round(elapsed, 3),
        "images_per_s": round(n / elapsed, 1) if elapsed > 0 else 0,
    }


def bench_export(imgd: str, lbld: str, fmt: str = "yolo") -> dict:
    from backend.app.services.exporter import export_yolo, export_coco
    with tempfile.TemporaryDirectory() as outd:
        t0 = time.perf_counter()
        if fmt == "yolo":
            manifest = export_yolo(imgd, lbld, outd, "BenchmarkDS", ["object"], "v1.0")
        else:
            manifest = export_coco(imgd, lbld, outd, "BenchmarkDS", ["object"], "v1.0")
        elapsed = time.perf_counter() - t0
        zip_size = os.path.getsize(manifest.zip_path)
        return {
            "stage": f"export_{fmt}",
            "images": manifest.num_images,
            "annotations": manifest.num_annotations,
            "zip_size_kb": round(zip_size / 1024, 1),
            "elapsed_s": round(elapsed, 3),
            "images_per_s": round(manifest.num_images / elapsed, 1) if elapsed > 0 else 0,
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ADB Pipeline Benchmark")
    parser.add_argument("--images", type=int, default=100,
                        help="Number of synthetic images to generate (default: 100)")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ADB Benchmark  —  {args.images} synthetic images")
    print(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as root:
        print(f"Generating {args.images} synthetic images…")
        imgd, lbld = generate_synthetic_dataset(root, args.images)

        benchmarks = []

        print("  [1/4] Deduplication…", end=" ", flush=True)
        r = bench_dedup(imgd)
        benchmarks.append(r)
        print(f"{r['images_per_s']} img/s")

        print("  [2/4] Image cleaning (dry_run)…", end=" ", flush=True)
        r = bench_clean(imgd, lbld)
        benchmarks.append(r)
        print(f"{r['images_per_s']} img/s")

        print("  [3/4] DQS evaluation…", end=" ", flush=True)
        r = bench_dqs(imgd, lbld)
        benchmarks.append(r)
        print(f"DQS={r['dqs_score']}  {r['images_per_s']} img/s")

        print("  [4a/4] YOLO export…", end=" ", flush=True)
        r = bench_export(imgd, lbld, "yolo")
        benchmarks.append(r)
        print(f"{r['zip_size_kb']} KB  {r['images_per_s']} img/s")

        print("  [4b/4] COCO export…", end=" ", flush=True)
        r = bench_export(imgd, lbld, "coco")
        benchmarks.append(r)
        print(f"{r['zip_size_kb']} KB  {r['images_per_s']} img/s")

    report = {
        "config": {"images": args.images},
        "results": benchmarks,
    }

    print(f"\n{'─'*60}")
    print("  Summary")
    print(f"{'─'*60}")
    for b in benchmarks:
        print(f"  {b['stage']:<25} {b['elapsed_s']:>6.3f}s  {b['images_per_s']:>8.1f} img/s")
    print(f"{'─'*60}\n")

    if args.outfile:
        with open(args.outfile, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {args.outfile}\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
