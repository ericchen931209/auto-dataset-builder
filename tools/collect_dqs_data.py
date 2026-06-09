"""
DQS Training Data Collection (V1.0 Experiment)

Downloads COCO128, creates quality variants, fine-tunes YOLOv11n on each,
and records (DQS features, mAP@0.5) pairs for Neural DQS training.

Usage:
    python3 tools/collect_dqs_data.py [--variants N] [--epochs E] [--outfile data.csv]
    python3 tools/collect_dqs_data.py --quick     # validation-only, no training

Strategy:
    For each quality variant we sub-sample or degrade the reference dataset,
    then fine-tune the last YOLOv11n layer for E epochs on the variant and
    evaluate mAP@0.5 on a fixed held-out split.

    30 variants × 3 epochs × ~1.5 min/variant ≈ 45 min on CPU (16 threads).
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ─── Config ───────────────────────────────────────────────────────────────────

COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
IMG_SIZE     = 320
WORKERS_N    = 0          # 0 = main thread (stable on CPU)
DEFAULT_EPOCHS = 3


@dataclass
class ExperimentRecord:
    variant_id: str
    variant_type: str
    n_images: int
    # DQS features
    annotation_quality: float
    sharpness: float
    clip_diversity: float
    lighting_diversity: float
    pose_diversity: float
    class_balance: float
    # Target
    map50: float
    map50_95: float
    # Meta
    training_epochs: int
    elapsed_s: float


# ─── Dataset download ─────────────────────────────────────────────────────────

def download_coco128(dest_dir: str) -> str:
    """Download and extract COCO128. Returns path to extracted dataset root."""
    import urllib.request, zipfile
    dest = Path(dest_dir)
    zip_path = dest / "coco128.zip"

    if (dest / "coco128").exists():
        print("  COCO128 already downloaded, skipping.")
        return str(dest / "coco128")

    print(f"  Downloading COCO128…")
    urllib.request.urlretrieve(COCO128_URL, str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(dest))
    zip_path.unlink()
    return str(dest / "coco128")


def _load_coco128(coco_root: str) -> tuple[list[str], list[str]]:
    """Return (image_paths, label_paths) for the train split."""
    img_dir = Path(coco_root) / "images" / "train2017"
    lbl_dir = Path(coco_root) / "labels" / "train2017"
    imgs = sorted(img_dir.glob("*.jpg"))
    lbls = [lbl_dir / (p.stem + ".txt") for p in imgs]
    return [str(p) for p in imgs], [str(p) for p in lbls]


# ─── Variant builders ─────────────────────────────────────────────────────────

def _copy_variant(img_paths, lbl_paths, out_img, out_lbl):
    """Copy selected images+labels to variant directories."""
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)
    for ip, lp in zip(img_paths, lbl_paths):
        shutil.copy2(ip, out_img)
        if Path(lp).exists():
            shutil.copy2(lp, out_lbl)
        else:
            Path(out_lbl, Path(ip).stem + ".txt").touch()


def _add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _copy_labels_with_missing(src_lbls: list[str], out_lbl: str, missing_frac: float,
                               rng: random.Random) -> None:
    """Copy labels but blank out missing_frac of them (simulates annotation gaps)."""
    Path(out_lbl).mkdir(parents=True, exist_ok=True)
    for lp in src_lbls:
        dst = os.path.join(out_lbl, Path(lp).name)
        if rng.random() < missing_frac:
            Path(dst).write_text("")   # empty = no annotations
        elif Path(lp).exists():
            shutil.copy2(lp, dst)
        else:
            Path(dst).write_text("")


def _copy_labels_with_noise(src_lbls: list[str], out_lbl: str, shift_frac: float,
                             rng: random.Random) -> None:
    """Copy labels with random bbox shifts (simulates annotation noise)."""
    Path(out_lbl).mkdir(parents=True, exist_ok=True)
    for lp in src_lbls:
        dst = os.path.join(out_lbl, Path(lp).name)
        if not Path(lp).exists():
            Path(dst).write_text("")
            continue
        lines_out = []
        for line in Path(lp).read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                cx = float(cx) + rng.uniform(-shift_frac, shift_frac)
                cy = float(cy) + rng.uniform(-shift_frac, shift_frac)
                cx = max(0.01, min(0.99, cx))
                cy = max(0.01, min(0.99, cy))
                lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w} {h}")
            else:
                lines_out.append(line)
        Path(dst).write_text("\n".join(lines_out))


def build_variants(
    img_paths: list[str],
    lbl_paths: list[str],
    base_dir: str,
    n_variants: int,
    rng: random.Random,
) -> list[dict]:
    """
    Build N quality variants of the reference dataset.
    Returns list of dicts with variant metadata.
    """
    variants = []
    n = len(img_paths)

    def _make_variant(variant_id, variant_type, selected_imgs, selected_lbls,
                      transform_img=None):
        vdir = os.path.join(base_dir, variant_id)
        out_img = os.path.join(vdir, "images", "train")
        out_lbl = os.path.join(vdir, "labels", "train")

        if transform_img:
            Path(out_img).mkdir(parents=True, exist_ok=True)
            Path(out_lbl).mkdir(parents=True, exist_ok=True)
            for ip, lp in zip(selected_imgs, selected_lbls):
                img = cv2.imread(ip)
                if img is not None:
                    img = transform_img(img)
                    cv2.imwrite(os.path.join(out_img, Path(ip).name), img)
                if Path(lp).exists():
                    shutil.copy2(lp, out_lbl)
                else:
                    Path(out_lbl, Path(ip).stem + ".txt").touch()
        else:
            _copy_variant(selected_imgs, selected_lbls, out_img, out_lbl)

        variants.append({
            "id": variant_id,
            "type": variant_type,
            "img_dir": out_img,
            "lbl_dir": out_lbl,
            "n_images": len(selected_imgs),
        })

    # 1. Full dataset baseline
    _make_variant("v00_full", "baseline", img_paths, lbl_paths)

    # 2–8. Random subsets (varying size: 20%, 40%, 60%, 80%)
    for i, frac in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        k = max(10, int(n * frac))
        sel = rng.sample(list(range(n)), k)
        _make_variant(
            f"v{i+1:02d}_subset_{int(frac*100)}pct",
            "size_variant",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
        )

    # 9–14. Gaussian noise (sigma 5, 15, 25, 35, 50, 75)
    for i, sigma in enumerate([5, 15, 25, 35, 50, 75]):
        k = max(10, int(n * 0.7))
        sel = rng.sample(list(range(n)), k)
        sigma_val = sigma
        _make_variant(
            f"v{i+9:02d}_noise_s{sigma}",
            "noise",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, s=sigma_val: _add_gaussian_noise(img, s),
        )

    # 15–19. Dark images (brightness factor 0.2, 0.4, 0.5, 0.6, 0.8)
    for i, factor in enumerate([0.2, 0.4, 0.5, 0.6, 0.8]):
        k = max(10, int(n * 0.6))
        sel = rng.sample(list(range(n)), k)
        f_val = factor
        _make_variant(
            f"v{i+15:02d}_dark_{int(factor*100)}",
            "brightness",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, f=f_val: _adjust_brightness(img, f),
        )

    # 20–24. Blur (kernel 3, 7, 11, 21, 41)
    for i, ksize in enumerate([3, 7, 11, 21, 41]):
        k = max(10, int(n * 0.7))
        sel = rng.sample(list(range(n)), k)
        k_val = ksize
        _make_variant(
            f"v{i+20:02d}_blur_k{ksize}",
            "blur",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, kz=k_val: cv2.GaussianBlur(img, (kz|1, kz|1), 0),
        )

    # 25–29. Mixed degradation + small size
    degradations = [
        ("noise+dark",   lambda img: _adjust_brightness(_add_gaussian_noise(img, 20), 0.6)),
        ("noise+blur",   lambda img: cv2.GaussianBlur(_add_gaussian_noise(img, 15), (7, 7), 0)),
        ("dark+blur",    lambda img: cv2.GaussianBlur(_adjust_brightness(img, 0.5), (11, 11), 0)),
        ("heavy_degrade",lambda img: cv2.GaussianBlur(_adjust_brightness(_add_gaussian_noise(img, 30), 0.4), (15, 15), 0)),
        ("small_clean",  None),
    ]
    for i, (dtype, tfm) in enumerate(degradations):
        k = 15 if dtype == "small_clean" else max(10, int(n * 0.5))
        sel = rng.sample(list(range(n)), k)
        _make_variant(
            f"v{i+25:02d}_{dtype.replace('+', '_')}",
            dtype,
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=tfm,
        )

    # 30–35. Label-missing variants (AQ drops as more labels removed)
    for i, miss_frac in enumerate([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
        vid = f"v{i+30:02d}_lbl_miss_{int(miss_frac*100)}"
        vdir = os.path.join(base_dir, vid)
        out_img = os.path.join(vdir, "images", "train")
        out_lbl = os.path.join(vdir, "labels", "train")
        Path(out_img).mkdir(parents=True, exist_ok=True)
        sel = rng.sample(list(range(n)), max(10, int(n * 0.7)))
        s_imgs = [img_paths[j] for j in sel]
        s_lbls = [lbl_paths[j] for j in sel]
        for ip in s_imgs:
            shutil.copy2(ip, out_img)
        _copy_labels_with_missing(s_lbls, out_lbl, miss_frac, rng)
        variants.append({"id": vid, "type": "label_missing",
                         "img_dir": out_img, "lbl_dir": out_lbl,
                         "n_images": len(s_imgs)})

    # 36–38. Label-noise variants (AQ drops as bbox shift increases)
    for i, shift in enumerate([0.03, 0.10, 0.20]):
        vid = f"v{i+36:02d}_lbl_noise_{int(shift*100)}"
        vdir = os.path.join(base_dir, vid)
        out_img = os.path.join(vdir, "images", "train")
        out_lbl = os.path.join(vdir, "labels", "train")
        Path(out_img).mkdir(parents=True, exist_ok=True)
        sel = rng.sample(list(range(n)), max(10, int(n * 0.7)))
        s_imgs = [img_paths[j] for j in sel]
        s_lbls = [lbl_paths[j] for j in sel]
        for ip in s_imgs:
            shutil.copy2(ip, out_img)
        _copy_labels_with_noise(s_lbls, out_lbl, shift, rng)
        variants.append({"id": vid, "type": "label_noise",
                         "img_dir": out_img, "lbl_dir": out_lbl,
                         "n_images": len(s_imgs)})

    # 39–43. Extra blur gradations (finer steps)
    for i, ksize in enumerate([5, 9, 15, 25, 35]):
        k = max(10, int(n * 0.7))
        sel = rng.sample(list(range(n)), k)
        k_val = ksize
        _make_variant(
            f"v{i+39:02d}_blur2_k{ksize}",
            "blur",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, kz=k_val: cv2.GaussianBlur(img, (kz|1, kz|1), 0),
        )

    # 44–51. More noise gradations (fill in the curve)
    for i, sigma in enumerate([2, 8, 12, 20, 30, 40, 60, 100]):
        k = max(10, int(n * 0.7))
        sel = rng.sample(list(range(n)), k)
        s_val = sigma
        _make_variant(
            f"v{i+44:02d}_noise2_s{sigma}",
            "noise",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, s=s_val: _add_gaussian_noise(img, s),
        )

    # 52–58. More brightness levels (dark + overexposed)
    for i, factor in enumerate([0.1, 0.15, 0.25, 0.35, 0.55, 0.7, 0.9]):
        k = max(10, int(n * 0.6))
        sel = rng.sample(list(range(n)), k)
        f_val = factor
        _make_variant(
            f"v{i+52:02d}_bright2_{int(factor*100)}",
            "brightness",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, f=f_val: _adjust_brightness(img, f),
        )

    # 59–65. More label-missing gradations
    for i, miss_frac in enumerate([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80]):
        vid = f"v{i+59:02d}_lbl_miss2_{int(miss_frac*100)}"
        vdir = os.path.join(base_dir, vid)
        out_img = os.path.join(vdir, "images", "train")
        out_lbl = os.path.join(vdir, "labels", "train")
        Path(out_img).mkdir(parents=True, exist_ok=True)
        sel = rng.sample(list(range(n)), max(10, int(n * 0.7)))
        s_imgs = [img_paths[j] for j in sel]
        s_lbls = [lbl_paths[j] for j in sel]
        for ip in s_imgs:
            shutil.copy2(ip, out_img)
        _copy_labels_with_missing(s_lbls, out_lbl, miss_frac, rng)
        variants.append({"id": vid, "type": "label_missing",
                         "img_dir": out_img, "lbl_dir": out_lbl,
                         "n_images": len(s_imgs)})

    # 66–80. Dense blur sweep (k=2,4,6,8,12,14,17,20,28,34,38,45,51,55,61)
    for i, ksize in enumerate([2,4,6,8,12,14,17,20,28,34,38,45,51,55,61]):
        k = max(10, int(n * 0.7))
        sel = rng.sample(list(range(n)), k)
        k_val = ksize
        _make_variant(
            f"v{i+66:02d}_blur3_k{ksize}",
            "blur",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, kz=k_val: cv2.GaussianBlur(img, (kz|1, kz|1), 0),
        )

    # 81–88. Overexposed + more extreme dark
    for i, factor in enumerate([0.05, 0.08, 0.12, 0.18, 1.1, 1.3, 1.6, 2.0]):
        k = max(10, int(n * 0.6))
        sel = rng.sample(list(range(n)), k)
        f_val = factor
        _make_variant(
            f"v{i+81:02d}_bright3_{int(factor*100)}",
            "brightness",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=lambda img, f=f_val: _adjust_brightness(img, f),
        )

    # 89–96. More combined degradations
    combos = [
        ("blur_dark_mild",   lambda img: cv2.GaussianBlur(_adjust_brightness(img, 0.7), (7,7), 0)),
        ("blur_dark_mod",    lambda img: cv2.GaussianBlur(_adjust_brightness(img, 0.5), (15,15), 0)),
        ("blur_dark_heavy",  lambda img: cv2.GaussianBlur(_adjust_brightness(img, 0.3), (25,25), 0)),
        ("noise_dark_mild",  lambda img: _adjust_brightness(_add_gaussian_noise(img, 10), 0.7)),
        ("noise_dark_mod",   lambda img: _adjust_brightness(_add_gaussian_noise(img, 25), 0.5)),
        ("noise_blur_mild",  lambda img: cv2.GaussianBlur(_add_gaussian_noise(img, 8), (5,5), 0)),
        ("noise_blur_mod",   lambda img: cv2.GaussianBlur(_add_gaussian_noise(img, 20), (11,11), 0)),
        ("noise_blur_heavy", lambda img: cv2.GaussianBlur(_add_gaussian_noise(img, 40), (21,21), 0)),
    ]
    for i, (cname, tfm) in enumerate(combos):
        k = max(10, int(n * 0.6))
        sel = rng.sample(list(range(n)), k)
        _make_variant(
            f"v{i+89:02d}_{cname}",
            "combined",
            [img_paths[j] for j in sel],
            [lbl_paths[j] for j in sel],
            transform_img=tfm,
        )

    return variants[:n_variants]


# ─── DQS feature extraction ───────────────────────────────────────────────────

def compute_dqs_features(img_dir: str, lbl_dir: str) -> dict:
    from models.dqs.feature_extractor import extract_features
    try:
        feats = extract_features(img_dir, lbl_dir)
        return {
            "annotation_quality": round(feats.annotation_quality, 4),
            "sharpness":          round(feats.sharpness, 4),
            "clip_diversity":     round(feats.clip_diversity, 4),
            "lighting_diversity": round(feats.lighting_diversity, 4),
            "pose_diversity":     round(feats.pose_diversity, 4),
            "class_balance":      round(feats.class_balance, 4),
        }
    except Exception as e:
        print(f"    [DQS warn] {e}")
        return {k: 0.0 for k in
                ["annotation_quality","sharpness","clip_diversity","lighting_diversity",
                 "pose_diversity","class_balance"]}


# ─── YOLO fine-tune & eval ────────────────────────────────────────────────────

def make_dataset_yaml(variant: dict, coco128_root: str, yaml_path: str):
    """Write a minimal dataset.yaml for this variant."""
    # Validation set = COCO128 val split (fixed across all variants)
    val_img = str(Path(coco128_root) / "images" / "train2017")  # COCO128 has no separate val
    content = f"""path: {variant['img_dir'].replace('/images/train', '')}
train: images/train
val: {val_img}

nc: 80
names: {list(range(80))}
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def run_yolo_experiment(
    variant: dict,
    coco128_yaml: str,
    project_dir: str,
    epochs: int,
    img_size: int,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Fine-tune YOLOv11n on variant for `epochs` epochs.
    Returns (mAP50, mAP50-95) evaluated on the fixed COCO128 val set.
    """
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")   # pre-trained weights downloaded automatically

    # Write variant yaml
    yaml_path = os.path.join(variant["img_dir"].replace("/images/train", ""), "dataset.yaml")
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

    img_dir_parent = str(Path(variant["img_dir"]).parent.parent)
    val_dir = str(Path(coco128_yaml).parent / "images" / "train2017")

    with open(yaml_path, "w") as f:
        f.write(f"path: {img_dir_parent}\ntrain: images/train\nval: {val_dir}\n"
                f"nc: 80\nnames: [" +
                ", ".join(f"'class{i}'" for i in range(80)) + "]\n")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=8,
        workers=WORKERS_N,
        project=project_dir,
        name=variant["id"],
        verbose=False,
        plots=False,
        save=False,
        device=device,
    )

    map50    = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    return map50, map50_95


def run_quick_eval(variant: dict, coco128_yaml: str) -> tuple[float, float]:
    """
    No-training quick mode: run YOLOv11n inference on variant images and
    use mean-confidence of detections as mAP proxy.
    Returns (proxy_score, 0.0).
    """
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    img_dir = variant["img_dir"]
    imgs = list(Path(img_dir).glob("*.jpg"))[:30]   # sample up to 30 for speed
    if not imgs:
        return 0.0, 0.0

    confs = []
    for img_path in imgs:
        try:
            preds = model(str(img_path), conf=0.1, verbose=False)
            for pred in preds:
                if pred.boxes is not None and len(pred.boxes):
                    confs.extend(pred.boxes.conf.tolist())
        except Exception:
            pass

    proxy = float(np.mean(confs)) if confs else 0.0
    return round(proxy, 4), 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",  type=int, default=30, help="Number of variants (max 30)")
    parser.add_argument("--epochs",    type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsize",   type=int, default=IMG_SIZE)
    parser.add_argument("--outfile",   type=str, default="dqs_training_data.csv")
    parser.add_argument("--device",    type=str, default="cpu", help="Training device: cpu, cuda, xpu")
    parser.add_argument("--quick",     action="store_true",
                        help="Skip YOLO training; use confidence proxy (fast, no GPU needed)")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip variants already present in --outfile")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    mode = "QUICK (confidence proxy)" if args.quick else f"FULL ({args.epochs} epochs)"
    print(f"  DQS Data Collection — {args.variants} variants — {mode}")
    print(f"{'='*65}\n")

    # Load already-done variants if resuming
    done_ids: set = set()
    if args.resume and Path(args.outfile).exists():
        with open(args.outfile) as f:
            done_ids = {row["variant_id"] for row in csv.DictReader(f)}
        print(f"  Resuming — {len(done_ids)} variants already recorded\n")

    with tempfile.TemporaryDirectory() as workdir:
        # 1. Download COCO128
        print("[1/3] Dataset setup")
        coco_root = download_coco128(workdir)
        img_paths, lbl_paths = _load_coco128(coco_root)
        coco128_yaml = str(Path(coco_root) / "coco128.yaml")
        print(f"  COCO128 loaded: {len(img_paths)} images\n")

        # 2. Build variants
        print("[2/3] Building quality variants")
        rng = random.Random(42)
        variants = build_variants(img_paths, lbl_paths, workdir, args.variants, rng)
        print(f"  {len(variants)} variants created\n")

        # 3. Run experiments
        print("[3/3] Running experiments\n")
        records: list[ExperimentRecord] = []
        project_dir = os.path.join(workdir, "runs")

        for i, variant in enumerate(variants):
            vid = variant["id"]
            if vid in done_ids:
                print(f"  [{i+1:02d}/{len(variants)}] {vid}  → already done, skip")
                continue

            print(f"  [{i+1:02d}/{len(variants)}] {vid}  ({variant['type']}, {variant['n_images']} imgs)", end=" ", flush=True)
            t0 = time.perf_counter()

            # DQS features
            feats = compute_dqs_features(variant["img_dir"], variant["lbl_dir"])

            # mAP
            try:
                if args.quick:
                    map50, map50_95 = run_quick_eval(variant, coco128_yaml)
                else:
                    map50, map50_95 = run_yolo_experiment(
                        variant, coco128_yaml, project_dir,
                        epochs=args.epochs, img_size=args.imgsize,
                        device=args.device,
                    )
            except Exception as e:
                print(f"\n    [WARN] YOLO failed: {e}")
                map50, map50_95 = 0.0, 0.0

            elapsed = time.perf_counter() - t0
            print(f"  mAP50={map50:.4f}  DQS≈{np.mean(list(feats.values())):.3f}  ({elapsed:.1f}s)")

            rec = ExperimentRecord(
                variant_id=vid,
                variant_type=variant["type"],
                n_images=variant["n_images"],
                **feats,
                map50=map50,
                map50_95=map50_95,
                training_epochs=0 if args.quick else args.epochs,
                elapsed_s=round(elapsed, 1),
            )
            records.append(rec)

            # Incremental save (so we can resume after interruption)
            write_header = not Path(args.outfile).exists() or i == 0
            with open(args.outfile, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(rec).keys()))
                if write_header and not done_ids:
                    w.writeheader()
                w.writerow(asdict(rec))

    # Summary
    all_map50 = [r.map50 for r in records]
    print(f"\n{'─'*65}")
    print(f"  Collected {len(records)} records  →  {args.outfile}")
    if all_map50:
        print(f"  mAP50 range: {min(all_map50):.4f} – {max(all_map50):.4f}"
              f"  mean={np.mean(all_map50):.4f}")
    print(f"{'─'*65}\n")
    print("Next step: python3 tools/train_neural_dqs.py --data", args.outfile)


if __name__ == "__main__":
    main()
