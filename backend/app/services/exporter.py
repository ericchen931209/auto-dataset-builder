"""
Dataset exporter — YOLO format, COCO JSON, and zip archive.
"""
import hashlib
import json
import os
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class ExportManifest:
    dataset_name: str
    version_tag: str
    export_format: str
    num_images: int
    num_annotations: int
    class_names: list[str]
    split: dict[str, int]          # {"train": N, "val": N, "test": N}
    checksum: str                  # SHA-256 of zip file
    created_at: str
    zip_path: str


# ─── Train/val/test split ─────────────────────────────────────────────────────

def _split_files(
    image_paths: list[str],
    train: float = 0.7,
    val: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic split of image paths into train/val/test."""
    paths = list(image_paths)
    random.Random(seed).shuffle(paths)
    n = len(paths)
    n_train = int(n * train)
    n_val   = int(n * val)
    return {
        "train": paths[:n_train],
        "val":   paths[n_train : n_train + n_val],
        "test":  paths[n_train + n_val :],
    }


def _read_yolo_label(label_path: str) -> list[dict]:
    boxes = []
    try:
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    boxes.append({
                        "class_id": int(p[0]),
                        "cx": float(p[1]), "cy": float(p[2]),
                        "w":  float(p[3]), "h":  float(p[4]),
                    })
    except OSError:
        pass
    return boxes


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── YOLO export ──────────────────────────────────────────────────────────────

def export_yolo(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    dataset_name: str,
    class_names: list[str],
    version_tag: str = "v1.0",
    train: float = 0.7,
    val: float = 0.2,
) -> ExportManifest:
    """
    Export dataset in YOLO format with train/val/test split.

    Output structure::

        output_dir/
          images/train/  val/  test/
          labels/train/  val/  test/
          dataset.yaml
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out     = Path(output_dir)

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    splits = _split_files([str(p) for p in image_files], train=train, val=val)

    total_annotations = 0
    split_counts: dict[str, int] = {}

    for split_name, paths in splits.items():
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        split_counts[split_name] = len(paths)

        for img_path in paths:
            stem = Path(img_path).stem
            shutil.copy2(img_path, out / "images" / split_name / Path(img_path).name)

            lbl_src = lbl_dir / f"{stem}.txt"
            lbl_dst = out / "labels" / split_name / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(str(lbl_src), str(lbl_dst))
                total_annotations += len(_read_yolo_label(str(lbl_src)))
            else:
                lbl_dst.touch()

    # dataset.yaml
    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(class_names)}\nnames: {class_names}\n"
    )
    (out / "dataset.yaml").write_text(yaml_content)

    # Zip
    zip_path = str(out) + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent))

    return ExportManifest(
        dataset_name=dataset_name,
        version_tag=version_tag,
        export_format="yolo",
        num_images=len(image_files),
        num_annotations=total_annotations,
        class_names=class_names,
        split=split_counts,
        checksum=_sha256(zip_path),
        created_at=datetime.utcnow().isoformat(),
        zip_path=zip_path,
    )


# ─── COCO JSON export ─────────────────────────────────────────────────────────

def export_coco(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    dataset_name: str,
    class_names: list[str],
    version_tag: str = "v1.0",
) -> ExportManifest:
    """
    Export dataset in COCO JSON format (all images in one split).

    Output structure::

        output_dir/
          images/           (all images flat)
          annotations.json  (COCO format)
    """
    import cv2

    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out     = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    coco: dict = {
        "info": {
            "description": dataset_name,
            "version": version_tag,
            "date_created": datetime.utcnow().isoformat(),
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    ann_id = 1
    for img_id, img_path in enumerate(image_files, start=1):
        shutil.copy2(str(img_path), out / "images" / img_path.name)

        # Get image dimensions
        img = cv2.imread(str(img_path))
        h, w = (img.shape[:2] if img is not None else (0, 0))

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        for box in _read_yolo_label(str(lbl_path)):
            # YOLO [cx,cy,w,h] → COCO [x,y,w,h] in pixels
            bw = box["w"] * w
            bh = box["h"] * h
            x  = (box["cx"] * w) - bw / 2
            y  = (box["cy"] * h) - bh / 2
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box["class_id"],
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    ann_path = out / "annotations.json"
    ann_path.write_text(json.dumps(coco, indent=2))

    # Zip
    zip_path = str(out) + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent))

    return ExportManifest(
        dataset_name=dataset_name,
        version_tag=version_tag,
        export_format="coco",
        num_images=len(image_files),
        num_annotations=ann_id - 1,
        class_names=class_names,
        split={"all": len(image_files)},
        checksum=_sha256(zip_path),
        created_at=datetime.utcnow().isoformat(),
        zip_path=zip_path,
    )
