"""
"ADB+AL" baseline for Section 6.2: one active-learning iteration on top of the
ADB-annotated training set.

1. Run the ADB-trained YOLOv11n model on the unlabeled pool (200 images).
2. Select the most uncertain images (max motorcycle-confidence < threshold).
3. Annotate the selected images with the same three-stage ADB pipeline.
4. Merge them into the ADB training set to form the ADB+AL training set
   (adb_al/images/train + adb_al/labels/train).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ultralytics import YOLO
from workers.annotator.three_stage_pipeline import run_three_stage_pipeline

ROOT = "data/motorcycle_coco"
MOTO_CLASS_IDX = 3  # COCO class index for "motorcycle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb-model", default=os.path.join(ROOT, "runs/adb/weights/best.pt"))
    ap.add_argument("--pool-dir", default=os.path.join(ROOT, "pool_images"))
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                     help="images with max motorcycle confidence below this are 'uncertain'")
    ap.add_argument("--max-select", type=int, default=100)
    args = ap.parse_args()

    model = YOLO(args.adb_model)
    files = sorted(f for f in os.listdir(args.pool_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = [os.path.join(args.pool_dir, f) for f in files]

    # ── Step 1+2: uncertainty sampling on the pool ──────────────────────────
    scores = []
    for i in range(0, len(paths), 16):
        batch = paths[i:i + 16]
        results = model.predict(batch, conf=0.01, classes=[MOTO_CLASS_IDX], verbose=False, device="cpu")
        for r in results:
            confs = [float(b.conf[0]) for b in r.boxes]
            max_conf = max(confs) if confs else 0.0
            scores.append((r.path, max_conf))

    scores.sort(key=lambda x: x[1])  # ascending: most uncertain first
    uncertain = [p for p, c in scores if c < args.conf_threshold][:args.max_select]
    print(f"Selected {len(uncertain)}/{len(files)} pool images as uncertain "
          f"(max_conf < {args.conf_threshold})")

    # ── Step 3: ADB three-stage annotation on selected pool images ──────────
    al_img_dir = os.path.join(ROOT, "yolo_dataset/adb_al/images/train")
    al_lbl_dir = os.path.join(ROOT, "yolo_dataset/adb_al/labels/train")
    os.makedirs(al_img_dir, exist_ok=True)
    os.makedirs(al_lbl_dir, exist_ok=True)

    # Copy ADB labels for the original 477 training images (images already symlinked by split script)
    adb_lbl_dir = os.path.join(ROOT, "yolo_dataset/adb/labels/train")
    n_base = 0
    for fn in os.listdir(adb_lbl_dir):
        shutil.copy(os.path.join(adb_lbl_dir, fn), os.path.join(al_lbl_dir, fn))
        n_base += 1

    # Symlink selected pool images into adb_al/images/train
    for p in uncertain:
        dst = os.path.join(al_img_dir, os.path.basename(p))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(p), dst)

    summary = run_three_stage_pipeline(
        image_paths=uncertain,
        labels_dir=al_lbl_dir,
        yolo_model="yolo11n.pt",
        confidence_threshold=0.25,
        target_classes=["motorcycle"],
        class_map={"motorcycle": 0},
        sam2_checkpoint="checkpoints/sam2_hiera_tiny.pt",
        sam2_config="sam2_hiera_t.yaml",
        llm_confidence_threshold=0.25,
    )

    stats = {
        "n_base_train": n_base,
        "n_pool_total": len(files),
        "n_selected_uncertain": len(uncertain),
        "selected_total_boxes": summary.total_boxes,
        "n_total_train": n_base + len(uncertain),
    }
    print(json.dumps(stats, indent=2))
    with open(os.path.join(ROOT, "adb_al_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
