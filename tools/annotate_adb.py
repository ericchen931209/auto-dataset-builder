"""
"ADB" baseline for Section 6.2: run the full three-stage annotation pipeline
(YOLOv11 proposal -> SAM2 bbox refinement -> CLIP zero-shot verification) on
the training images and write single-class (motorcycle=0) YOLO labels.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from workers.annotator.three_stage_pipeline import run_three_stage_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/motorcycle_coco/yolo_dataset/adb/images/train")
    ap.add_argument("--labels-out", default="data/motorcycle_coco/yolo_dataset/adb/labels/train")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    image_paths = [os.path.join(args.images_dir, f) for f in files]

    os.makedirs(args.labels_out, exist_ok=True)

    t0 = time.time()
    summary = run_three_stage_pipeline(
        image_paths=image_paths,
        labels_dir=args.labels_out,
        yolo_model="yolo11n.pt",
        confidence_threshold=args.conf,
        target_classes=["motorcycle"],
        class_map={"motorcycle": 0},
        sam2_checkpoint="checkpoints/sam2_hiera_tiny.pt",
        sam2_config="sam2_hiera_t.yaml",
        llm_confidence_threshold=args.conf,
    )
    elapsed = time.time() - t0

    stats = {
        "total_images": summary.total_images,
        "total_boxes": summary.total_boxes,
        "sam2_refined": summary.sam2_refined,
        "sam2_fallback_images": summary.sam2_fallback_images,
        "llm_rejected": summary.llm_rejected,
        "llm_backend": summary.llm_backend,
        "elapsed_sec": elapsed,
    }
    print(json.dumps(stats, indent=2))
    with open("data/motorcycle_coco/adb_annotation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
