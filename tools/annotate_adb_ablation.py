"""
Component-wise ablation of the ADB three-stage pipeline (Section 7.5,
"left as future work" in the paper). Re-runs the YOLOv11->SAM2->CLIP-verify
pipeline used for the "ADB" baseline (tools/annotate_adb.py) with one or both
stages disabled:

  --variant no_sam2         : YOLOv11 -> CLIP zero-shot verify (Stage 2 skipped)
  --variant no_clip          : YOLOv11 -> SAM2 refinement only  (Stage 3 skipped)
  --variant no_sam2_no_clip  : YOLOv11 proposals only (Stages 2 & 3 skipped)
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
    ap.add_argument("--variant", required=True, choices=["no_sam2", "no_clip", "no_sam2_no_clip"])
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    images_dir = args.images_dir or f"data/motorcycle_coco/yolo_dataset/adb_{args.variant}/images/train"
    labels_out = args.labels_out or f"data/motorcycle_coco/yolo_dataset/adb_{args.variant}/labels/train"

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    image_paths = [os.path.join(images_dir, f) for f in files]

    os.makedirs(labels_out, exist_ok=True)

    t0 = time.time()
    summary = run_three_stage_pipeline(
        image_paths=image_paths,
        labels_dir=labels_out,
        yolo_model="yolo11n.pt",
        confidence_threshold=args.conf,
        target_classes=["motorcycle"],
        class_map={"motorcycle": 0},
        sam2_checkpoint="checkpoints/sam2_hiera_tiny.pt",
        sam2_config="sam2_hiera_t.yaml",
        llm_confidence_threshold=args.conf,
        skip_sam2=(args.variant in ("no_sam2", "no_sam2_no_clip")),
        skip_llm_verify=(args.variant in ("no_clip", "no_sam2_no_clip")),
    )
    elapsed = time.time() - t0

    stats = {
        "variant": f"adb_{args.variant}",
        "total_images": summary.total_images,
        "total_boxes": summary.total_boxes,
        "sam2_refined": summary.sam2_refined,
        "sam2_fallback_images": summary.sam2_fallback_images,
        "llm_rejected": summary.llm_rejected,
        "llm_backend": summary.llm_backend,
        "elapsed_sec": elapsed,
    }
    print(json.dumps(stats, indent=2))
    with open(f"data/motorcycle_coco/adb_{args.variant}_annotation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
