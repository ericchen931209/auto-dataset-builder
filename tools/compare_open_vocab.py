"""
Quick comparison: Stage 1 closed-vocab YOLOv11 (COCO-80) vs open-vocab
YOLO-World on a small sample of the COCO2017 motorcycle test images.

Usage:
    python3 tools/compare_open_vocab.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.annotator.yolo_annotator import run_yolo_batch
from workers.annotator.open_vocab_detector import run_yolo_world_batch

IMAGES_DIR = Path("data/motorcycle_coco/yolo_dataset/manual/images/test")
N_SAMPLES = 5

# Open-vocab classes deliberately go beyond COCO-80 to demonstrate the
# "natural language -> arbitrary class" capability.
OPEN_VOCAB_CLASSES = ["motorcycle", "scooter", "helmet", "license plate"]


def main():
    images = sorted(IMAGES_DIR.glob("*.jpg"))[:N_SAMPLES]
    image_paths = [str(p) for p in images]
    print(f"Comparing on {len(image_paths)} images:\n  " + "\n  ".join(p.name for p in images))

    print("\n=== Stage 1 (yolo11, COCO-80, target_classes=['motorcycle']) ===")
    yolo11_results = run_yolo_batch(
        image_paths=image_paths,
        model_path="yolo11n.pt",
        target_classes=["motorcycle"],
        confidence_threshold=0.25,
    )
    for r in yolo11_results:
        names = [f"{b.class_name}({b.confidence:.2f})" for b in r.boxes]
        print(f"  {Path(r.image_path).name}: {names}")

    print(f"\n=== Stage 1 (yolo_world, open-vocab classes={OPEN_VOCAB_CLASSES}) ===")
    world_results = run_yolo_world_batch(
        image_paths=image_paths,
        class_names=OPEN_VOCAB_CLASSES,
        confidence_threshold=0.25,
    )
    for r in world_results:
        if not r.success:
            print(f"  {Path(r.image_path).name}: ERROR - {r.error}")
            continue
        names = [f"{b.class_name}({b.confidence:.2f})" for b in r.boxes]
        print(f"  {Path(r.image_path).name}: {names}")


if __name__ == "__main__":
    main()
