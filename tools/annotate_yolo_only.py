"""
"YOLO-only" baseline for Section 6.2: run a pretrained YOLOv11n model on the
training images and keep only "motorcycle" detections above a confidence
threshold as pseudo-labels (remapped to single class 0). No SAM2 refinement,
no LLM verification.
"""
import argparse
import os

from ultralytics import YOLO

MOTO_CLASS_IDX = 3  # COCO class index for "motorcycle" in yolo11n.pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/motorcycle_coco/yolo_dataset/yolo_only/images/train")
    ap.add_argument("--labels-out", default="data/motorcycle_coco/yolo_dataset/yolo_only/labels/train")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO("yolo11n.pt")
    files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = [os.path.join(args.images_dir, f) for f in files]

    os.makedirs(args.labels_out, exist_ok=True)

    total_boxes = 0
    for i in range(0, len(paths), 16):
        batch = paths[i:i + 16]
        batch_files = files[i:i + 16]
        results = model.predict(batch, conf=args.conf, classes=[MOTO_CLASS_IDX], verbose=False, device="cpu")
        for fname, r in zip(batch_files, results):
            stem = os.path.splitext(fname)[0]
            lines = []
            for box in r.boxes:
                cx, cy, w, h = box.xywhn[0].tolist()
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += len(lines)
            with open(os.path.join(args.labels_out, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Wrote pseudo-labels for {len(files)} images, {total_boxes} total boxes "
          f"({total_boxes / len(files):.2f} boxes/img)")


if __name__ == "__main__":
    main()
