"""
Train YOLOv11n on a Section 6.2 motorcycle-detection variant (manual /
yolo_only / adb / adb_al) and evaluate mAP@0.5 / mAP@0.5:0.95 on the shared
test split (Manual ground truth).
"""
import argparse
import json
import os
import time

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["manual", "yolo_only", "adb", "adb_al"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--root", default="data/motorcycle_coco/yolo_dataset")
    ap.add_argument("--project", default="data/motorcycle_coco/runs")
    args = ap.parse_args()

    yaml_path = os.path.join(args.root, args.method, "dataset.yaml")

    model = YOLO("yolo11n.pt")

    t0 = time.time()
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=8,
        workers=0,
        project=args.project,
        name=args.method,
        exist_ok=True,
        verbose=False,
        plots=False,
        save=True,
        device="cpu",
    )
    train_time = time.time() - t0

    # Evaluate on test split
    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=args.imgsz,
        project=args.project,
        name=args.method + "_test",
        exist_ok=True,
        verbose=False,
        plots=False,
        device="cpu",
    )

    map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    result = {
        "method": args.method,
        "epochs": args.epochs,
        "train_time_sec": train_time,
        "map50": map50,
        "map50_95": map50_95,
    }
    print(json.dumps(result, indent=2))

    out_path = os.path.join("data/motorcycle_coco", f"result_{args.method}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
