"""
Split the 597-image motorcycle benchmark into train/val/test (80/10/10) and
lay out the YOLO directory structure for each Section 6.2 method variant
(manual / yolo_only / adb / adb_al). val/test always use the Manual
ground-truth labels so all four methods are scored against the same
reference. Only the train labels differ between methods (written by other
scripts); this script creates the "manual" train labels and the directory
skeleton (image symlinks) for the other three.
"""
import argparse
import json
import os
import random

ROOT = "data/motorcycle_coco"
METHODS = ["manual", "yolo_only", "adb", "adb_al"]


def link(src, dst):
    src = os.path.abspath(src)
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(ROOT, "images")
    lbl_dir = os.path.join(ROOT, "labels_manual")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    rng = random.Random(args.seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    print({k: len(v) for k, v in splits.items()})

    with open(os.path.join(ROOT, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    for method in METHODS:
        for split_name, split_files in splits.items():
            img_out = os.path.join(ROOT, "yolo_dataset", method, "images", split_name)
            lbl_out = os.path.join(ROOT, "yolo_dataset", method, "labels", split_name)
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)

            for fn in split_files:
                link(os.path.join(img_dir, fn), os.path.join(img_out, fn))

            # val/test: always manual ground truth. train: manual method only here;
            # other methods' train labels are written by their own annotation scripts.
            if split_name != "train" or method == "manual":
                for fn in split_files:
                    stem = os.path.splitext(fn)[0]
                    src_lbl = os.path.join(lbl_dir, stem + ".txt")
                    dst_lbl = os.path.join(lbl_out, stem + ".txt")
                    if os.path.exists(src_lbl) and not os.path.lexists(dst_lbl):
                        link(src_lbl, dst_lbl)

        yaml_path = os.path.join(ROOT, "yolo_dataset", method, "dataset.yaml")
        abs_root = os.path.abspath(os.path.join(ROOT, "yolo_dataset", method))
        with open(yaml_path, "w") as f:
            f.write(f"path: {abs_root}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n")
            f.write("names:\n  0: motorcycle\n")

    print("Done.")


if __name__ == "__main__":
    main()
