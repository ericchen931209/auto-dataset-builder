"""
Build the Section 6.2 "Motorcycle Detection" benchmark dataset from COCO2017.

Selects N images from COCO train2017 that contain at least one ground-truth
"motorcycle" annotation, downloads them, and writes single-class YOLO labels
(class 0 = motorcycle) derived from the COCO human annotations. These labels
serve as the "Manual" ground-truth baseline for Section 6.2.

A separate, disjoint pool of motorcycle images is also written out (unlabeled)
for use as the active-learning candidate pool (ADB+AL).
"""
import argparse
import json
import os
import random
import urllib.request
import concurrent.futures as cf

COCO_IMG_URL = "http://images.cocodataset.org/train2017/{file_name}"


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def download_one(args):
    file_name, dest_path = args
    if os.path.exists(dest_path):
        return True
    try:
        urllib.request.urlretrieve(COCO_IMG_URL.format(file_name=file_name), dest_path)
        return True
    except Exception as e:
        print(f"  failed {file_name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="data/motorcycle_coco/annotations/instances_train2017.json")
    ap.add_argument("--out-dir", default="data/motorcycle_coco")
    ap.add_argument("--n-labeled", type=int, default=600, help="images for the main labeled benchmark")
    ap.add_argument("--n-pool", type=int, default=200, help="extra images for AL unlabeled pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    print("Loading COCO annotations...")
    d = json.load(open(args.annotations))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    moto_id = [k for k, v in cats.items() if v == "motorcycle"][0]

    images = {im["id"]: im for im in d["images"]}
    anns_by_img = {}
    for ann in d["annotations"]:
        if ann["category_id"] == moto_id:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_ids = sorted(anns_by_img.keys())
    print(f"Total images with motorcycle annotations: {len(img_ids)}")

    rng = random.Random(args.seed)
    rng.shuffle(img_ids)

    n_total = args.n_labeled + args.n_pool
    if n_total > len(img_ids):
        raise SystemExit(f"Requested {n_total} images but only {len(img_ids)} available")

    labeled_ids = img_ids[: args.n_labeled]
    pool_ids = img_ids[args.n_labeled : n_total]

    img_dir = os.path.join(args.out_dir, "images")
    lbl_dir = os.path.join(args.out_dir, "labels_manual")
    pool_dir = os.path.join(args.out_dir, "pool_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    # Download labeled set
    download_jobs = []
    for iid in labeled_ids:
        fn = images[iid]["file_name"]
        download_jobs.append((fn, os.path.join(img_dir, fn)))
    for iid in pool_ids:
        fn = images[iid]["file_name"]
        download_jobs.append((fn, os.path.join(pool_dir, fn)))

    print(f"Downloading {len(download_jobs)} images with {args.workers} workers...")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(download_one, download_jobs))
    print(f"Downloaded: {sum(results)}/{len(results)}")

    # Write YOLO labels for labeled set (Manual ground truth)
    print("Writing Manual ground-truth YOLO labels...")
    for iid in labeled_ids:
        im = images[iid]
        fn = im["file_name"]
        w, h = im["width"], im["height"]
        lines = []
        for ann in anns_by_img[iid]:
            cx, cy, bw, bh = coco_bbox_to_yolo(ann["bbox"], w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = os.path.splitext(fn)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # Save manifest
    manifest = {
        "labeled_files": [images[i]["file_name"] for i in labeled_ids],
        "pool_files": [images[i]["file_name"] for i in pool_ids],
        "moto_category_id": moto_id,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Labeled: {len(labeled_ids)}, Pool: {len(pool_ids)}")


if __name__ == "__main__":
    main()
