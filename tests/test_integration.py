"""
ADB End-to-End Integration Test (V1.0)

Runs the full pipeline on a synthetic mini-dataset (10 images, no GPU needed):
  1. Image deduplication
  2. Frame-based cleaning
  3. DQS feature extraction + Neural DQS score
  4. Active learning convergence checker
  5. YOLO export + zip
  6. COCO export + zip
  7. Version snapshot + diff

All tests use only in-memory / temp-dir data — no Docker, no internet, no GPU.
Run:
    python3 tests/test_integration.py
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import cv2
import traceback
from pathlib import Path

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
results = []

def test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, ""))
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        print(f"{FAIL} {name}\n       → {tb}")
        results.append((name, False, tb))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_dataset(root: str, n_good=8, n_dark=1, n_blur=1) -> tuple[str, str]:
    """Create synthetic images + labels in root/images and root/labels."""
    imgd = os.path.join(root, "images")
    lbld = os.path.join(root, "labels")
    os.makedirs(imgd, exist_ok=True)
    os.makedirs(lbld, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(n_good):
        img = (rng.integers(60, 200, (480, 640, 3))).astype(np.uint8)
        cv2.imwrite(f"{imgd}/{i:03d}.jpg", img)
        with open(f"{lbld}/{i:03d}.txt", "w") as f:
            cx, cy = 0.3 + 0.4 * rng.random(), 0.3 + 0.4 * rng.random()
            w, h   = 0.1 + 0.3 * rng.random(), 0.1 + 0.2 * rng.random()
            conf   = 0.6 + 0.35 * rng.random()
            f.write(f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {conf:.4f}\n")

    for i in range(n_good, n_good + n_dark):
        img = np.full((480, 640, 3), 3, dtype=np.uint8)
        cv2.imwrite(f"{imgd}/{i:03d}.jpg", img)

    for i in range(n_good + n_dark, n_good + n_dark + n_blur):
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        img = cv2.GaussianBlur(img, (101, 101), 0)
        cv2.imwrite(f"{imgd}/{i:03d}.jpg", img)

    return imgd, lbld


# ─── Integration tests ────────────────────────────────────────────────────────

def test_e2e_dedup():
    """Deduplication removes exact copies and keeps unique images."""
    from workers.collector.deduplicator import remove_duplicates
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f"{d}/orig.jpg", img)
        shutil.copy(f"{d}/orig.jpg", f"{d}/dup.jpg")
        img2 = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f"{d}/uniq.jpg", img2)
        r = remove_duplicates(d, threshold=8)
        assert r["removed"] == 1 and r["kept"] == 2


def test_e2e_cleaning():
    """Cleaning pipeline removes dark and blurry images."""
    from workers.cleaner.cleaning_pipeline import clean_dataset
    with tempfile.TemporaryDirectory() as root:
        imgd, lbld = _make_dataset(root, n_good=5, n_dark=2, n_blur=1)
        report = clean_dataset(imgd, lbld)
        assert report.removed >= 2        # CleaningReport is a dataclass
        assert report.kept <= 5


def test_e2e_dqs_pipeline():
    """Full DQS feature extraction + score prediction on synthetic data."""
    from models.dqs.feature_extractor import extract_features
    from models.dqs.neural_dqs import predict
    with tempfile.TemporaryDirectory() as root:
        imgd, lbld = _make_dataset(root, n_good=8, n_dark=0, n_blur=0)
        # extract_features takes directory paths, not lists
        feats = extract_features(imgd, lbld)
        assert 0.0 <= feats.annotation_quality <= 1.0
        assert 0.0 <= feats.diversity <= 1.0
        score = predict(feats.to_vector())
        assert 0.0 <= score <= 1.0


def test_e2e_active_learning_convergence():
    """AL convergence checker terminates correctly under synthetic DQS values."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.80, max_iterations=5, min_delta=0.01)
    # Simulate improving DQS until threshold
    scores = [0.50, 0.62, 0.71, 0.81]
    result = None
    for s in scores:
        result = checker.step(s, uncertain_count=max(1, int((0.81 - s) * 20)))
        if result.should_stop:
            break
    assert result.should_stop
    assert result.reason == StopReason.DQS_THRESHOLD
    assert checker.iteration == 4


def test_e2e_export_yolo():
    """Full YOLO export creates valid zip with correct split structure."""
    from backend.app.services.exporter import export_yolo
    with tempfile.TemporaryDirectory() as root:
        imgd, lbld = _make_dataset(root, n_good=10, n_dark=0, n_blur=0)
        outd = os.path.join(root, "export_yolo")
        os.makedirs(outd)
        manifest = export_yolo(imgd, lbld, outd, "IntegrationDS",
                               ["motorcycle"], "v1.0", train=0.7, val=0.2)
        assert manifest.num_images == 10
        assert manifest.num_annotations > 0
        assert manifest.export_format == "yolo"
        assert sum(manifest.split.values()) == 10
        assert Path(manifest.zip_path).exists()
        assert (Path(outd) / "dataset.yaml").exists()


def test_e2e_export_coco():
    """Full COCO export creates valid zip with annotations.json."""
    import json
    from backend.app.services.exporter import export_coco
    with tempfile.TemporaryDirectory() as root:
        imgd, lbld = _make_dataset(root, n_good=6, n_dark=0, n_blur=0)
        outd = os.path.join(root, "export_coco")
        os.makedirs(outd)
        manifest = export_coco(imgd, lbld, outd, "IntegrationDS",
                               ["motorcycle"], "v1.0")
        assert manifest.num_images == 6
        ann = json.loads((Path(outd) / "annotations.json").read_text())
        assert len(ann["images"]) == 6
        assert len(ann["annotations"]) == manifest.num_annotations
        assert ann["categories"][0]["name"] == "motorcycle"


def test_e2e_version_snapshot_and_diff():
    """Create two snapshots, verify diff detects image changes."""
    from backend.app.services.version_control import create_snapshot, diff_versions
    with tempfile.TemporaryDirectory() as root:
        imgd1, lbld1 = _make_dataset(root + "/v1", n_good=5, n_dark=0, n_blur=0)
        imgd2 = root + "/v2/images"
        lbld2 = root + "/v2/labels"
        os.makedirs(imgd2); os.makedirs(lbld2)
        # Copy 4 images verbatim from v1 → v2 (same bytes → unchanged)
        src_imgs = sorted(Path(imgd1).glob("*.jpg"))
        for p in src_imgs[:4]:
            shutil.copy(str(p), os.path.join(imgd2, p.name))
        # Add one new image (not in v1)
        rng = np.random.default_rng(99)
        cv2.imwrite(f"{imgd2}/new_999.jpg",
                    rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
        # v1 has 5 images; v2 has 4 (copied) + 1 (new) = 5
        # Diff: added=[new_999.jpg], removed=[src_imgs[4].name], unchanged=4

        snpd = root + "/snapshots"
        snap1 = create_snapshot(42, "v1.0", imgd1, lbld1, snpd)
        snap2 = create_snapshot(42, "v1.1", imgd2, lbld2, snpd)

        diff = diff_versions(snap1["snapshot_path"], snap2["snapshot_path"],
                             "v1.0", "v1.1")
        assert "new_999.jpg" in diff.added
        assert len(diff.removed) == 1
        assert diff.unchanged == 4


def test_e2e_full_pipeline_smoke():
    """Smoke test: dedup → clean → DQS → export all succeed without error."""
    from workers.collector.deduplicator import remove_duplicates
    from workers.cleaner.cleaning_pipeline import clean_dataset
    from models.dqs.feature_extractor import extract_features
    from models.dqs.neural_dqs import predict
    from backend.app.services.exporter import export_yolo

    with tempfile.TemporaryDirectory() as root:
        imgd, lbld = _make_dataset(root, n_good=8, n_dark=1, n_blur=1)
        outd = os.path.join(root, "out")
        os.makedirs(outd)

        # Step 1: dedup
        dedup = remove_duplicates(imgd, threshold=8)
        assert dedup["kept"] >= 8

        # Step 2: clean
        clean = clean_dataset(imgd, lbld)
        assert clean.removed >= 1            # CleaningReport is a dataclass

        # Step 3: DQS  (extract_features takes directory strings)
        feats = extract_features(imgd, lbld)
        score = predict(feats.to_vector())
        assert 0.0 <= score <= 1.0

        # Step 4: export
        manifest = export_yolo(imgd, lbld, outd, "SmokeDS", ["object"], "v1.0")
        assert Path(manifest.zip_path).exists()


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ADB Integration Test Suite (V1.0)")
    print("=" * 60 + "\n")

    test("E2E: deduplication",                  test_e2e_dedup)
    test("E2E: image cleaning pipeline",        test_e2e_cleaning)
    test("E2E: DQS extraction + prediction",    test_e2e_dqs_pipeline)
    test("E2E: active learning convergence",    test_e2e_active_learning_convergence)
    test("E2E: YOLO export + zip",              test_e2e_export_yolo)
    test("E2E: COCO export + zip",              test_e2e_export_coco)
    test("E2E: version snapshot + diff",        test_e2e_version_snapshot_and_diff)
    test("E2E: full pipeline smoke test",       test_e2e_full_pipeline_smoke)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed / {len(results)} total")
    print("=" * 60 + "\n")

    if failed:
        print("Failed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  ✗ {name}\n    {err}")
        sys.exit(1)
    else:
        print("  All integration tests passed! ✓\n")
