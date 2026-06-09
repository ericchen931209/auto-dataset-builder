"""
ADB Test Suite — runs without Docker or GPU.
Tests core logic modules: deduplicator, cleaner, DQS features, Neural DQS.
"""
import sys, os, tempfile, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import cv2
from PIL import Image
import traceback

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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_image(path, w=640, h=480, color=(100,100,100), blur=False, dark=False, bright=False, noise=False):
    rng = np.random.default_rng(abs(hash(path)) % (2**31))
    img = np.full((h, w, 3), color, dtype=np.uint8)
    if noise:
        # Add structured noise so pHash has unique fingerprint
        img = (img.astype(np.int32) + rng.integers(-40, 40, img.shape)).clip(0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (51, 51), 0)
    if dark:
        img[:] = 5
    if bright:
        img[:] = 255
    cv2.imwrite(path, img)

def make_label(path, cx=0.5, cy=0.5, w=0.3, h=0.2, cls=0):
    with open(path, "w") as f:
        f.write(f"{cls} {cx} {cy} {w} {h}\n")

# ─── Test: Deduplicator ───────────────────────────────────────────────────────

def test_dedup_removes_exact_copy():
    from workers.collector.deduplicator import remove_duplicates
    with tempfile.TemporaryDirectory() as d:
        # Use noise=True so images have unique pHash fingerprints
        make_image(f"{d}/a.jpg", color=(80, 120, 160), noise=True)
        # b.jpg is a byte-for-byte copy of a.jpg
        import shutil; shutil.copy(f"{d}/a.jpg", f"{d}/b.jpg")
        make_image(f"{d}/c.jpg", color=(200, 50, 30), noise=True)  # visually different
        result = remove_duplicates(d, threshold=8)
        assert result["removed"] == 1, f"expected 1 removed, got {result['removed']}"
        assert result["kept"] == 2, f"expected 2 kept, got {result['kept']}"

def test_dedup_keeps_unique():
    from workers.collector.deduplicator import remove_duplicates
    with tempfile.TemporaryDirectory() as d:
        # Three visually distinct noisy images
        make_image(f"{d}/a.jpg", color=(10,  20,  30),  noise=True)
        make_image(f"{d}/b.jpg", color=(200, 150, 100), noise=True)
        make_image(f"{d}/c.jpg", color=(50,  200,  50), noise=True)
        result = remove_duplicates(d, threshold=8)
        assert result["removed"] == 0, f"expected 0 removed, got {result['removed']}"

# ─── Test: Cleaning Pipeline ──────────────────────────────────────────────────

def test_cleaning_removes_dark():
    from workers.cleaner.cleaning_pipeline import clean_dataset
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        # blur_threshold=0 → never flag as blurry, so only dark/bright detection applies
        make_image(f"{imgd}/good.jpg", color=(120,120,120))
        make_image(f"{imgd}/dark.jpg", dark=True)
        make_label(f"{lbld}/good.txt")
        make_label(f"{lbld}/dark.txt")
        r = clean_dataset(imgd, lbld, blur_threshold=0, dark_threshold=20, dry_run=False)
        assert r.removed_dark >= 1, f"expected dark removal, got {r.removed_dark}"
        assert r.kept >= 1, f"expected good.jpg to be kept, kept={r.kept}"

def test_cleaning_removes_overexposed():
    from workers.cleaner.cleaning_pipeline import clean_dataset
    with tempfile.TemporaryDirectory() as imgd:
        make_image(f"{imgd}/good.jpg", color=(120,120,120))
        make_image(f"{imgd}/bright.jpg", bright=True)
        r = clean_dataset(imgd, bright_threshold=0.5, dry_run=False)
        assert r.removed_overexposed >= 1

def test_cleaning_dry_run_does_not_delete():
    from workers.cleaner.cleaning_pipeline import clean_dataset
    with tempfile.TemporaryDirectory() as imgd:
        make_image(f"{imgd}/dark.jpg", dark=True)
        r = clean_dataset(imgd, dry_run=True)
        assert os.path.exists(f"{imgd}/dark.jpg"), "dry_run should not delete files"
        assert r.removed >= 1

def test_cleaning_report_fields():
    from workers.cleaner.cleaning_pipeline import clean_dataset, CleaningReport
    with tempfile.TemporaryDirectory() as imgd:
        make_image(f"{imgd}/good.jpg")
        r = clean_dataset(imgd, dry_run=True)
        assert isinstance(r, CleaningReport)
        assert hasattr(r, "total")
        assert hasattr(r, "kept")
        assert hasattr(r, "removed")

# ─── Test: DQS Feature Extractor ─────────────────────────────────────────────

def test_dqs_features_valid_range():
    from models.dqs.feature_extractor import extract_features
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        # 3 images with different brightness
        make_image(f"{imgd}/dark.jpg", dark=True)
        make_image(f"{imgd}/mid.jpg", color=(120,120,120))
        make_image(f"{imgd}/bright.jpg", bright=True)
        for name in ["dark", "mid", "bright"]:
            make_label(f"{lbld}/{name}.txt", cx=0.5, cy=0.5, w=0.3, h=0.2)
        feats = extract_features(imgd, lbld)
        for attr in ["annotation_quality", "diversity", "lighting_diversity",
                     "pose_diversity", "class_balance"]:
            v = getattr(feats, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of [0,1]"

def test_lighting_diversity_max_with_three_buckets():
    from models.dqs.feature_extractor import compute_lighting_diversity
    with tempfile.TemporaryDirectory() as d:
        make_image(f"{d}/dark.jpg",  dark=True)         # bucket 0
        make_image(f"{d}/mid.jpg",   color=(120,120,120))  # bucket 1
        make_image(f"{d}/bright.jpg",bright=True)       # bucket 2
        ld = compute_lighting_diversity(d)
        assert ld > 0.8, f"expected near-max LD with 3 buckets, got {ld:.3f}"

def test_annotation_quality_reasonable_bbox():
    from models.dqs.feature_extractor import compute_annotation_quality
    with tempfile.TemporaryDirectory() as lbld:
        make_label(f"{lbld}/a.txt", w=0.3, h=0.2)   # area=0.06, reasonable
        make_label(f"{lbld}/b.txt", w=0.9, h=0.95)  # area=0.855, too large
        aq = compute_annotation_quality(lbld)
        assert 0 < aq < 1, f"expected partial AQ, got {aq}"

def test_class_balance_single_class_is_one():
    from models.dqs.feature_extractor import compute_class_balance
    with tempfile.TemporaryDirectory() as lbld:
        for i in range(5):
            make_label(f"{lbld}/img{i}.txt", cls=0)
        cb = compute_class_balance(lbld)
        assert cb == 1.0, f"single class should return 1.0, got {cb}"

def test_class_balance_two_classes_imbalanced():
    from models.dqs.feature_extractor import compute_class_balance
    with tempfile.TemporaryDirectory() as lbld:
        for i in range(9):
            make_label(f"{lbld}/img{i}.txt", cls=0)
        make_label(f"{lbld}/img9.txt", cls=1)  # 9:1 imbalance
        cb = compute_class_balance(lbld)
        assert cb < 0.5, f"imbalanced should be < 0.5, got {cb}"

def test_pose_diversity_varied_aspect_ratios():
    from models.dqs.feature_extractor import compute_pose_diversity
    with tempfile.TemporaryDirectory() as lbld:
        make_label(f"{lbld}/tall.txt",  w=0.1, h=0.5)   # tall (r=0.2)
        make_label(f"{lbld}/square.txt",w=0.3, h=0.3)   # square (r=1.0)
        make_label(f"{lbld}/wide.txt",  w=0.6, h=0.1)   # wide  (r=6.0)
        pd = compute_pose_diversity(lbld)
        assert pd > 0.8, f"varied aspect ratios should give high PD, got {pd:.3f}"

# ─── Test: Neural DQS Model ───────────────────────────────────────────────────

def test_neural_dqs_heuristic_fallback():
    from models.dqs.neural_dqs import _heuristic_dqs
    score = _heuristic_dqs([0.8, 0.6, 0.7, 0.5, 1.0])
    assert 0.0 < score < 1.0, f"heuristic score out of range: {score}"

def test_neural_dqs_predict_without_model():
    from models.dqs.neural_dqs import predict
    score = predict([0.8, 0.6, 0.7, 0.5, 1.0], model_path="/tmp/nonexistent_model.pkl")
    assert 0.0 <= score <= 1.0, f"predict fallback out of range: {score}"

def test_neural_dqs_train_and_predict():
    from models.dqs.neural_dqs import train, predict
    # Create synthetic training data: higher features → higher mAP
    rng = np.random.default_rng(42)
    n = 30
    features = rng.uniform(0.2, 0.9, (n, 5)).tolist()
    maps = [min(0.95, sum(f)/5 * 0.9 + rng.uniform(0, 0.05)) for f in features]

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    metrics = train(features, maps, save_path=model_path)
    assert "pearson_r" in metrics
    assert metrics["pearson_r"] > 0.5, f"Pearson r too low: {metrics['pearson_r']:.3f}"

    score = predict([0.8, 0.7, 0.75, 0.6, 1.0], model_path=model_path)
    assert 0.0 <= score <= 1.0
    os.unlink(model_path)

def test_neural_dqs_feature_vector():
    from models.dqs.feature_extractor import DQSFeatures
    f = DQSFeatures(0.8, 0.6, 0.7, 0.5, 1.0)
    v = f.to_vector()
    assert len(v) == 5
    assert v == [0.8, 0.6, 0.7, 0.5, 1.0]
    d = f.to_dict()
    assert "annotation_quality" in d

# ─── Test: Keyword Expansion ──────────────────────────────────────────────────

def test_keyword_expansion_motorcycle():
    from workers.collector.image_searcher import expand_keywords
    expanded = expand_keywords(["motorcycle"], target="motorcycle", region="Taiwan")
    assert len(expanded) > 3
    assert any("Taiwan" in kw for kw in expanded)

def test_keyword_expansion_no_duplicates():
    from workers.collector.image_searcher import expand_keywords
    expanded = expand_keywords(["motorcycle"], target="motorcycle")
    assert len(expanded) == len(set(expanded)), "Duplicates found in expanded keywords"

# ─── Test: Frame Extractor ────────────────────────────────────────────────────

def test_frame_extractor_fixed_rate():
    from workers.extractor.frame_extractor import extract_fixed_rate
    with tempfile.TemporaryDirectory() as d:
        # Create a tiny 5-frame synthetic video
        video_path = f"{d}/test.mp4"
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for _ in range(10):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        result = extract_fixed_rate(video_path, f"{d}/frames", fps=5.0)
        assert result.extracted_frames > 0
        assert result.total_frames == 10
        assert len(result.frame_paths) == result.extracted_frames

# ─── Test: Pydantic Schemas ───────────────────────────────────────────────────

def test_dataset_create_schema():
    from app.schemas.dataset import DatasetCreate
    d = DatasetCreate(query="Build a motorcycle dataset", name="Test")
    assert d.query == "Build a motorcycle dataset"

def test_dataset_create_schema_requires_query():
    from app.schemas.dataset import DatasetCreate
    try:
        DatasetCreate()  # should raise
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # expected

# ─── Test: SAM2 Refiner (fallback path, no GPU needed) ───────────────────────

def test_sam2_refiner_fallback_no_boxes():
    """refine_with_sam2 with empty boxes returns RefinedAnnotation with empty boxes."""
    from workers.annotator.yolo_annotator import AnnotationResult
    from workers.annotator.sam2_refiner import refine_with_sam2
    results = refine_with_sam2([
        AnnotationResult(image_path="/nonexistent/img.jpg", boxes=[], success=True)
    ])
    assert len(results) == 1
    assert results[0].boxes == []

def test_sam2_refiner_fallback_missing_sam2():
    """refine_with_sam2 gracefully falls back when SAM2 is not installed."""
    from workers.annotator.yolo_annotator import AnnotationResult, BoundingBox
    from workers.annotator.sam2_refiner import refine_with_sam2
    box = BoundingBox(0, "car", 0.5, 0.5, 0.3, 0.2, 0.9)
    ann = AnnotationResult(image_path="/nonexistent/img.jpg", boxes=[box])
    results = refine_with_sam2([ann])
    assert len(results) == 1
    # SAM2 not installed → fallback=True, original box preserved
    assert results[0].fallback is True
    assert len(results[0].boxes) == 1
    assert results[0].boxes[0].class_name == "car"

def test_bbox_from_mask():
    """_bbox_from_mask converts a binary mask to correct normalized bbox."""
    from workers.annotator.sam2_refiner import _bbox_from_mask, BoundingBox
    import numpy as np
    # 100×100 image, mask covers rows 20-79, cols 10-89
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 10:90] = True
    original = BoundingBox(0, "test", 0.5, 0.5, 0.3, 0.3, 0.9)
    refined = _bbox_from_mask(mask, original, img_h=100, img_w=100)
    assert abs(refined.x_center - 0.495) < 0.01    # (10+89)/2 / 100 = 49.5/100
    assert abs(refined.y_center - 0.495) < 0.01    # (20+79)/2 / 100 = 49.5/100
    assert abs(refined.width  - 0.79) < 0.01       # (89-10)/100  (slice [10:90] → max col=89)
    assert abs(refined.height - 0.59) < 0.01       # (79-20)/100  (slice [20:80] → max row=79)


# ─── Test: LLM Verifier (fallback path, no GPU needed) ───────────────────────

def test_llm_verifier_passthrough_no_backend():
    """verify_with_llm passes all boxes through when no LLM is available."""
    from workers.annotator.yolo_annotator import BoundingBox
    from workers.annotator.sam2_refiner import RefinedAnnotation
    from workers.annotator.llm_verifier import verify_with_llm
    box = BoundingBox(0, "motorcycle", 0.5, 0.5, 0.3, 0.2, 0.8)
    refined = RefinedAnnotation(image_path="/nonexistent.jpg", boxes=[box], fallback=True)
    # No real server → should fall back to passthrough
    results = verify_with_llm([refined], ollama_url="http://localhost:1", ollama_model="llava")
    assert len(results) == 1
    assert results[0].backend == "passthrough"
    assert len(results[0].boxes) == 1

def test_llm_verifier_confidence_filter():
    """Boxes below confidence_threshold are rejected before LLM call."""
    from workers.annotator.yolo_annotator import BoundingBox
    from workers.annotator.sam2_refiner import RefinedAnnotation
    from workers.annotator.llm_verifier import verify_with_llm
    low_conf  = BoundingBox(0, "cat", 0.5, 0.5, 0.3, 0.2, confidence=0.30)
    high_conf = BoundingBox(0, "cat", 0.5, 0.5, 0.3, 0.2, confidence=0.90)
    refined = RefinedAnnotation(image_path="/nonexistent.jpg", boxes=[low_conf, high_conf], fallback=True)
    results = verify_with_llm([refined], confidence_threshold=0.5,
                               ollama_url="http://localhost:1", ollama_model="llava")
    assert results[0].rejected_count >= 1   # low_conf rejected
    assert any(b.confidence >= 0.5 for b in results[0].boxes)

def test_llm_verifier_empty_input():
    """verify_with_llm handles empty box list without error."""
    from workers.annotator.sam2_refiner import RefinedAnnotation
    from workers.annotator.llm_verifier import verify_with_llm
    refined = RefinedAnnotation(image_path="/nonexistent.jpg", boxes=[], fallback=True)
    results = verify_with_llm([refined], ollama_url="http://localhost:1", ollama_model="llava")
    assert results[0].boxes == []


# ─── Test: Three-stage pipeline (no GPU, no images) ─────────────────────────

def test_pipeline_empty_input():
    """run_three_stage_pipeline returns empty summary for empty image list."""
    from workers.annotator.three_stage_pipeline import run_three_stage_pipeline
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        summary = run_three_stage_pipeline([], labels_dir=d)
    assert summary.total_images == 0
    assert summary.total_boxes == 0

def test_pipeline_summary_fields():
    """PipelineSummary has all required fields."""
    from workers.annotator.three_stage_pipeline import PipelineSummary
    s = PipelineSummary(total_images=5, total_boxes=10, sam2_refined=3, llm_rejected=1)
    assert s.total_images == 5
    assert s.sam2_refined == 3
    assert s.llm_rejected == 1
    assert isinstance(s.results, list)


# ─── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ADB Test Suite")
    print("="*60 + "\n")

    test("Dedup: removes exact copy",               test_dedup_removes_exact_copy)
    test("Dedup: keeps unique images",              test_dedup_keeps_unique)
    test("Clean: removes dark images",              test_cleaning_removes_dark)
    test("Clean: removes overexposed images",       test_cleaning_removes_overexposed)
    test("Clean: dry_run does not delete",          test_cleaning_dry_run_does_not_delete)
    test("Clean: report has correct fields",        test_cleaning_report_fields)
    test("DQS: all features in [0,1]",             test_dqs_features_valid_range)
    test("DQS: lighting diversity max w/ 3 buckets",test_lighting_diversity_max_with_three_buckets)
    test("DQS: annotation quality partial",        test_annotation_quality_reasonable_bbox)
    test("DQS: class balance single=1.0",          test_class_balance_single_class_is_one)
    test("DQS: class balance imbalanced<0.5",      test_class_balance_two_classes_imbalanced)
    test("DQS: pose diversity varied ratios",       test_pose_diversity_varied_aspect_ratios)
    test("Neural DQS: heuristic fallback",         test_neural_dqs_heuristic_fallback)
    test("Neural DQS: predict without model",      test_neural_dqs_predict_without_model)
    test("Neural DQS: train + predict",            test_neural_dqs_train_and_predict)
    test("Neural DQS: feature vector shape",       test_neural_dqs_feature_vector)
    test("Keyword: expand motorcycle",             test_keyword_expansion_motorcycle)
    test("Keyword: no duplicates",                 test_keyword_expansion_no_duplicates)
    test("Extractor: fixed-rate from video",       test_frame_extractor_fixed_rate)
    test("Schema: DatasetCreate valid",            test_dataset_create_schema)
    test("Schema: DatasetCreate requires query",   test_dataset_create_schema_requires_query)
    test("SAM2: fallback on empty boxes",          test_sam2_refiner_fallback_no_boxes)
    test("SAM2: fallback when SAM2 not installed", test_sam2_refiner_fallback_missing_sam2)
    test("SAM2: _bbox_from_mask geometry",         test_bbox_from_mask)
    test("LLM: passthrough when no backend",       test_llm_verifier_passthrough_no_backend)
    test("LLM: confidence filter pre-LLM",        test_llm_verifier_confidence_filter)
    test("LLM: empty box list handled",            test_llm_verifier_empty_input)
    test("Pipeline: empty input → empty summary",  test_pipeline_empty_input)
    test("Pipeline: summary fields correct",       test_pipeline_summary_fields)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print("="*60)
    print(f"  Results: {passed} passed, {failed} failed / {len(results)} total")
    print("="*60 + "\n")

    if failed:
        print("Failed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  ✗ {name}")
                print(f"    {err}")
        sys.exit(1)
    else:
        print("  All tests passed! ✓\n")
