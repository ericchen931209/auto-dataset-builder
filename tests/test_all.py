"""
ADB Test Suite — runs without Docker or GPU.
Tests core logic modules: deduplicator, cleaner, DQS features, Neural DQS.
"""
import sys, os, tempfile, math
from pathlib import Path
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
        for attr in ["annotation_quality", "sharpness", "clip_diversity",
                     "lighting_diversity", "pose_diversity", "class_balance"]:
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
    f = DQSFeatures(0.8, 0.6, 0.7, 0.5, 1.0, 0.9)
    v = f.to_vector()
    assert len(v) == 6
    assert v == [0.8, 0.6, 0.7, 0.5, 1.0, 0.9]
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

def test_version_tag_validation_accepts_safe_tags():
    from app.api.v1.datasets import _validate_version_tag
    assert _validate_version_tag("v1.0") == "v1.0"
    assert _validate_version_tag("v1.2-beta_3") == "v1.2-beta_3"

def test_version_tag_validation_rejects_path_traversal():
    from app.api.v1.datasets import _validate_version_tag
    from fastapi import HTTPException
    for bad in ["../../etc", "v1/../..", "..", ".hidden", "", "a/b"]:
        try:
            _validate_version_tag(bad)
            assert False, f"Should have rejected {bad!r}"
        except HTTPException as e:
            assert e.status_code == 400

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
    """verify_with_llm passes all boxes through when no LLM backend is available."""
    from unittest.mock import patch
    from workers.annotator.yolo_annotator import BoundingBox
    from workers.annotator.sam2_refiner import RefinedAnnotation
    from workers.annotator import llm_verifier
    box = BoundingBox(0, "motorcycle", 0.5, 0.5, 0.3, 0.2, 0.8)
    refined = RefinedAnnotation(image_path="/nonexistent.jpg", boxes=[box], fallback=True)
    # Force no backend (Qwen-VL/Ollama/CLIP all unavailable) → should fall back to passthrough
    with patch.object(llm_verifier, "_load_backend", return_value=None):
        results = llm_verifier.verify_with_llm([refined], ollama_url="http://localhost:1", ollama_model="llava")
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

def test_pipeline_yolo_world_requires_classes():
    """detector_backend='yolo_world' without open_vocab_classes/target_classes raises ValueError."""
    from workers.annotator.three_stage_pipeline import run_three_stage_pipeline
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            run_three_stage_pipeline(["/nonexistent.jpg"], labels_dir=d, detector_backend="yolo_world")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "open_vocab_classes" in str(e)

def test_pipeline_auto_uses_target_classes_for_open_vocab():
    """detector_backend='auto' with a non-COCO class falls back to YOLO-World using target_classes."""
    import sys, tempfile
    from unittest.mock import patch
    from workers.annotator import three_stage_pipeline as pipeline
    from workers.annotator.yolo_annotator import AnnotationResult

    captured = {}

    def fake_run_yolo_world_batch(image_paths, class_names, **kwargs):
        captured["class_names"] = class_names
        return [AnnotationResult(image_path=p, boxes=[], success=True) for p in image_paths]

    with tempfile.TemporaryDirectory() as d:
        with patch.object(pipeline, "run_yolo_world_batch", side_effect=fake_run_yolo_world_batch):
            summary = pipeline.run_three_stage_pipeline(
                ["/nonexistent.jpg"], labels_dir=d, target_classes=["scooter helmet"]
            )

    assert captured["class_names"] == ["scooter helmet"]
    assert summary.total_images == 1

def test_pipeline_unknown_backend():
    """An unrecognized detector_backend raises ValueError."""
    from workers.annotator.three_stage_pipeline import run_three_stage_pipeline
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            run_three_stage_pipeline(["/nonexistent.jpg"], labels_dir=d, detector_backend="bogus")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "bogus" in str(e)


# ─── Test: Automatic Backend Selection (coco_classes.select_detector_backend) ─

def test_select_backend_empty_classes_uses_yolo11():
    """No target_classes → yolo11 (existing well-tested COCO-80 path)."""
    from workers.annotator.coco_classes import select_detector_backend
    assert select_detector_backend(None) == "yolo11"
    assert select_detector_backend([]) == "yolo11"

def test_select_backend_coco_classes_uses_yolo11():
    """All requested classes covered by COCO-80 → yolo11."""
    from workers.annotator.coco_classes import select_detector_backend
    assert select_detector_backend(["car", "person"]) == "yolo11"
    assert select_detector_backend(["Motorcycle"]) == "yolo11"  # case-insensitive

def test_select_backend_non_coco_classes_uses_yolo_world():
    """A class outside COCO-80 → yolo_world (open-vocabulary)."""
    from workers.annotator.coco_classes import select_detector_backend
    assert select_detector_backend(["helmet"]) == "yolo_world"

def test_select_backend_mixed_classes_uses_yolo_world():
    """Mix of COCO-80 and non-COCO classes → yolo_world (covers all requested classes)."""
    from workers.annotator.coco_classes import select_detector_backend
    assert select_detector_backend(["motorcycle", "helmet"]) == "yolo_world"


# ─── Test: Open-Vocabulary Detector (YOLO-World, no GPU/network needed) ──────

def test_open_vocab_empty_classes():
    """run_yolo_world_batch with no class_names returns failure results."""
    from workers.annotator.open_vocab_detector import run_yolo_world_batch
    results = run_yolo_world_batch(["/nonexistent.jpg"], class_names=[])
    assert len(results) == 1
    assert results[0].success is False
    assert "class_names" in results[0].error

def test_open_vocab_missing_ultralytics():
    """run_yolo_world_batch reports failure when ultralytics is not importable."""
    from unittest.mock import patch
    import builtins
    from workers.annotator import open_vocab_detector

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        results = open_vocab_detector.run_yolo_world_batch(
            ["/nonexistent.jpg"], class_names=["scooter"]
        )
    assert len(results) == 1
    assert results[0].success is False
    assert "ultralytics" in results[0].error

def test_open_vocab_batch_with_mocked_model():
    """run_yolo_world_batch maps detections to BoundingBox using the open-vocab class list."""
    import sys
    from unittest.mock import MagicMock, patch
    from workers.annotator import open_vocab_detector

    class_names = ["scooter", "helmet"]

    fake_box = MagicMock()
    fake_box.cls = [0]
    fake_box.conf = [0.77]
    fake_box.xywhn = [MagicMock(tolist=lambda: [0.5, 0.5, 0.2, 0.3])]

    fake_pred = MagicMock()
    fake_pred.boxes = [fake_box]

    fake_model = MagicMock()
    fake_model.return_value = [fake_pred]

    fake_yolo_cls = MagicMock(return_value=fake_model)
    fake_ultralytics = MagicMock(YOLO=fake_yolo_cls)

    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        results = open_vocab_detector.run_yolo_world_batch(
            ["/nonexistent.jpg"], class_names=class_names, model_path="yolov8s-worldv2.pt"
        )

    fake_model.set_classes.assert_called_once_with(class_names)
    assert len(results) == 1
    assert results[0].success is True
    assert len(results[0].boxes) == 1
    box = results[0].boxes[0]
    assert box.class_name == "scooter"
    assert box.confidence == 0.77


# ─── Test: Uncertainty Sampler ───────────────────────────────────────────────

def test_uncertainty_sampler_no_labels():
    """Images with no label file are treated as fully uncertain (score=0.0)."""
    from workers.active_learning.uncertainty_sampler import sample_uncertain_images
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        make_image(f"{imgd}/a.jpg")
        make_image(f"{imgd}/b.jpg")
        # No label files created
        result = sample_uncertain_images(imgd, lbld, strategy="min_conf", threshold=0.5)
        assert len(result) == 2
        assert all(r.score == 0.0 for r in result)
        assert all(r.num_boxes == 0 for r in result)

def test_uncertainty_sampler_high_conf_excluded():
    """Images whose min-conf is above threshold are not returned."""
    from workers.active_learning.uncertainty_sampler import sample_uncertain_images
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        make_image(f"{imgd}/a.jpg")
        # Label file with high-confidence box (6th column = conf)
        with open(f"{lbld}/a.txt", "w") as f:
            f.write("0 0.5 0.5 0.3 0.2 0.95\n")
        result = sample_uncertain_images(imgd, lbld, strategy="min_conf", threshold=0.5)
        assert len(result) == 0   # conf=0.95 > threshold=0.5 → not uncertain

def test_uncertainty_sampler_low_conf_included():
    """Images whose min-conf is below threshold are returned."""
    from workers.active_learning.uncertainty_sampler import sample_uncertain_images
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        make_image(f"{imgd}/a.jpg")
        with open(f"{lbld}/a.txt", "w") as f:
            f.write("0 0.5 0.5 0.3 0.2 0.30\n")   # conf=0.30 < 0.5
        result = sample_uncertain_images(imgd, lbld, strategy="min_conf", threshold=0.5)
        assert len(result) == 1
        assert result[0].min_confidence == 0.30

def test_uncertainty_sampler_top_k():
    """top_k limits number of returned samples."""
    from workers.active_learning.uncertainty_sampler import sample_uncertain_images
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        for i in range(5):
            make_image(f"{imgd}/{i}.jpg")
            # All uncertain (no labels)
        result = sample_uncertain_images(imgd, lbld, strategy="min_conf",
                                         threshold=0.5, top_k=3)
        assert len(result) == 3

def test_uncertainty_sampler_entropy():
    """Entropy strategy flags images with high class-distribution uncertainty."""
    from workers.active_learning.uncertainty_sampler import sample_uncertain_images
    with tempfile.TemporaryDirectory() as imgd, tempfile.TemporaryDirectory() as lbld:
        make_image(f"{imgd}/a.jpg")
        # Two equally-distributed classes → max entropy
        with open(f"{lbld}/a.txt", "w") as f:
            f.write("0 0.5 0.5 0.3 0.2\n")
            f.write("1 0.2 0.2 0.1 0.1\n")
        result = sample_uncertain_images(imgd, lbld, strategy="entropy", threshold=0.0)
        assert len(result) == 1
        assert result[0].score > 0   # entropy > 0 for 2 classes


# ─── Test: Convergence Checker ────────────────────────────────────────────────

def test_convergence_stops_at_threshold():
    """ConvergenceChecker stops when DQS >= dqs_threshold."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.75, max_iterations=10)
    result = checker.step(dqs_score=0.80, uncertain_count=5)
    assert result.should_stop is True
    assert result.reason == StopReason.DQS_THRESHOLD

def test_convergence_stops_at_max_iterations():
    """ConvergenceChecker stops after max_iterations regardless of DQS."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.99, max_iterations=3)
    for _ in range(2):
        r = checker.step(dqs_score=0.50, uncertain_count=5)
        assert not r.should_stop
    r = checker.step(dqs_score=0.50, uncertain_count=5)
    assert r.should_stop
    assert r.reason == StopReason.MAX_ITERATIONS

def test_convergence_stops_no_uncertain():
    """ConvergenceChecker stops when no uncertain images remain."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.99, max_iterations=10)
    result = checker.step(dqs_score=0.50, uncertain_count=0)
    assert result.should_stop is True
    assert result.reason == StopReason.NO_UNCERTAIN

def test_convergence_stops_stalled():
    """ConvergenceChecker stops when DQS improvement stalls."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.99, max_iterations=20, min_delta=0.01, window=2)
    # Feed 3 nearly identical DQS values
    checker.step(0.60, 5)
    checker.step(0.601, 5)
    r = checker.step(0.6015, 5)  # delta over window ≈ 0.0015 < 0.01
    assert r.should_stop is True
    assert r.reason == StopReason.DQS_STALLED

def test_convergence_continues_when_improving():
    """ConvergenceChecker continues when DQS is improving and below threshold."""
    from workers.active_learning.convergence_checker import ConvergenceChecker, StopReason
    checker = ConvergenceChecker(dqs_threshold=0.99, max_iterations=20, min_delta=0.01, window=2)
    r1 = checker.step(0.50, 10)
    r2 = checker.step(0.60, 8)   # +0.10 > 0.01 → not stalled
    assert not r1.should_stop
    assert not r2.should_stop

def test_convergence_summary_fields():
    """ConvergenceChecker.summary() returns all expected keys."""
    from workers.active_learning.convergence_checker import ConvergenceChecker
    checker = ConvergenceChecker()
    checker.step(0.5, 3)
    checker.step(0.6, 2)
    s = checker.summary()
    assert "total_iterations" in s
    assert "final_dqs" in s
    assert "history" in s
    assert len(s["history"]) == 2


# ─── Test: Exporter ──────────────────────────────────────────────────────────

def test_export_yolo_structure():
    """export_yolo creates images/labels split dirs and dataset.yaml."""
    from backend.app.services.exporter import export_yolo
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as outd:
        for i in range(5):
            make_image(f"{imgd}/{i}.jpg", noise=True)
            make_label(f"{lbld}/{i}.txt")
        manifest = export_yolo(imgd, lbld, outd, "TestDS", ["motorcycle"], "v1.0",
                                train=0.6, val=0.2)
        out = Path(outd)
        assert (out / "dataset.yaml").exists()
        assert (out / "images" / "train").is_dir()
        assert (out / "images" / "val").is_dir()
        assert (out / "images" / "test").is_dir()
        assert manifest.num_images == 5
        assert manifest.export_format == "yolo"
        assert len(manifest.checksum) == 64   # sha256 hex

def test_export_yolo_split_sum():
    """train+val+test image counts sum to total."""
    from backend.app.services.exporter import export_yolo
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as outd:
        for i in range(10):
            make_image(f"{imgd}/{i}.jpg", noise=True)
        manifest = export_yolo(imgd, lbld, outd, "TestDS", ["car"], "v1.0",
                                train=0.7, val=0.2)
        assert sum(manifest.split.values()) == 10

def test_export_coco_json():
    """export_coco creates annotations.json with correct COCO structure."""
    from backend.app.services.exporter import export_coco
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as outd:
        make_image(f"{imgd}/a.jpg", noise=True)
        make_label(f"{lbld}/a.txt", cx=0.5, cy=0.5, w=0.3, h=0.2, cls=0)
        manifest = export_coco(imgd, lbld, outd, "TestDS", ["motorcycle"], "v1.0")
        import json
        ann = json.loads((Path(outd) / "annotations.json").read_text())
        assert "images" in ann and "annotations" in ann and "categories" in ann
        assert ann["categories"][0]["name"] == "motorcycle"
        assert manifest.num_annotations == 1

def test_export_coco_bbox_format():
    """COCO annotations use [x, y, w, h] pixel format (not normalized)."""
    from backend.app.services.exporter import export_coco
    import json
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as outd:
        make_image(f"{imgd}/a.jpg", w=100, h=100)
        # YOLO: cx=0.5 cy=0.5 w=0.5 h=0.5 → COCO: x=25 y=25 w=50 h=50
        with open(f"{lbld}/a.txt", "w") as f:
            f.write("0 0.5 0.5 0.5 0.5\n")
        export_coco(imgd, lbld, outd, "TestDS", ["obj"], "v1.0")
        ann = json.loads((Path(outd) / "annotations.json").read_text())
        bbox = ann["annotations"][0]["bbox"]
        assert abs(bbox[0] - 25) < 1   # x
        assert abs(bbox[1] - 25) < 1   # y
        assert abs(bbox[2] - 50) < 1   # w
        assert abs(bbox[3] - 50) < 1   # h


# ─── Test: Version Control ────────────────────────────────────────────────────

def test_create_snapshot_produces_zip():
    """create_snapshot writes a zip and returns checksum + image count."""
    from backend.app.services.version_control import create_snapshot
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as snpd:
        for i in range(3):
            make_image(f"{imgd}/{i}.jpg", noise=True)
            make_label(f"{lbld}/{i}.txt")
        result = create_snapshot(1, "v1.0", imgd, lbld, snpd)
        assert result["total_images"] == 3
        assert len(result["checksum"]) == 64
        assert result["snapshot_path"].endswith(".zip")
        assert Path(result["snapshot_path"]).exists()

def test_diff_versions_added_removed():
    """diff_versions detects added and removed images between snapshots."""
    import shutil as _shutil
    from backend.app.services.version_control import create_snapshot, diff_versions
    with tempfile.TemporaryDirectory() as imgd1, \
         tempfile.TemporaryDirectory() as imgd2, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as snpd:
        make_image(f"{imgd1}/a.jpg", noise=True)
        make_image(f"{imgd1}/b.jpg", noise=True)
        _shutil.copy(f"{imgd1}/a.jpg", f"{imgd2}/a.jpg")  # same bytes → unchanged
        make_image(f"{imgd2}/c.jpg", noise=True)           # c added, b removed
        snap1 = create_snapshot(1, "v1.0", imgd1, lbld, snpd)
        snap2 = create_snapshot(1, "v1.1", imgd2, lbld, snpd)
        diff = diff_versions(snap1["snapshot_path"], snap2["snapshot_path"],
                             "v1.0", "v1.1")
        assert "c.jpg" in diff.added
        assert "b.jpg" in diff.removed
        assert diff.unchanged >= 1   # a.jpg present in both

def test_diff_versions_identical():
    """diff_versions reports zero added/removed for identical snapshots."""
    from backend.app.services.version_control import create_snapshot, diff_versions
    with tempfile.TemporaryDirectory() as imgd, \
         tempfile.TemporaryDirectory() as lbld, \
         tempfile.TemporaryDirectory() as snpd:
        make_image(f"{imgd}/a.jpg", noise=True)
        snap = create_snapshot(1, "v1.0", imgd, lbld, snpd)
        snap2 = create_snapshot(1, "v1.1", imgd, lbld, snpd)
        diff = diff_versions(snap["snapshot_path"], snap2["snapshot_path"], "v1.0", "v1.1")
        assert diff.added == []
        assert diff.removed == []
        assert diff.unchanged == 1


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
    test("Datasets API: version_tag accepts safe", test_version_tag_validation_accepts_safe_tags)
    test("Datasets API: version_tag rejects ../",  test_version_tag_validation_rejects_path_traversal)
    test("SAM2: fallback on empty boxes",          test_sam2_refiner_fallback_no_boxes)
    test("SAM2: fallback when SAM2 not installed", test_sam2_refiner_fallback_missing_sam2)
    test("SAM2: _bbox_from_mask geometry",         test_bbox_from_mask)
    test("LLM: passthrough when no backend",       test_llm_verifier_passthrough_no_backend)
    test("LLM: confidence filter pre-LLM",        test_llm_verifier_confidence_filter)
    test("LLM: empty box list handled",            test_llm_verifier_empty_input)
    test("Pipeline: empty input → empty summary",  test_pipeline_empty_input)
    test("Pipeline: summary fields correct",       test_pipeline_summary_fields)
    test("Pipeline: yolo_world requires classes",   test_pipeline_yolo_world_requires_classes)
    test("Pipeline: auto uses target_classes",      test_pipeline_auto_uses_target_classes_for_open_vocab)
    test("Pipeline: unknown backend raises",        test_pipeline_unknown_backend)
    test("Backend select: empty → yolo11",          test_select_backend_empty_classes_uses_yolo11)
    test("Backend select: COCO classes → yolo11",   test_select_backend_coco_classes_uses_yolo11)
    test("Backend select: non-COCO → yolo_world",   test_select_backend_non_coco_classes_uses_yolo_world)
    test("Backend select: mixed → yolo_world",      test_select_backend_mixed_classes_uses_yolo_world)
    test("OpenVocab: empty classes → failure",      test_open_vocab_empty_classes)
    test("OpenVocab: missing ultralytics handled",  test_open_vocab_missing_ultralytics)
    test("OpenVocab: mocked model → BoundingBox",   test_open_vocab_batch_with_mocked_model)
    test("AL Sampler: no labels → uncertain",       test_uncertainty_sampler_no_labels)
    test("AL Sampler: high conf excluded",          test_uncertainty_sampler_high_conf_excluded)
    test("AL Sampler: low conf included",           test_uncertainty_sampler_low_conf_included)
    test("AL Sampler: top_k limits results",        test_uncertainty_sampler_top_k)
    test("AL Sampler: entropy strategy",            test_uncertainty_sampler_entropy)
    test("AL Convergence: stops at DQS threshold", test_convergence_stops_at_threshold)
    test("AL Convergence: stops at max iterations",test_convergence_stops_at_max_iterations)
    test("AL Convergence: stops no uncertain",     test_convergence_stops_no_uncertain)
    test("AL Convergence: stops when stalled",     test_convergence_stops_stalled)
    test("AL Convergence: continues improving",    test_convergence_continues_when_improving)
    test("AL Convergence: summary fields",         test_convergence_summary_fields)
    test("Export YOLO: directory structure",       test_export_yolo_structure)
    test("Export YOLO: split counts sum",          test_export_yolo_split_sum)
    test("Export COCO: json structure",            test_export_coco_json)
    test("Export COCO: bbox pixel format",         test_export_coco_bbox_format)
    test("VersionCtrl: snapshot produces zip",     test_create_snapshot_produces_zip)
    test("VersionCtrl: diff added/removed",        test_diff_versions_added_removed)
    test("VersionCtrl: diff identical",            test_diff_versions_identical)

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
