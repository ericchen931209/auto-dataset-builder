"""
Active Learning Loop Orchestrator (V0.7).

Wires together:
  - DQS evaluation      (models/dqs)
  - Uncertainty sampling (uncertainty_sampler)
  - Re-annotation        (workers/annotator/three_stage_pipeline)
  - Convergence check   (convergence_checker)

The loop terminates when the dataset DQS score reaches the target threshold,
the improvement stalls, or the iteration cap is hit.
"""
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from workers.active_learning.uncertainty_sampler import (
    sample_uncertain_images,
    ImageUncertainty,
)
from workers.active_learning.convergence_checker import (
    ConvergenceChecker,
    ConvergenceResult,
    StopReason,
)

logger = logging.getLogger(__name__)


@dataclass
class ALConfig:
    # DQS convergence
    dqs_threshold: float = 0.75
    max_iterations: int = 10
    min_delta: float = 0.005
    stall_window: int = 2

    # Uncertainty sampling
    strategy: str = "min_conf"        # "min_conf" | "mean_conf" | "entropy"
    uncertainty_threshold: float = 0.5
    top_k: int | None = None          # None = re-annotate all uncertain images

    # Re-annotation (three-stage pipeline)
    yolo_model: str = "yolov11n.pt"
    confidence_threshold: float = 0.25
    target_classes: list[str] | None = None
    class_map: dict[str, int] | None = None
    sam2_checkpoint: str = "checkpoints/sam2_hiera_tiny.pt"
    sam2_config: str = "sam2_hiera_t.yaml"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llava"

    # Misc
    dqs_model_path: str | None = None  # pre-trained Neural DQS; None = heuristic


@dataclass
class ALLoopResult:
    converged: bool
    stop_reason: StopReason
    total_iterations: int
    final_dqs: float
    convergence_history: list[dict] = field(default_factory=list)
    re_annotated_counts: list[int] = field(default_factory=list)


# ─── DQS helper (decoupled for testability) ───────────────────────────────────

def _evaluate_dqs(images_dir: str, labels_dir: str, model_path: str | None) -> float:
    """Compute DQS for the current dataset state."""
    try:
        from models.dqs.feature_extractor import extract_features
        from models.dqs.neural_dqs import predict

        image_files = [
            str(p) for p in Path(images_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        label_files = [
            str(Path(labels_dir) / (Path(p).stem + ".txt"))
            for p in image_files
        ]

        features = extract_features(image_files, label_files)
        feature_vec = [
            features.annotation_quality,
            features.diversity,
            features.lighting_diversity,
            features.pose_diversity,
            features.class_balance,
        ]
        return predict(feature_vec, model_path=model_path)

    except Exception as e:
        logger.warning(f"DQS evaluation failed ({e}), returning 0.0")
        return 0.0


# ─── Re-annotation helper ─────────────────────────────────────────────────────

def _reannotate(
    uncertain: list[ImageUncertainty],
    labels_dir: str,
    config: ALConfig,
) -> int:
    """Re-run the three-stage pipeline on uncertain images, overwriting their labels."""
    from workers.annotator.three_stage_pipeline import run_three_stage_pipeline

    image_paths = [u.image_path for u in uncertain]
    if not image_paths:
        return 0

    summary = run_three_stage_pipeline(
        image_paths=image_paths,
        labels_dir=labels_dir,
        yolo_model=config.yolo_model,
        confidence_threshold=config.confidence_threshold,
        target_classes=config.target_classes,
        class_map=config.class_map,
        sam2_checkpoint=config.sam2_checkpoint,
        sam2_config=config.sam2_config,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    return summary.total_boxes


# ─── Public API ───────────────────────────────────────────────────────────────

def run_active_learning_loop(
    images_dir: str,
    labels_dir: str,
    config: ALConfig | None = None,
) -> ALLoopResult:
    """
    Run the full active learning loop until convergence or iteration limit.

    Each iteration:
      1. Evaluate DQS on the current dataset
      2. Check convergence (stop if met)
      3. Sample uncertain images
      4. Re-annotate them with the three-stage pipeline
      5. Repeat

    Args:
        images_dir: Path to directory containing dataset images.
        labels_dir: Path to directory containing YOLO label files (read + written).
        config: ALConfig; uses sensible defaults if None.

    Returns:
        ALLoopResult with final DQS, stop reason, and per-iteration history.
    """
    if config is None:
        config = ALConfig()

    checker = ConvergenceChecker(
        dqs_threshold=config.dqs_threshold,
        max_iterations=config.max_iterations,
        min_delta=config.min_delta,
        window=config.stall_window,
    )

    re_annotated_counts: list[int] = []
    convergence_result: ConvergenceResult | None = None

    logger.info(
        f"[AL] Starting loop — target DQS={config.dqs_threshold}, "
        f"max_iter={config.max_iterations}, strategy={config.strategy}"
    )

    while True:
        # ── Step 1: Evaluate DQS ─────────────────────────────────────────────
        dqs = _evaluate_dqs(images_dir, labels_dir, config.dqs_model_path)

        # ── Step 2: Sample uncertain images ──────────────────────────────────
        uncertain = sample_uncertain_images(
            images_dir=images_dir,
            labels_dir=labels_dir,
            strategy=config.strategy,
            top_k=config.top_k,
            threshold=config.uncertainty_threshold,
        )

        # ── Step 3: Check convergence ─────────────────────────────────────────
        convergence_result = checker.step(dqs, len(uncertain))
        if convergence_result.should_stop:
            break

        # ── Step 4: Re-annotate ───────────────────────────────────────────────
        n_boxes = _reannotate(uncertain, labels_dir, config)
        re_annotated_counts.append(len(uncertain))
        logger.info(
            f"[AL iter {checker.iteration}] "
            f"Re-annotated {len(uncertain)} images → {n_boxes} boxes"
        )

    summary = checker.summary()
    return ALLoopResult(
        converged=(convergence_result.reason == StopReason.DQS_THRESHOLD),
        stop_reason=convergence_result.reason,
        total_iterations=checker.iteration,
        final_dqs=summary["final_dqs"] or 0.0,
        convergence_history=summary["history"],
        re_annotated_counts=re_annotated_counts,
    )
