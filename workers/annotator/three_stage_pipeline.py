"""
Three-Stage Annotation Pipeline (V0.5)

  Stage 1 — Proposal generation (workers/annotator/yolo_annotator.py,
                                  workers/annotator/open_vocab_detector.py)
  Stage 2 — SAM2 refinement     (workers/annotator/sam2_refiner.py)
  Stage 3 — Vision LLM verify   (workers/annotator/llm_verifier.py)

Stage 1 supports two interchangeable detector backends:
  - "yolo11"     COCO-80 pretrained YOLOv11 (run_yolo_batch)
  - "yolo_world" open-vocabulary YOLO-World (run_yolo_world_batch),
                 accepts arbitrary text-prompt class names via
                 open_vocab_classes.

By default (detector_backend="auto"), the backend is chosen automatically
from target_classes: if every requested class is in COCO-80 the well-tested
"yolo11" path is used unchanged; otherwise it falls back to "yolo_world"
using target_classes as the open-vocabulary prompt list. Callers therefore
get arbitrary-class support "for free" — just pass target_classes as before.

Each stage degrades gracefully: if a heavy model is unavailable the pipeline
continues using the previous stage's output unchanged.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from workers.annotator.yolo_annotator import (
    BoundingBox,
    AnnotationResult,
    run_yolo_batch,
    save_yolo_labels,
)
from workers.annotator.open_vocab_detector import run_yolo_world_batch
from workers.annotator.coco_classes import select_detector_backend
from workers.annotator.sam2_refiner import refine_with_sam2, RefinedAnnotation
from workers.annotator.llm_verifier import verify_with_llm, VerifiedAnnotation

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)
    # Per-stage metadata
    yolo_count: int = 0
    sam2_refined: int = 0
    llm_rejected: int = 0
    sam2_fallback: bool = False
    llm_backend: str = "passthrough"


@dataclass
class PipelineSummary:
    total_images: int = 0
    total_boxes: int = 0
    sam2_refined: int = 0
    llm_rejected: int = 0
    sam2_fallback_images: int = 0
    llm_backend: str = "passthrough"
    results: list[PipelineResult] = field(default_factory=list)


def run_three_stage_pipeline(
    image_paths: list[str],
    labels_dir: str,
    # Stage 1 options
    detector_backend: str = "auto",  # "auto" | "yolo11" | "yolo_world"
    yolo_model: str = "yolov11n.pt",
    yolo_world_model: str = "yolov8s-worldv2.pt",
    open_vocab_classes: list[str] | None = None,
    confidence_threshold: float = 0.25,
    target_classes: list[str] | None = None,
    class_map: dict[str, int] | None = None,
    # Stage 2 options
    sam2_checkpoint: str = "checkpoints/sam2_hiera_tiny.pt",
    sam2_config: str = "sam2_hiera_t.yaml",
    # Stage 3 options
    llm_confidence_threshold: float = 0.5,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llava",
) -> PipelineSummary:
    """
    Run the full three-stage annotation pipeline on a list of images.

    Writes YOLO-format .txt label files to labels_dir.
    Returns a PipelineSummary with per-image and aggregate statistics.

    detector_backend selects Stage 1:
      - "auto"       (default) pick "yolo11" if target_classes is empty or
                     fully covered by COCO-80, else "yolo_world".
      - "yolo11"     COCO-80 pretrained YOLOv11, filtered by target_classes.
      - "yolo_world" open-vocabulary YOLO-World, using open_vocab_classes
                     (or target_classes if not given) as the text-prompt
                     vocabulary.
    """
    if not image_paths:
        return PipelineSummary()

    if detector_backend == "auto":
        detector_backend = select_detector_backend(target_classes)
        logger.info(f"[Stage 1] detector_backend='auto' -> {detector_backend!r} "
                    f"(target_classes={target_classes})")

    # ── Stage 1: detection proposals ──────────────────────────────────────────
    if detector_backend == "yolo_world":
        open_vocab_classes = open_vocab_classes or target_classes
        if not open_vocab_classes:
            raise ValueError("open_vocab_classes (or target_classes) is required when detector_backend='yolo_world'")
        logger.info(f"[Stage 1] YOLO-World (open-vocab) inference on {len(image_paths)} images, "
                    f"classes={open_vocab_classes}")
        yolo_results: list[AnnotationResult] = run_yolo_world_batch(
            image_paths=image_paths,
            class_names=open_vocab_classes,
            model_path=yolo_world_model,
            confidence_threshold=confidence_threshold,
        )
    elif detector_backend == "yolo11":
        logger.info(f"[Stage 1] YOLOv11 (COCO-80) inference on {len(image_paths)} images")
        yolo_results = run_yolo_batch(
            image_paths=image_paths,
            model_path=yolo_model,
            confidence_threshold=confidence_threshold,
            target_classes=target_classes,
        )
    else:
        raise ValueError(f"Unknown detector_backend: {detector_backend!r} (expected 'auto', 'yolo11', or 'yolo_world')")

    logger.info(f"[Stage 1] Done — {sum(len(r.boxes) for r in yolo_results)} boxes total")

    # ── Stage 2: SAM2 refinement ──────────────────────────────────────────────
    logger.info("[Stage 2] SAM2 bbox refinement")
    sam2_results: list[RefinedAnnotation] = refine_with_sam2(
        annotation_results=yolo_results,
        checkpoint=sam2_checkpoint,
        config=sam2_config,
    )
    sam2_total_refined = sum(r.refined_count for r in sam2_results)
    sam2_fallback_count = sum(1 for r in sam2_results if r.fallback)
    logger.info(f"[Stage 2] Refined {sam2_total_refined} boxes "
                f"({sam2_fallback_count} images used YOLO fallback)")

    # ── Stage 3: Vision LLM verification ─────────────────────────────────────
    logger.info("[Stage 3] Vision LLM label verification")
    verified_results: list[VerifiedAnnotation] = verify_with_llm(
        refined_results=sam2_results,
        confidence_threshold=llm_confidence_threshold,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )
    llm_total_rejected = sum(r.rejected_count for r in verified_results)
    llm_backend = verified_results[0].backend if verified_results else "passthrough"
    logger.info(f"[Stage 3] Backend={llm_backend}, rejected {llm_total_rejected} boxes")

    # ── Write label files ─────────────────────────────────────────────────────
    final_annotations = [
        AnnotationResult(image_path=v.image_path, boxes=v.boxes)
        for v in verified_results
    ]
    save_yolo_labels(final_annotations, labels_dir=labels_dir, class_map=class_map)

    # ── Build summary ─────────────────────────────────────────────────────────
    pipeline_results: list[PipelineResult] = []
    for yolo, sam2, verified in zip(yolo_results, sam2_results, verified_results):
        pipeline_results.append(PipelineResult(
            image_path=yolo.image_path,
            boxes=verified.boxes,
            yolo_count=len(yolo.boxes),
            sam2_refined=sam2.refined_count,
            llm_rejected=verified.rejected_count,
            sam2_fallback=sam2.fallback,
            llm_backend=verified.backend,
        ))

    summary = PipelineSummary(
        total_images=len(image_paths),
        total_boxes=sum(len(r.boxes) for r in pipeline_results),
        sam2_refined=sam2_total_refined,
        llm_rejected=llm_total_rejected,
        sam2_fallback_images=sam2_fallback_count,
        llm_backend=llm_backend,
        results=pipeline_results,
    )

    logger.info(
        f"[Pipeline] Complete — {summary.total_images} images, "
        f"{summary.total_boxes} final boxes "
        f"(SAM2 refined={summary.sam2_refined}, LLM rejected={summary.llm_rejected})"
    )
    return summary
