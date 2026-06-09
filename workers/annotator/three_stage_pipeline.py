"""
Three-Stage Annotation Pipeline (V0.5)

  Stage 1 — YOLOv11 proposal    (workers/annotator/yolo_annotator.py)
  Stage 2 — SAM2 refinement     (workers/annotator/sam2_refiner.py)
  Stage 3 — Vision LLM verify   (workers/annotator/llm_verifier.py)

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
    yolo_model: str = "yolov11n.pt",
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
    """
    if not image_paths:
        return PipelineSummary()

    # ── Stage 1: YOLO proposals ───────────────────────────────────────────────
    logger.info(f"[Stage 1] YOLO inference on {len(image_paths)} images")
    yolo_results: list[AnnotationResult] = run_yolo_batch(
        image_paths=image_paths,
        model_path=yolo_model,
        confidence_threshold=confidence_threshold,
        target_classes=target_classes,
    )
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
