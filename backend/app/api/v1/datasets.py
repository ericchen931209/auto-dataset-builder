import os
import re

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Literal, List

from app.db.base import get_db
from app.db.models import Dataset, DatasetStatus, DatasetVersion
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetSummary

router = APIRouter(prefix="/datasets", tags=["datasets"])

# version_tag is interpolated directly into filesystem paths (exports/,
# snapshots/) — restrict to a safe charset to prevent path traversal
# (e.g. version_tag="../../etc").
_VERSION_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_version_tag(version_tag: str) -> str:
    if ".." in version_tag or not _VERSION_TAG_RE.match(version_tag):
        raise HTTPException(status_code=400, detail=f"Invalid version_tag: {version_tag!r}")
    return version_tag


@router.get("/", response_model=List[DatasetSummary])
def list_datasets(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).offset(skip).limit(limit).all()


@router.post("/", response_model=DatasetResponse, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    name = payload.name or f"Dataset: {payload.query[:50]}"

    dataset = Dataset(
        name=name,
        description=payload.query,
        status=DatasetStatus.PENDING,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # TODO (V0.2): parse NL query and dispatch collection job
    # from workers.collector.tasks import collect_youtube
    # collect_youtube.delay(dataset.id, keywords=[...])

    return dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()


@router.post("/{dataset_id}/evaluate-dqs", response_model=dict)
def evaluate_dqs(dataset_id: int, db: Session = Depends(get_db)):
    """
    Compute Neural DQS for an annotated dataset.
    Requires images and labels to be present in storage.
    """
    import os
    from models.dqs.feature_extractor import extract_features
    from models.dqs.neural_dqs import predict

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    storage_path = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")
    image_dir = os.path.join(storage_path, str(dataset_id), "frames")
    labels_dir = os.path.join(storage_path, str(dataset_id), "labels")

    if not os.path.isdir(image_dir) or not os.path.isdir(labels_dir):
        raise HTTPException(status_code=422, detail="Dataset images/labels not found on disk")

    features = extract_features(image_dir, labels_dir)
    score = predict(features.to_vector())

    # Persist to database
    dataset.dqs_score = score
    dataset.dqs_annotation_quality = features.annotation_quality
    dataset.dqs_diversity = features.clip_diversity
    dataset.dqs_lighting = features.lighting_diversity
    dataset.dqs_pose = features.pose_diversity
    dataset.dqs_class_balance = features.class_balance
    db.commit()

    return {
        "dataset_id": dataset_id,
        "dqs_score": score,
        "features": features.to_dict(),
    }


# ─── V0.9: Export ─────────────────────────────────────────────────────────────

@router.post("/{dataset_id}/export", response_model=dict)
def export_dataset(
    dataset_id: int,
    fmt: Literal["yolo", "coco"] = "yolo",
    version_tag: str = "v1.0",
    db: Session = Depends(get_db),
):
    """
    Export dataset as a zip archive in YOLO or COCO JSON format.
    Returns download URL.
    """
    from app.services.exporter import export_yolo, export_coco

    _validate_version_tag(version_tag)

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    storage_path = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")
    images_dir = os.path.join(storage_path, str(dataset_id), "frames")
    labels_dir = os.path.join(storage_path, str(dataset_id), "labels")
    output_dir = os.path.join(storage_path, str(dataset_id), "exports", version_tag, fmt)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        raise HTTPException(status_code=422, detail="Dataset images not found on disk")

    class_names = [dataset.target_class] if dataset.target_class else ["object"]

    if fmt == "yolo":
        manifest = export_yolo(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            dataset_name=dataset.name,
            class_names=class_names,
            version_tag=version_tag,
        )
    else:
        manifest = export_coco(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            dataset_name=dataset.name,
            class_names=class_names,
            version_tag=version_tag,
        )

    return {
        "dataset_id": dataset_id,
        "format": fmt,
        "version_tag": version_tag,
        "num_images": manifest.num_images,
        "num_annotations": manifest.num_annotations,
        "checksum": manifest.checksum,
        "download_url": f"/api/v1/datasets/{dataset_id}/download?version_tag={version_tag}&fmt={fmt}",
    }


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: int,
    version_tag: str = "v1.0",
    fmt: Literal["yolo", "coco"] = "yolo",
    db: Session = Depends(get_db),
):
    """Stream the exported zip file for download."""
    _validate_version_tag(version_tag)

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    storage_path = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")
    zip_path = os.path.join(
        storage_path, str(dataset_id), "exports", version_tag, fmt
    ) + ".zip"

    if not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="Export not found. Call /export first.")

    filename = f"{dataset.name}_{version_tag}_{fmt}.zip".replace(" ", "_")
    return FileResponse(zip_path, media_type="application/zip", filename=filename)


# ─── V0.9: Version Control ────────────────────────────────────────────────────

@router.post("/{dataset_id}/versions", response_model=dict, status_code=201)
def create_version(
    dataset_id: int,
    version_tag: str,
    notes: str = "",
    db: Session = Depends(get_db),
):
    """Create a versioned snapshot of the current dataset state."""
    from app.services.version_control import create_snapshot

    _validate_version_tag(version_tag)

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    existing = (
        db.query(DatasetVersion)
        .filter_by(dataset_id=dataset_id, version_tag=version_tag)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Version {version_tag!r} already exists")

    storage_path = os.getenv("DATASET_STORAGE_PATH", "/app/datasets")
    images_dir   = os.path.join(storage_path, str(dataset_id), "frames")
    labels_dir   = os.path.join(storage_path, str(dataset_id), "labels")
    snapshots_root = os.path.join(storage_path, "snapshots")

    if not os.path.isdir(images_dir):
        raise HTTPException(status_code=422, detail="Dataset images not found on disk")

    snap = create_snapshot(
        dataset_id=dataset_id,
        version_tag=version_tag,
        images_dir=images_dir,
        labels_dir=labels_dir,
        snapshots_root=snapshots_root,
        notes=notes,
    )

    dv = DatasetVersion(
        dataset_id=dataset_id,
        version_tag=version_tag,
        snapshot_path=snap["snapshot_path"],
        total_images=snap["total_images"],
        dqs_score=dataset.dqs_score,
        notes=notes,
    )
    db.add(dv)
    db.commit()
    db.refresh(dv)

    return {
        "id": dv.id,
        "dataset_id": dataset_id,
        "version_tag": version_tag,
        "total_images": snap["total_images"],
        "checksum": snap["checksum"],
        "snapshot_path": snap["snapshot_path"],
        "notes": notes,
    }


@router.get("/{dataset_id}/versions", response_model=list)
def list_versions(dataset_id: int, db: Session = Depends(get_db)):
    """List all snapshots for a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    versions = (
        db.query(DatasetVersion)
        .filter_by(dataset_id=dataset_id)
        .order_by(DatasetVersion.created_at)
        .all()
    )
    return [
        {
            "id": v.id,
            "version_tag": v.version_tag,
            "total_images": v.total_images,
            "dqs_score": v.dqs_score,
            "notes": v.notes,
            "created_at": v.created_at.isoformat(),
        }
        for v in versions
    ]


@router.get("/{dataset_id}/versions/diff", response_model=dict)
def diff_versions(
    dataset_id: int,
    from_tag: str,
    to_tag: str,
    db: Session = Depends(get_db),
):
    """Show what changed between two snapshots."""
    from app.services.version_control import diff_versions as _diff

    def _get_version(tag: str) -> DatasetVersion:
        v = (
            db.query(DatasetVersion)
            .filter_by(dataset_id=dataset_id, version_tag=tag)
            .first()
        )
        if not v:
            raise HTTPException(status_code=404, detail=f"Version {tag!r} not found")
        return v

    va = _get_version(from_tag)
    vb = _get_version(to_tag)

    diff = _diff(va.snapshot_path, vb.snapshot_path, from_tag, to_tag)
    return {
        "from_tag": diff.from_tag,
        "to_tag": diff.to_tag,
        "added": diff.added,
        "removed": diff.removed,
        "unchanged": diff.unchanged,
        "total_from": diff.total_from,
        "total_to": diff.total_to,
    }
