from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.db.base import get_db
from app.db.models import Dataset, DatasetStatus
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetSummary

router = APIRouter(prefix="/datasets", tags=["datasets"])


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
    dataset.dqs_diversity = features.diversity
    dataset.dqs_lighting = features.lighting_diversity
    dataset.dqs_pose = features.pose_diversity
    dataset.dqs_class_balance = features.class_balance
    db.commit()

    return {
        "dataset_id": dataset_id,
        "dqs_score": score,
        "features": features.to_dict(),
    }
