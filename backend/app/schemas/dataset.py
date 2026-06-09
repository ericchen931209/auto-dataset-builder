from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from app.db.models import DatasetStatus


class DatasetCreate(BaseModel):
    query: str = Field(..., description="Natural language dataset request")
    name: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {"query": "Build a Taiwan motorcycle detection dataset"}}}


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: DatasetStatus
    version: str
    target_class: Optional[str]
    task_type: Optional[str]
    region: Optional[str]
    total_images: int
    annotated_images: int
    dqs_score: Optional[float]
    dqs_annotation_quality: Optional[float]
    dqs_diversity: Optional[float]
    dqs_lighting: Optional[float]
    dqs_pose: Optional[float]
    dqs_class_balance: Optional[float]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetSummary(BaseModel):
    id: int
    name: str
    status: DatasetStatus
    total_images: int
    dqs_score: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: int
    dataset_id: int
    job_type: str
    status: str
    progress: float
    progress_message: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
