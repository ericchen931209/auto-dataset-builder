import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Enum, Text, JSON
)
from sqlalchemy.orm import relationship
from app.db.base import Base


class DatasetStatus(str, enum.Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    ANNOTATING = "annotating"
    CLEANING = "cleaning"
    EVALUATING = "evaluating"
    READY = "ready"
    FAILED = "failed"


class JobType(str, enum.Enum):
    COLLECT = "collect"
    EXTRACT_FRAMES = "extract_frames"
    ANNOTATE = "annotate"
    CLEAN = "clean"
    EVALUATE_DQS = "evaluate_dqs"
    ACTIVE_LEARNING = "active_learning"
    EXPORT = "export"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.PENDING)
    version = Column(String(20), default="0.1")

    # Parsed from NL input
    target_class = Column(String(100), nullable=True)
    task_type = Column(String(50), nullable=True)     # object_detection, segmentation, etc.
    region = Column(String(100), nullable=True)

    # Stats
    total_images = Column(Integer, default=0)
    annotated_images = Column(Integer, default=0)
    dqs_score = Column(Float, nullable=True)

    # Quality breakdown
    dqs_annotation_quality = Column(Float, nullable=True)
    dqs_diversity = Column(Float, nullable=True)
    dqs_lighting = Column(Float, nullable=True)
    dqs_pose = Column(Float, nullable=True)
    dqs_class_balance = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    images = relationship("Image", back_populates="dataset", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="dataset", cascade="all, delete-orphan")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    source_url = Column(String(1024), nullable=True)
    source_type = Column(String(50), nullable=True)   # youtube, google_images, camera

    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # Quality flags
    blur_score = Column(Float, nullable=True)
    brightness_mean = Column(Float, nullable=True)
    is_clean = Column(Boolean, default=True)

    # Embedding for diversity calculation (stored as JSON list)
    clip_embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    class_name = Column(String(100), nullable=False)
    class_id = Column(Integer, default=0)

    # YOLO format (normalized)
    bbox_x_center = Column(Float, nullable=False)
    bbox_y_center = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)

    # Confidence from detection model
    confidence = Column(Float, nullable=True)

    # Verification status
    yolo_verified = Column(Boolean, default=False)
    sam2_refined = Column(Boolean, default=False)
    llm_verified = Column(Boolean, default=False)
    human_verified = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="annotations")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    job_type = Column(Enum(JobType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)

    celery_task_id = Column(String(255), nullable=True)
    progress = Column(Float, default=0.0)       # 0.0 ~ 1.0
    progress_message = Column(String(512), nullable=True)

    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="jobs")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    version_tag = Column(String(20), nullable=False)     # v1.0, v1.1, ...
    snapshot_path = Column(String(512), nullable=True)   # zip path
    total_images = Column(Integer, default=0)
    dqs_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")
