from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.base import get_db
from app.db.models import Job
from app.schemas.dataset import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=List[JobResponse])
def list_jobs(dataset_id: int | None = None, skip: int = 0, limit: int = 50,
              db: Session = Depends(get_db)):
    q = db.query(Job)
    if dataset_id:
        q = q.filter(Job.dataset_id == dataset_id)
    return q.offset(skip).limit(limit).all()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
