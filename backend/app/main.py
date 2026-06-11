from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.db.base import Base, get_engine

settings = get_settings()

# Create tables on startup (use Alembic migrations in production)
Base.metadata.create_all(bind=get_engine())

app = FastAPI(
    title="Auto Dataset Builder",
    description="LLM-Assisted Automatic Dataset Construction Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# allow_origins="*" + allow_credentials=True is rejected by browsers (and a
# misconfiguration risk if this is ever changed to reflect the request Origin).
# The API doesn't use cookie-based auth, so credentials aren't needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0", "service": "auto-dataset-builder"}
