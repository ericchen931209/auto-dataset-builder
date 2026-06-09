from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    secret_key: str = "change_this_in_production"
    api_v1_prefix: str = "/api/v1"

    # Database
    postgres_user: str = "adb_user"
    postgres_password: str = "adb_password"
    postgres_db: str = "adb_db"
    postgres_host: str = "db"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Storage
    dataset_storage_path: str = "/app/datasets"

    # Model paths
    yolo_model: str = "yolov11n.pt"
    sam2_model: str = "sam2_hiera_small.pt"
    llm_model: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Image search (optional)
    google_search_api_key: str = ""
    google_search_cx: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
