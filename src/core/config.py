"""Application configuration via Pydantic Settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "FrameMind"
    app_version: str = "0.2.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")  # type: ignore[assignment]

    # Storage
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: Path = Field(default=Path("./data"))
    max_video_size_mb: int = 20480
    allowed_video_formats: list[str] = Field(
        default_factory=lambda: ["mp4", "mkv", "webm", "avi", "mov"]
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/framemind.db"

    # ML - Spatial (CLIP)
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: Literal["cpu", "cuda", "mps"] = "cpu"
    frame_extraction_fps: float = Field(default=2.0, gt=0, le=60)
    max_frames_per_video: int = 1000
    target_keyframes: int = 30
    spatial_batch_size: int = Field(default=32, ge=1, le=128)

    # ML - Temporal (X-CLIP)
    xclip_model: str = "microsoft/xclip-base-patch32"
    temporal_window_frames: int = 8  # Checked against the loaded checkpoint
    temporal_fps: float = Field(default=8.0, gt=0, le=60)
    temporal_stride: float = Field(default=0.5, ge=0, lt=1)
    temporal_batch_size: int = Field(default=8, ge=1, le=64)
    use_temporal: bool = True  # Enable temporal stream

    # GPU Parallelization
    multi_gpu: bool = True  # Use all available GPUs
    prefetch_batches: int = 2  # Batches to prefetch
    fusion_alpha: float = 0.5  # Spatial vs temporal weight (0.5 = equal)

    # Shot detection
    shot_threshold: float = 0.3
    min_scene_length: int = 10  # minimum frames between scene changes

    # FAISS
    use_faiss: bool = True  # Use FAISS for vector search

    # VLM
    vlm_provider: Literal["openai", "anthropic"] = "openai"
    vlm_model: str = "gpt-4o"
    vlm_api_key: str = ""
    vlm_max_retries: int = 3
    vlm_timeout: float = 60.0

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Worker
    worker_concurrency: int = 1
    job_timeout: int = 600  # 10 minutes

    chunk_seconds: int = Field(default=60, ge=1, le=600)
    decode_size: int = Field(default=224, ge=32, le=1024)
    index_cache_mb: int = Field(default=256, ge=1)
    candidate_count: int = Field(default=50, ge=1, le=1000)
    max_intervals: int = Field(default=5, ge=1, le=20)
    context_seconds: float = Field(default=2.0, ge=0, le=30)
    model_revision: str = "main"
    autogaze_enabled: bool = False
    autogaze_url: str = ""
    autogaze_token: str = ""
    autogaze_timeout: float = 300.0
    video_max_frames: int = Field(default=128, ge=16, le=1024)
    video_max_tiles: int = Field(default=12, ge=1, le=48)

    @field_validator("storage_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path and ensure it exists."""
        path = Path(v)
        return path

    @property
    def redis_url_str(self) -> str:
        """Get Redis URL as string."""
        return str(self.redis_url)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
