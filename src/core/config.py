"""Application configuration via Pydantic Settings."""
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
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
    app_version: str = "0.1.0"
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
    max_video_size_mb: int = 500
    allowed_video_formats: list[str] = Field(
        default_factory=lambda: ["mp4", "mkv", "webm", "avi", "mov"]
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/framemind.db"

    # ML
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: Literal["cpu", "cuda", "mps"] = "cpu"
    frame_extraction_fps: float = 2.0
    max_frames_per_video: int = 1000
    target_keyframes: int = 30

    # Shot detection
    shot_threshold: float = 0.3
    min_scene_length: int = 10  # minimum frames between scene changes

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
    worker_concurrency: int = 4
    job_timeout: int = 600  # 10 minutes

    @field_validator("storage_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path and ensure it exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
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
