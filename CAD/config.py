from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://mts_user:mts_password@127.0.0.1:5433/mts_hackathon"
    CONFIDENCE_THRESHOLD: float = Field(default=0.25, ge=0.001, le=1)
    DETECTOR_WEIGHTS: Path = ROOT / "weights" / "detector_v2.pt"
    CLASSIFIER_WEIGHTS: Path = ROOT / "weights" / "classifier_v2.pt"
    CLASSIFIER_THRESHOLD: float | None = Field(default=None, ge=0, le=1)
    DETECTOR_IMAGE_SIZE: int = Field(default=384, gt=0)
    DETECTOR_VIEW: Literal["full", "center", "tiles"] = "full"
    DETECTOR_MAX_DETECTIONS: int = Field(default=100, gt=0)
    ML_DEVICE: str = "auto"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
