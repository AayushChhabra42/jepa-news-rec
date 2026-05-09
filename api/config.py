"""Runtime configuration for the FastAPI demo service."""

from functools import lru_cache
import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JEPA_", env_file=".env", extra="ignore")

    processed_dir: Path = Field(default=ROOT_DIR / "data" / "processed")
    checkpoint_path: Path = Field(default=ROOT_DIR / "checkpoints" / "jepa_best.pt")
    finetuned_checkpoint_path: Path = Field(default=ROOT_DIR / "checkpoints" / "finetuned_model.pt")
    xgb_checkpoint_path: Path = Field(default=ROOT_DIR / "checkpoints" / "xgb_ranker.pkl")
    top_k_default: int = 50
    max_seq_len: int = 50
    predictor_type: str = "transformer"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def processed_data_path(self) -> Path:
        return self.processed_dir / "processed_data.pkl"

    @property
    def news_path(self) -> Path:
        return self.processed_dir / "news.pkl"

    @property
    def user_sequences_path(self) -> Path:
        return self.processed_dir / "user_sequences.pkl"

    @property
    def impressions_path(self) -> Path:
        return self.processed_dir / "impressions.pkl"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.processed_dir = Path(os.path.expandvars(str(settings.processed_dir))).resolve()
    settings.checkpoint_path = Path(os.path.expandvars(str(settings.checkpoint_path))).resolve()
    settings.finetuned_checkpoint_path = Path(
        os.path.expandvars(str(settings.finetuned_checkpoint_path))
    ).resolve()
    settings.xgb_checkpoint_path = Path(os.path.expandvars(str(settings.xgb_checkpoint_path))).resolve()
    return settings
