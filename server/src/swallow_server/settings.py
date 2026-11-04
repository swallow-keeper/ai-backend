from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_registry() -> Path:
    packaged = Path(__file__).resolve().parents[3] / "registry" / "model_registry.yaml"
    if packaged.exists():
        return packaged
    return Path("registry/model_registry.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWALLOW_", extra="ignore")

    database_url: str = "sqlite:///data/swallow.db"
    registry_path: Path = Field(default_factory=_default_registry)
    dev_token: str = "dev-token"
    dev_display_name: str = "개발용 사용자"
    data_key: str = "dev-only-not-for-production"
