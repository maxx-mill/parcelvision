from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """12-factor config — every value comes from the environment (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "parcelvision"
    postgres_user: str = "parcelvision"
    postgres_password: str = "parcelvision-dev-only"

    redis_url: str = "redis://localhost:6379/0"

    inference_backend: str = "local_cpu"
    aoi_bbox_limit_km2: float = 1.0
    naip_year: int | None = None

    frontend_origin: str = "http://localhost:3000"

    @field_validator("naip_year", mode="before")
    @classmethod
    def _blank_env_is_none(cls, v: object) -> object:
        # .env ships `NAIP_YEAR=` (unset); pydantic won't parse "" as int|None.
        return None if v == "" else v

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
