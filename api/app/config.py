from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_DB_PASSWORD = "parcelvision-dev-only"


class Settings(BaseSettings):
    """12-factor config — every value comes from the environment (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "development" (default) or "production". Production fails fast on dev secrets.
    app_env: str = "development"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "parcelvision"
    postgres_user: str = "parcelvision"
    postgres_password: str = _DEV_DB_PASSWORD

    redis_url: str = "redis://localhost:6379/0"

    inference_backend: str = "rfdetr"
    aoi_bbox_limit_km2: float = 1.0
    naip_year: int | None = None
    # Per-client cap on job creation (a capped-AOI CPU job still costs minutes).
    job_rate_limit_per_min: int = 10

    frontend_origin: str = "http://localhost:3000"

    # Parcel lookup (Chapter 7) — same county ArcGIS service the worker validates against.
    parcel_service_url: str = (
        "https://maps.stlouisco.com/hosting/rest/services/Maps/AGS_Parcels/MapServer/0"
    )
    parcel_locator_field: str = "LOCATOR"
    parcel_address_field: str = "PROP_ADD"

    @field_validator("naip_year", mode="before")
    @classmethod
    def _blank_env_is_none(cls, v: object) -> object:
        # .env ships `NAIP_YEAR=` (unset); pydantic won't parse "" as int|None.
        return None if v == "" else v

    @model_validator(mode="after")
    def _no_dev_secrets_in_prod(self) -> "Settings":
        # Fail fast rather than quietly shipping the shared dev password to prod.
        if self.app_env.lower() == "production" and self.postgres_password == _DEV_DB_PASSWORD:
            raise ValueError(
                "APP_ENV=production but POSTGRES_PASSWORD is the built-in dev default; "
                "set a real secret."
            )
        return self

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
