import math
import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


def bbox_area_km2(bbox: list[float]) -> float:
    """Approximate area of a WGS84 bbox in km². Good to <1% at these AOI sizes."""
    min_lon, min_lat, max_lon, max_lat = bbox
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians((min_lat + max_lat) / 2))
    return (max_lat - min_lat) * km_per_deg_lat * (max_lon - min_lon) * km_per_deg_lon


class JobCreate(BaseModel):
    bbox: list[float] = Field(
        description="[min_lon, min_lat, max_lon, max_lat] in EPSG:4326",
        min_length=4,
        max_length=4,
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, v: list[float]) -> list[float]:
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("longitude out of range or min >= max")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("latitude out of range or min >= max")
        return v


class JobOut(BaseModel):
    id: uuid.UUID
    status: str
    bbox: list[float]
    backend: str
    error: str | None = None
    building_count: int | None = None
    is_seed: bool = False
    validation_status: str | None = None
    validation_error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ValidationSummary(BaseModel):
    parcels_total: int
    parcels_with_buildings: int
    parcels_empty: int
    buildings_total: int
    buildings_off_parcel: int
    buildings_crossing: int


class HealthOut(BaseModel):
    status: str
    db: bool
    redis: bool
