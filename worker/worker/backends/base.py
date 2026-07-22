from pathlib import Path
from typing import Protocol

import geopandas as gpd


class Backend(Protocol):
    """A segmentation backend turns imagery (or, for the fake backend, just the
    bbox) into a GeoDataFrame of building polygons with a `confidence` column,
    in any CRS with `crs` set. Post-processing handles the rest."""

    name: str
    needs_imagery: bool

    def extract(self, raster_paths: list[Path], bbox: list[float]) -> gpd.GeoDataFrame: ...
