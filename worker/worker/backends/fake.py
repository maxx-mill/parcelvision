"""Deterministic synthetic footprints for tests/CI — no imagery, no ML deps.

Never present this backend's output as real detections. It exists so the whole
job lifecycle (enqueue -> stages -> PostGIS -> GeoJSON out) can run in seconds
in environments that can't hold torch, and so integration tests are stable.
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


class FakeBackend:
    name = "fake"
    needs_imagery = False

    # ~12x9 m "houses" every 40 m, i.e. a caricature of a residential block.
    SPACING_M = 40.0
    WIDTH_M = 12.0
    DEPTH_M = 9.0

    def extract(self, raster_paths: list[Path], bbox: list[float]) -> gpd.GeoDataFrame:
        aoi = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        utm = aoi.estimate_utm_crs()
        minx, miny, maxx, maxy = aoi.to_crs(utm).total_bounds

        geoms = []
        y = miny + self.SPACING_M / 2
        while y + self.DEPTH_M < maxy:
            x = minx + self.SPACING_M / 2
            while x + self.WIDTH_M < maxx:
                geoms.append(box(x, y, x + self.WIDTH_M, y + self.DEPTH_M))
                x += self.SPACING_M
            y += self.SPACING_M
        gdf = gpd.GeoDataFrame({"confidence": [0.99] * len(geoms)}, geometry=geoms, crs=utm)
        # The UTM bounding rect of a geographic bbox bows past the bbox itself
        # after reprojection; keep only footprints strictly inside the AOI.
        return gdf[gdf.to_crs(4326).within(box(*bbox))].reset_index(drop=True)
