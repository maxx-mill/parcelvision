from pathlib import Path

import geopandas as gpd
import pandas as pd


class LocalCPUBackend:
    """Pretrained Mask R-CNN (geoai `building_footprints_usa` checkpoint,
    auto-downloaded from Hugging Face on first run and cached in the
    `model_cache` volume). CPU inference: expect minutes per km², which is why
    AOI size is capped and jobs are async."""

    name = "local_cpu"
    needs_imagery = True
    device = "cpu"

    def extract(self, raster_paths: list[Path], bbox: list[float]) -> gpd.GeoDataFrame:
        from geoai import BuildingFootprintExtractor  # defers torch import to job time

        extractor = BuildingFootprintExtractor(device=self.device)
        parts: list[gpd.GeoDataFrame] = []
        for path in raster_paths:
            gdf = extractor.process_raster(
                str(path),
                batch_size=2,
                filter_edges=False,  # tiles are pre-clipped to the AOI; edges are real
                confidence_threshold=0.5,
                overlap=0.25,
                nms_iou_threshold=0.5,
                mask_threshold=0.5,
                min_object_area=50,  # pixels: ~18 m² at NAIP 0.6 m — floor of MIN_AREA_SQM
                simplify_tolerance=1.0,
            )
            if gdf is not None and not gdf.empty:
                # Adjacent NAIP tiles can sit in different UTM zones; align to the first.
                parts.append(gdf if not parts else gdf.to_crs(parts[0].crs))
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
