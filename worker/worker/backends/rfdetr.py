"""RF-DETR-Seg backend — merve/rf-detr-seg-satellite-buildings via transformers.

Chosen from the detector eval (worker/scripts/eval_detectors.py): on residential
parcels it gave the best precision (0.52 vs Mask R-CNN's 0.41) and best F1, i.e.
cleaner per-structure footprints with fewer phantom buildings — the right bias
for parcel-level property intelligence. transformers already ships
RfDetrForInstanceSegmentation, so no new dependency.

The model is a generic instance-seg net with no geospatial wrapper, so we tile
the georeferenced raster ourselves and map each tile's masks back to world
coordinates via the window's affine transform, then NMS across tile overlaps.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np

HF_REPO = "merve/rf-detr-seg-satellite-buildings"
WINDOW = 640
OVERLAP = 128
CONF_THRESHOLD = 0.3
NMS_IOU = 0.5


class RFDetrBackend:
    name = "rfdetr"
    needs_imagery = True
    device = "cpu"

    def extract(self, raster_paths: list[Path], bbox: list[float]) -> gpd.GeoDataFrame:
        import pandas as pd

        predict = self._load()
        parts: list[gpd.GeoDataFrame] = []
        for path in raster_paths:
            gdf = _tiled_detect(str(path), predict)
            if not gdf.empty:
                parts.append(gdf if not parts else gdf.to_crs(parts[0].crs))
        if not parts:
            return gpd.GeoDataFrame({"confidence": []}, geometry=[], crs=4326)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)

    def _load(self):
        """Return predict(rgb_uint8, window_affine) -> (polys_geo, scores)."""
        import torch
        from PIL import Image
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape
        from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

        proc = AutoImageProcessor.from_pretrained(HF_REPO)
        model = RfDetrForInstanceSegmentation.from_pretrained(HF_REPO).to(self.device).eval()

        def predict(rgb, tf):
            inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            res = proc.post_process_instance_segmentation(
                outputs, threshold=CONF_THRESHOLD, target_sizes=[rgb.shape[:2]]
            )[0]
            seg = res["segmentation"].cpu().numpy()
            polys, scores = [], []
            for info in res["segments_info"]:
                mask = (seg == info["id"]).astype("uint8")
                for geom, val in rio_shapes(mask, mask=mask.astype(bool), transform=tf):
                    if val:
                        poly = shape(geom)
                        if poly.is_valid and poly.area > 0:
                            polys.append(poly)
                            scores.append(float(info["score"]))
            return polys, scores

        return predict


def _tiled_detect(tif: str, predict) -> gpd.GeoDataFrame:
    import rasterio
    from rasterio.windows import Window

    polys, scores = [], []
    with rasterio.open(tif) as src:
        crs = src.crs
        step = WINDOW - OVERLAP
        for row in range(0, src.height, step):
            for col in range(0, src.width, step):
                w = min(WINDOW, src.width - col)
                h = min(WINDOW, src.height - row)
                if w < 64 or h < 64:
                    continue
                win = Window(col, row, w, h)
                rgb = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype("uint8")
                p, s = predict(rgb, src.window_transform(win))
                polys.extend(p)
                scores.extend(s)
    if not polys:
        return gpd.GeoDataFrame({"confidence": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame({"confidence": scores}, geometry=polys, crs=crs)
    return _nms_polygons(gdf, NMS_IOU)


def _nms_polygons(gdf: gpd.GeoDataFrame, iou_thr: float) -> gpd.GeoDataFrame:
    """Greedy polygon NMS to merge duplicate detections across tile overlaps."""
    gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept, keep_idx = [], []
    for i, g in enumerate(gdf.geometry):
        if not g.is_valid:
            g = g.buffer(0)
        if any(
            (inter := g.intersection(kg).area) and inter / (g.area + kg.area - inter) > iou_thr
            for kg in kept
        ):
            continue
        kept.append(g)
        keep_idx.append(i)
    return gdf.iloc[keep_idx].reset_index(drop=True)
