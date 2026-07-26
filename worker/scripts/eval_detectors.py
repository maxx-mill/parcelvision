"""Compare building detectors for the parcel-level real-estate use case.

Business framing (see memory: parcelvision-business-goal): we care about clean
per-structure detection on residential/commercial *parcels*, not recall on large
downtown blocks. So we score each detector against Overture footprints (a
per-structure reference) on residential + commercial AOIs, and also report
detections-per-parcel as a proxy for over/under-segmentation.

Detectors:
  * maskrcnn — current default (geoai building_footprints_usa Mask R-CNN)
  * yolov8m  — keremberke/yolov8m-building-segmentation (satellite, ultralytics)
  * rfdetr   — merve/rf-detr-seg-satellite-buildings (transformers RF-DETR-Seg)

The last two are generic instance-seg models with no geospatial wrapper, so a
shared sliding-window harness maps their per-tile masks back to georeferenced
polygons. Run inside the eval image (worker-full + ultralytics):

    docker compose run --rm --no-deps worker python scripts/eval_detectors.py
"""

import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import Polygon, box, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.pipeline.parcels import fetch_parcels  # noqa: E402
from worker.pipeline.postprocess import postprocess  # noqa: E402

# Residential + commercial St. Louis County parcels (the real-estate use case).
AOIS = {
    "residential": [-90.3167, 38.6465, -90.3111, 38.6501],
    "commercial": [-90.3470, 38.6235, -90.3445, 38.6258],
}
CACHE = Path("/data/imagery/detector_eval")
IOU_MATCH = 0.5
WINDOW = 640
OVERLAP = 128


# --------------------------------------------------------------------------- #
# Shared sliding-window harness for generic instance-seg models
# --------------------------------------------------------------------------- #
def _nms_polygons(gdf: gpd.GeoDataFrame, iou_thr: float = 0.5) -> gpd.GeoDataFrame:
    """Greedy polygon NMS to merge duplicate detections across tile overlaps."""
    if gdf.empty:
        return gdf
    gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept: list[Polygon] = []
    keep_idx: list[int] = []
    for i, g in enumerate(gdf.geometry):
        if not g.is_valid:
            g = g.buffer(0)
        dup = False
        for kg in kept:
            inter = g.intersection(kg).area
            if inter and inter / (g.area + kg.area - inter) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(g)
            keep_idx.append(i)
    return gdf.iloc[keep_idx].reset_index(drop=True)


def tiled_detect(tif: str, predict) -> gpd.GeoDataFrame:
    """predict(rgb_uint8_HxWx3, window_affine) -> (list[Polygon_geo], list[score])."""
    polys: list[Polygon] = []
    scores: list[float] = []
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
                arr = src.read([1, 2, 3], window=win)
                rgb = np.transpose(arr, (1, 2, 0)).astype("uint8")
                p, s = predict(rgb, src.window_transform(win))
                polys.extend(p)
                scores.extend(s)
    if not polys:
        return gpd.GeoDataFrame({"confidence": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame({"confidence": scores}, geometry=polys, crs=crs)
    return _nms_polygons(gdf, IOU_MATCH)


# --------------------------------------------------------------------------- #
# Detector adapters
# --------------------------------------------------------------------------- #
def detect_maskrcnn(tif: str) -> gpd.GeoDataFrame:
    from geoai import BuildingFootprintExtractor

    ex = BuildingFootprintExtractor(device="cpu")
    return ex.process_raster(
        tif,
        batch_size=2,
        filter_edges=False,
        confidence_threshold=0.5,
        overlap=0.25,
        nms_iou_threshold=0.5,
        mask_threshold=0.5,
        min_object_area=50,
        simplify_tolerance=1.0,
    )


def detect_yolov8(tif: str) -> gpd.GeoDataFrame:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights = hf_hub_download("keremberke/yolov8m-building-segmentation", "best.pt")
    model = YOLO(weights)

    def predict(rgb, tf):
        res = model.predict(rgb, conf=0.25, iou=0.5, imgsz=WINDOW, verbose=False)[0]
        polys, scores = [], []
        if res.masks is not None:
            confs = res.boxes.conf.cpu().numpy()
            # ultralytics aligns masks and boxes 1:1
            for xy, c in zip(res.masks.xy, confs, strict=True):  # xy: Nx2 pixel coords
                if len(xy) < 3:
                    continue
                geo = [tf * (float(x), float(y)) for x, y in xy]
                poly = Polygon(geo)
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
                    scores.append(float(c))
        return polys, scores

    return tiled_detect(tif, predict)


def detect_rfdetr(tif: str) -> gpd.GeoDataFrame:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

    repo = "merve/rf-detr-seg-satellite-buildings"
    proc = AutoImageProcessor.from_pretrained(repo)
    model = RfDetrForInstanceSegmentation.from_pretrained(repo).eval()

    def predict(rgb, tf):
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_instance_segmentation(
            outputs, threshold=0.3, target_sizes=[rgb.shape[:2]]
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

    return tiled_detect(tif, predict)


# YOLOv8m dropped after the first eval (recall 0.11 — resolution mismatch at
# NAIP 0.6 m). detect_yolov8 stays defined above for reference/reproducibility.
DETECTORS = {
    "maskrcnn": detect_maskrcnn,
    "rfdetr": detect_rfdetr,
}
# Imagery sources to compare. leaf-off is the new default; naip is the leaf-on
# baseline (cached for residential; commercial unavailable during PC outages).
IMAGERY = ["mo_leafoff", "naip"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> dict:
    if pred is None or pred.empty:
        return {"detections": 0, "precision": 0, "recall": 0, "f1": 0, "per_parcel": 0.0}
    pred = pred.to_crs(ref.crs)
    pairs = []
    for pi, pg in enumerate(pred.geometry):
        if not pg.is_valid:
            pg = pg.buffer(0)
        for ri, rg in enumerate(ref.geometry):
            inter = pg.intersection(rg).area
            if inter == 0:
                continue
            iou = inter / (pg.area + rg.area - inter)
            if iou >= IOU_MATCH:
                pairs.append((iou, pi, ri))
    pairs.sort(reverse=True)
    up, ur = set(), set()
    tp = 0
    for _, pi, ri in pairs:
        if pi in up or ri in ur:
            continue
        up.add(pi)
        ur.add(ri)
        tp += 1
    fp = len(pred) - tp
    fn = len(ref) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # detections per parcel that actually has a structure — over-seg proxy
    pred_utm = pred.to_crs(parcels.crs)
    joined = gpd.sjoin(pred_utm, parcels, predicate="intersects", how="inner")
    per_parcel = len(joined) / max(joined["index_right"].nunique(), 1)
    return {
        "detections": len(pred),
        "tp": tp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "per_parcel": round(per_parcel, 2),
    }


def get_reference(name: str, bbox: list[float]) -> gpd.GeoDataFrame:
    import geoai

    ref_path = CACHE / f"{name}_overture.geojson"
    if not ref_path.exists():
        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ref_path))
    ref = gpd.read_file(ref_path)
    ref = ref[ref.geometry.intersects(box(*bbox))]
    return ref.to_crs(ref.estimate_utm_crs())


def get_tiles(name: str, bbox: list[float], source: str) -> list[Path] | None:
    """Fetch (and cache) the AOI tile for one imagery source. NAIP reuses any
    cached clip_*.tif so residential still works during a Planetary Computer
    outage; returns None when a source is unavailable (skip, don't abort)."""
    from worker.pipeline.fetch import _fetch_mo_leafoff, _fetch_naip

    aoi_dir = CACHE / name / source
    aoi_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(aoi_dir.glob("*.tif"))
    if existing:
        return existing
    # Legacy residential NAIP tiles were cached one level up as clip_*.tif.
    if source == "naip":
        legacy = sorted((CACHE / name).glob("clip_*.tif"))
        if legacy:
            return legacy
    try:
        if source == "mo_leafoff":
            return _fetch_mo_leafoff(bbox, aoi_dir)
        if source == "naip":
            return _fetch_naip(bbox, aoi_dir, None)
    except Exception as exc:
        print(f"    ({source} unavailable: {str(exc)[:50]})")
    return None


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, bbox in AOIS.items():
        try:
            ref = get_reference(name, bbox)
            parcels = fetch_parcels(bbox).to_crs(ref.crs)
        except Exception as exc:
            print(f"\n=== AOI {name}: SKIPPED (reference/parcels failed: {str(exc)[:60]}) ===")
            continue
        print(f"\n=== AOI {name}: {len(ref)} reference buildings, {len(parcels)} parcels ===")
        for source in IMAGERY:
            tiles = get_tiles(name, bbox, source)
            if not tiles:
                continue
            tif = str(tiles[0])
            for det_name, det in DETECTORS.items():
                try:
                    clean = postprocess(det(tif), bbox)
                    s = score(clean, ref, parcels)
                    rows.append({"aoi": name, "imagery": source, "detector": det_name, **s})
                    print(f"  {source:11s} {det_name:9s} {s}")
                except Exception as exc:
                    print(f"  {source:11s} {det_name:9s} FAILED: {exc}")
                    traceback.print_exc()

    print("\n================ SUMMARY (per-structure vs Overture, IoU>=0.5) ================")
    df = pd.DataFrame(rows)
    if df.empty or "f1" not in df:
        print("no successful runs")
        return
    print(df.to_string(index=False))
    print("\nby imagery source (mean):")
    print(df.groupby("imagery")[["precision", "recall", "f1"]].mean().round(3).to_string())
    print("\nby detector (mean):")
    print(
        df.groupby("detector")[["precision", "recall", "f1", "per_parcel"]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
