"""Score building extraction against Overture reference footprints and sweep
inference parameters. Run inside the worker-full container:

    docker compose run --rm --no-deps worker python scripts/eval_extraction.py

Overture buildings (largely Microsoft/OSM-derived) are an imperfect but
independent reference: good enough to rank configs by precision/recall, not
good enough to be treated as ground truth. Imagery and the Overture layer are
cached under /data/imagery/eval_cache, so re-runs only pay inference time.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapely.geometry import box  # noqa: E402

from worker.pipeline.fetch import fetch_imagery  # noqa: E402
from worker.pipeline.postprocess import postprocess  # noqa: E402

DEMO_BBOX = [-90.3167, 38.6465, -90.3111, 38.6501]
CACHE = Path("/data/imagery/eval_cache")
IOU_MATCH = 0.5

CONFIGS = [
    # name, chip_size, confidence_threshold, mask_threshold
    ("baseline-512", 512, 0.5, 0.5),
    ("chip-1024", 1024, 0.5, 0.5),
    ("chip-1024-conf40", 1024, 0.4, 0.4),
    ("conf40", 512, 0.4, 0.4),
]


def get_rasters() -> list[Path]:
    marker = CACHE / ".complete"
    if marker.exists():
        return sorted(CACHE.glob("clip_*.tif"))
    CACHE.mkdir(parents=True, exist_ok=True)
    rasters = fetch_imagery(DEMO_BBOX, CACHE)
    marker.touch()
    return rasters


def get_reference() -> gpd.GeoDataFrame:
    ref_path = CACHE / "overture_buildings.geojson"
    if not ref_path.exists():
        import geoai

        geoai.download_overture_buildings(bbox=tuple(DEMO_BBOX), output=str(ref_path))
    ref = gpd.read_file(ref_path)
    ref = ref[ref.geometry.intersects(box(*DEMO_BBOX))]
    return ref.to_crs(ref.estimate_utm_crs())


def score(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame) -> dict:
    """Greedy IoU matching: each reference footprint pairs with at most one
    detection. n and m are ~10², so brute force is fine."""
    pred = pred.to_crs(ref.crs)
    pairs = []
    for pi, pg in enumerate(pred.geometry):
        for ri, rg in enumerate(ref.geometry):
            inter = pg.intersection(rg).area
            if inter == 0:
                continue
            iou = inter / (pg.area + rg.area - inter)
            if iou >= IOU_MATCH:
                pairs.append((iou, pi, ri))
    pairs.sort(reverse=True)
    used_p, used_r, ious = set(), set(), []
    for iou, pi, ri in pairs:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        ious.append(iou)
    tp = len(ious)
    fp = len(pred) - tp
    fn = len(ref) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "detections": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_iou": round(sum(ious) / max(len(ious), 1), 3),
    }


def main() -> None:
    from geoai import BuildingFootprintExtractor

    rasters = get_rasters()
    ref = get_reference()
    print(f"rasters: {[p.name for p in rasters]}; reference footprints: {len(ref)}\n")

    extractor = BuildingFootprintExtractor(device="cpu")  # model loads once
    rows = []
    for name, chip, conf, mask in CONFIGS:
        parts = []
        for p in rasters:
            gdf = extractor.process_raster(
                str(p),
                batch_size=2,
                filter_edges=False,
                chip_size=(chip, chip),
                confidence_threshold=conf,
                overlap=0.25,
                nms_iou_threshold=0.5,
                mask_threshold=mask,
                min_object_area=50,
                simplify_tolerance=1.0,
            )
            if gdf is not None and not gdf.empty:
                parts.append(gdf if not parts else gdf.to_crs(parts[0].crs))
        pred = (
            postprocess(gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs), DEMO_BBOX)
            if parts
            else gpd.GeoDataFrame(geometry=[], crs=4326)
        )
        row = {"config": name, **score(pred, ref)}
        rows.append(row)
        print(row)

    print("\n== summary (sorted by f1) ==")
    for r in sorted(rows, key=lambda r: -r["f1"]):
        print(r)


if __name__ == "__main__":
    main()
