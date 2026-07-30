"""Shared enumeration of the roof-condition label pool (damage v5 scripts).

Replays make_label_pool.py's building order EXACTLY — same AOIs, same >=20 px
filter on the same 0.15 m / 2000 px tiles — so the integer idx a human labelled
still maps to the same building. Emits fixed-GSD roof tiles through the shared
``worker.pipeline.roof_chip`` module, so training, active-learning mining, and
production inference all crop identical chips.
"""

import sys
import time
import warnings
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geoai  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.transform import from_bounds, rowcol  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from worker.pipeline.roof_chip import roof_tiles  # noqa: E402

POOL = Path("/data/imagery/label_pool")
IMG_URL = (
    "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
    "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer/exportImage"
)
# EXACT same AOI list + order as make_label_pool.py (Palm St held out for eval).
AOIS = [
    [-90.2100, 38.6555, -90.2060, 38.6580],
    [-90.2260, 38.6560, -90.2210, 38.6588],
    [-90.2160, 38.6615, -90.2110, 38.6642],
    [-90.2340, 38.6520, -90.2295, 38.6548],
    [-90.2050, 38.6600, -90.2010, 38.6625],
    [-90.4450, 38.5900, -90.4405, 38.5928],
    [-90.3550, 38.5950, -90.3505, 38.5978],
    [-90.4600, 38.6000, -90.4555, 38.6028],
    [-90.3350, 38.6270, -90.3305, 38.6298],
    [-90.2980, 38.6180, -90.2935, 38.6208],
]


def hires(bbox, mpp=0.15, maxpx=2000):
    """Fetch a leaf-off RGB tile for bbox (EPSG:4326) -> (arr, 3857 transform)."""
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    sx, sy = maxx - minx, maxy - miny
    scale = min(maxpx / (sx / mpp), maxpx / (sy / mpp), 1.0)
    w, h = int(sx / mpp * scale), int(sy / mpp * scale)
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{w},{h}",
        "format": "png",
        "f": "image",
    }
    for a in range(4):
        try:
            r = requests.get(IMG_URL, params=params, timeout=120)
            r.raise_for_status()
            arr = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
            return arr, from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
        except Exception:
            if a == 3:
                raise
            time.sleep(8 * (a + 1))


_tiles: dict[int, tuple] = {}


def aoi_tile(ai):
    if ai not in _tiles:
        _tiles[ai] = hires(AOIS[ai])
    return _tiles[ai]


def iter_buildings():
    """Yield (idx, aoi, geom_3857) in the exact make_label_pool order/filter."""
    idx = 0
    for ai, bbox in enumerate(AOIS):
        ov = POOL / f"ov_{ai}.geojson"
        if not ov.exists():
            geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
        try:
            gdf = gpd.read_file(ov).to_crs("EPSG:3857")
        except Exception:
            continue
        arr, tf = aoi_tile(ai)
        H, W, _ = arr.shape
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            r0, c0 = rowcol(tf, minx, maxy)
            r1, c1 = rowcol(tf, maxx, miny)
            r0, r1 = max(0, min(r0, r1)), min(H, max(r0, r1))
            c0, c1 = max(0, min(c0, c1)), min(W, max(c0, c1))
            if r1 - r0 < 20 or c1 - c0 < 20:
                continue
            yield idx, ai, geom
            idx += 1


def building_tiles(ai, geom):
    """Fixed-GSD, polygon-masked roof tiles for one building (uint8 lists)."""
    arr, tf = aoi_tile(ai)
    return roof_tiles(arr, tf, geom)
