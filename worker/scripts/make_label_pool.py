"""Damage v4 — generate a roof-chip pool for HAND labeling.

Weak geographic labels (v3) under-call obvious damage. This crops actual roof
chips at 0.15 m from mixed AOIs (north-city derelict + intact suburbs + intact
flat-roof commercial to kill the flat-roof confound), saves them indexed, and
builds montages a human labels by eye. The retrainer reads chips.jsonl + the
labels a human records.
"""

import json
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
from PIL import Image, ImageDraw  # noqa: E402
from rasterio.transform import from_bounds, rowcol  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

OUT = Path("/data/imagery/label_pool")
(OUT / "chips").mkdir(parents=True, exist_ok=True)
CHIP = 128
# Mixed AOIs. Palm St held out for eval; these are OTHER blocks for labeling.
AOIS = [
    # north-city derelict/mixed (expect both damaged + intact)
    [-90.2100, 38.6555, -90.2060, 38.6580],
    [-90.2260, 38.6560, -90.2210, 38.6588],
    [-90.2160, 38.6615, -90.2110, 38.6642],
    [-90.2340, 38.6520, -90.2295, 38.6548],
    [-90.2050, 38.6600, -90.2010, 38.6625],
    # intact suburbs
    [-90.4450, 38.5900, -90.4405, 38.5928],
    [-90.3550, 38.5950, -90.3505, 38.5978],
    [-90.4600, 38.6000, -90.4555, 38.6028],
    # intact flat-roof commercial / mixed (teach flat != damaged)
    [-90.3350, 38.6270, -90.3305, 38.6298],
    [-90.2980, 38.6180, -90.2935, 38.6208],
]


def hires(bbox, mpp=0.15, maxpx=2000):
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    sx, sy = maxx - minx, maxy - miny
    scale = min(maxpx / (sx / mpp), maxpx / (sy / mpp), 1.0)
    w, h = int(sx / mpp * scale), int(sy / mpp * scale)
    url = (
        "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
        "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer/exportImage"
    )
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
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            arr = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
            return arr, from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
        except Exception:
            if a == 3:
                raise
            time.sleep(8 * (a + 1))


def main():
    meta = []
    idx = 0
    for ai, bbox in enumerate(AOIS):
        ov = OUT / f"ov_{ai}.geojson"
        if not ov.exists():
            geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
        try:
            gdf = gpd.read_file(ov).to_crs("EPSG:3857")
        except Exception:
            continue
        arr, tf = hires(bbox)
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
            chip = Image.fromarray(arr[r0:r1, c0:c1]).resize((CHIP, CHIP))
            chip.save(OUT / "chips" / f"{idx:04d}.png")
            meta.append({"idx": idx, "aoi": ai})
            idx += 1
        print(f"  AOI {ai}: pool now {idx}")
    (OUT / "chips.jsonl").write_text("\n".join(json.dumps(m) for m in meta))

    # montages of 24 chips each, index labelled per cell
    per = 24
    cols = 6
    for page in range((idx + per - 1) // per):
        cell = 150
        rows = (per + cols - 1) // cols
        m = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 22))
        dr = ImageDraw.Draw(m)
        for j in range(per):
            i = page * per + j
            if i >= idx:
                break
            c = Image.open(OUT / "chips" / f"{i:04d}.png").resize((cell - 6, cell - 6))
            x, y = (j % cols) * cell, (j // cols) * cell
            m.paste(c, (x + 3, y + 3))
            dr.rectangle([x + 2, y + 2, x + 40, y + 16], fill=(0, 0, 0))
            dr.text((x + 4, y + 4), str(i), fill=(255, 240, 0))
        m.save(OUT / f"montage_{page:02d}.png")
    print(f"\npool: {idx} chips, {(idx + per - 1) // per} montages -> {OUT}")


if __name__ == "__main__":
    main()
