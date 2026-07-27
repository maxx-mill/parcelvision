"""Validate zero-shot CLIP roof-damage scoring against REAL damaged vs intact
roofs. Damaged AOI = 2020 Palm St (north St. Louis City vacancy); normal AOI =
the University City demo block. Uses Overture footprints to crop actual roofs
from 0.15 m leaf-off imagery, scores each with CLIP, and saves the highest- and
lowest-scored chips so a human can check whether CLIP is right.

    docker run ... python scripts/validate_damage_clip.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io import BytesIO  # noqa: E402

import geoai  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.transform import from_bounds, rowcol  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

OUT = Path("/data/imagery/damage_val")
OUT.mkdir(parents=True, exist_ok=True)
AOIS = {
    "palm_damaged": [-90.2035, 38.6548, -90.1999, 38.6568],
    "demo_normal": [-90.3150, 38.6478, -90.3120, 38.6498],
}
INTACT = [
    "an aerial view of a house with an intact, well-maintained roof",
    "a residential building in good repair seen from above",
    "a normal grey shingled roof in good condition",
    "an ordinary apartment or house roof with vents and skylights",
    "a solid intact rooftop, no damage",
]
DAMAGED = [
    "an aerial view of a roof with a large hole or missing section",
    "a collapsed caved-in roof exposing the building interior",
    "a burned-out roofless abandoned building",
    "a rooftop covered in debris or overgrown with vegetation",
]


def hires(bbox, mpp=0.15, maxpx=2000):
    # PNG (not tiff) at a capped size — the ImageServer 500s on large tiff
    # exports; we build the affine ourselves so georeferencing is exact anyway.
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    spanx, spany = maxx - minx, maxy - miny
    scale = min(maxpx / (spanx / mpp), maxpx / (spany / mpp), 1.0)
    w, h = int(spanx / mpp * scale), int(spany / mpp * scale)
    url = (
        "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
        "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer/exportImage"
    )
    r = requests.get(
        url,
        params={
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{w},{h}",
            "format": "png",
            "f": "image",
        },
        timeout=120,
    )
    r.raise_for_status()
    arr = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
    transform = from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
    return arr, transform


print("loading CLIP…")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
PROMPTS = INTACT + DAMAGED


def damaged_score(chip: Image.Image) -> float:
    with torch.no_grad():
        inp = proc(text=PROMPTS, images=chip, return_tensors="pt", padding=True)
        logits = model(**inp).logits_per_image[0]  # (6,) image↔prompt scores
    intact = logits[: len(INTACT)].max()
    damaged = logits[len(INTACT) :].max()
    return float(torch.softmax(torch.stack([intact, damaged]), 0)[1])


for name, bbox in AOIS.items():
    gdf = gpd.GeoDataFrame()
    ov = OUT / f"{name}_overture.geojson"
    if not ov.exists():
        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
    gdf = gpd.read_file(ov).to_crs("EPSG:3857")
    arr, transform = hires(bbox)
    H, W, _ = arr.shape
    scored = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        r0, c0 = rowcol(transform, minx, maxy)
        r1, c1 = rowcol(transform, maxx, miny)
        r0, r1 = max(0, min(r0, r1)), min(H, max(r0, r1))
        c0, c1 = max(0, min(c0, c1)), min(W, max(c0, c1))
        if r1 - r0 < 12 or c1 - c0 < 12:
            continue
        chip = Image.fromarray(arr[r0:r1, c0:c1])
        scored.append((damaged_score(chip), chip))
    if not scored:
        print(f"{name}: no roofs")
        continue
    scores = np.array([s for s, _ in scored])
    print(
        f"\n{name}: n={len(scores)} damaged_score "
        f"mean={scores.mean():.2f} median={np.median(scores):.2f} "
        f"p90={np.percentile(scores, 90):.2f} max={scores.max():.2f}"
    )
    scored.sort(key=lambda x: -x[0])
    for i, (s, chip) in enumerate(scored[:4]):
        chip.resize((128, 128)).save(OUT / f"{name}_HI_{i}_{s:.2f}.png")
    for i, (s, chip) in enumerate(scored[-4:]):
        chip.resize((128, 128)).save(OUT / f"{name}_LO_{i}_{s:.2f}.png")
print("\nsaved example chips to", OUT)
