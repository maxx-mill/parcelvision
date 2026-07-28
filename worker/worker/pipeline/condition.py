"""Stage 3.5: per-structure roof condition (Chapter 6 — damage v4, in-domain).

Off-the-shelf models (CLIP, xView2 SegFormer, RescueNet YOLO) all failed to
transfer to our imagery, so we train an IN-DOMAIN ResNet18 on leaf-off roof
chips. v3 used weak geographic labels and under-called obvious damage; v4
(worker/scripts/{make_label_pool,train_condition_v4}.py) uses ~120 HAND-labelled
0.15 m chips + augmentation and is markedly crisper. Held-out per-building
validation, 2020 Palm St (damaged) vs the demo (intact):

  P(damaged) median   v3 -> v4
  Palm damaged        0.97 -> 1.00   (and intact buildings WITHIN Palm -> 0.00)
  demo intact         0.18 -> 0.02

It's strong on residential roofs (the business target) and, unlike CLIP, scores
a swimming pool 0.00. It still over-flags large flat/institutional roofs (a
campus building is not a house) — framed honestly, not a certified inspection.

Per footprint we record roof_damage_score (classifier P(damaged)) and
tarp_fraction (an unambiguous complementary colour signal), and a condition
flag: tarp > damaged > review > ok. The chip crop needs higher resolution than
the detection imagery, so this stage fetches its own ~0.15 m leaf-off tile.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "roof_condition_resnet18.pt"
CHIP_PX = 128
CHIP_MPP = 0.15  # native leaf-off res; matches the v4 classifier's training chips

# Condition thresholds on classifier P(damaged). Calibrated on held-out eval
# (intact demo median 0.18, damaged Palm St median 0.97).
DAMAGED_SCORE = 0.60
REVIEW_SCORE = 0.35

# Blue-tarp colour rule (0-255 RGB) — strict so winter's blue-grey cast on
# ordinary roofs scores ~0; kept as an unambiguous complementary signal.
TARP_BLUE_MIN = 120
TARP_B_OVER_R = 1.6
TARP_B_OVER_G = 1.5
TARP_FLAG_FRACTION = 0.10

DEFAULT = {"condition": "ok", "roof_damage_score": None, "tarp_fraction": None}


def tarp_fraction(rgb: np.ndarray) -> float:
    """Fraction of roof pixels matching the vivid blue-tarp colour rule.
    rgb: (N, 3) uint8. Pure-numpy so it unit-tests without imagery."""
    if rgb.size == 0:
        return 0.0
    r, g, b = rgb[:, 0].astype(float), rgb[:, 1].astype(float), rgb[:, 2].astype(float)
    is_tarp = (b >= TARP_BLUE_MIN) & (b >= TARP_B_OVER_R * r) & (b >= TARP_B_OVER_G * g)
    return round(float(is_tarp.mean()), 4)


def flag(score: float | None, tarp_frac: float) -> str:
    """Condition flag from the damage score + tarp fraction. tarp wins (it's the
    most actionable), then the classifier's damaged/review bands."""
    if tarp_frac >= TARP_FLAG_FRACTION:
        return "tarp"
    if score is None:
        return "ok"
    if score >= DAMAGED_SCORE:
        return "damaged"
    if score >= REVIEW_SCORE:
        return "review"
    return "ok"


_model = None


def _load_model():
    global _model
    if _model is None:
        import torch
        from torchvision import models

        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        net.eval()
        _model = net
    return _model


def _fetch_hires(bbox: list[float]) -> tuple[np.ndarray, object] | None:
    """~0.3 m leaf-off tile for the AOI (higher res than detection imagery)."""
    import requests
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    sx, sy = maxx - minx, maxy - miny
    scale = min(1600 / (sx / CHIP_MPP), 1600 / (sy / CHIP_MPP), 1.0)
    w, h = int(sx / CHIP_MPP * scale), int(sy / CHIP_MPP * scale)
    if w < 32 or h < 32:
        return None
    url = os.environ.get(
        "MO_IMAGESERVER_URL",
        "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
        "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer",
    )
    resp = requests.get(
        f"{url}/exportImage",
        params={
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{w},{h}",
            "format": "png",
            "f": "image",
        },
        timeout=90,
    )
    resp.raise_for_status()
    from io import BytesIO

    from PIL import Image

    arr = np.asarray(Image.open(BytesIO(resp.content)).convert("RGB"))
    return arr, from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])


def assess_footprints(gdf, bbox: list[float], raster_paths=None):
    """Attach roof-condition columns to each footprint. Fetches a high-res
    leaf-off tile and runs the in-domain classifier + tarp colour rule per roof.
    Falls back to 'ok' defaults when imagery/model are unavailable (fake backend,
    outside Missouri, or a service outage) — condition is a bonus, never blocks."""
    gdf = gdf.copy()
    if gdf.empty:
        for k in DEFAULT:
            gdf[k] = []
        return gdf
    try:
        fetched = _fetch_hires(bbox)
        model = _load_model()
    except Exception as exc:  # noqa: BLE001 — condition is best-effort
        logger.warning("condition skipped (%s); defaulting to ok", exc)
        for k, v in DEFAULT.items():
            gdf[k] = [v] * len(gdf)
        return gdf
    if fetched is None:
        for k, v in DEFAULT.items():
            gdf[k] = [v] * len(gdf)
        return gdf

    import torch
    from PIL import Image
    from rasterio.transform import rowcol
    from torchvision import transforms

    arr, transform = fetched
    H, W, _ = arr.shape
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    geoms = gdf.to_crs("EPSG:3857").geometry
    scores, tarps = [], []
    for geom in geoms:
        minx, miny, maxx, maxy = geom.bounds
        r0, c0 = rowcol(transform, minx, maxy)
        r1, c1 = rowcol(transform, maxx, miny)
        r0, r1 = max(0, min(r0, r1)), min(H, max(r0, r1))
        c0, c1 = max(0, min(c0, c1)), min(W, max(c0, c1))
        if r1 - r0 < 8 or c1 - c0 < 8:
            scores.append(None)
            tarps.append(0.0)
            continue
        patch = arr[r0:r1, c0:c1]
        tarps.append(tarp_fraction(patch.reshape(-1, 3)))
        chip = Image.fromarray(patch).resize((CHIP_PX, CHIP_PX))
        with torch.no_grad():
            p = torch.softmax(model(norm(chip).unsqueeze(0)), 1)[0, 1].item()
        scores.append(round(p, 3))

    gdf["roof_damage_score"] = scores
    gdf["tarp_fraction"] = tarps
    gdf["condition"] = [flag(s, t) for s, t in zip(scores, tarps, strict=True)]
    return gdf
