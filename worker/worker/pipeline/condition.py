"""Stage 3.5: per-structure roof condition (Chapter 6 — damage v5, in-domain).

Off-the-shelf models (CLIP, xView2 SegFormer, RescueNet YOLO) all failed to
transfer to our imagery, so we train an IN-DOMAIN ResNet18 on leaf-off roof
chips. v5 keeps that idea but adopts a research-backed recipe (see
worker/scripts/train_condition_v5.py):

  * Fixed-GSD, polygon-masked, multi-scale chips (worker.pipeline.roof_chip),
    identical at train + inference. v4 cropped each footprint's bbox and resized
    to 128 px, so texture scale varied with building size and the model
    shortcut-learned "big squashed roof == damaged" — the institutional
    false-positive bug. Now every tile covers a constant 19.2 m of ground; big
    roofs are tiled and their per-tile scores averaged.
  * Focal loss + strong augmentation (scale-jitter, rotation, RandomErasing).
  * D4 test-time augmentation (8 orientations; top-down roofs are rotation-
    arbitrary) + temperature-scaled, calibrated P(damaged).
  * Active learning: v5's first 120 labels were all north-city, so the model was
    out-of-distribution (~0.5) on suburban roofs. Uncertainty + hard-negative
    (big-roof) mining surfaced 60 intact suburban/commercial roofs to label,
    giving 180 labels (45 damaged / 135 intact).

Honest held-out numbers (grouped 4-fold CV + Palm St vs demo): building-level
best-separating balanced accuracy ~0.81; recalibrating from v4 to v5 cuts the
demo's institutional false-"damaged" from 22 -> 3 (of 64). The residual "review"
band is genuine model uncertainty, not false confidence — a hard problem on a
single leaf-off image with 45 damaged examples from one neighbourhood.

Per footprint we record roof_damage_score (mean per-tile P(damaged)) and
tarp_fraction (an unambiguous complementary colour signal over masked roof
pixels), and a condition flag: tarp > damaged > review > ok. The chip crop
needs higher resolution than the detection imagery, so this stage fetches its
own ~0.15 m leaf-off tile.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "roof_condition_resnet18.pt"
CHIP_PX = 128
CHIP_MPP = 0.15  # native leaf-off res; matches the v4 classifier's training chips

# Condition thresholds on the calibrated (temperature-scaled, D4-TTA) P(damaged).
# Chosen from the held-out score distributions — intact demo p50=0.38, damaged
# Palm St p50=0.64, best separating cut ~0.57 (balanced acc 0.81). DAMAGED is set
# high for precision (only 5% of intact demo reaches it); REVIEW catches the rest.
DAMAGED_SCORE = 0.60
REVIEW_SCORE = 0.50

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
_temperature = 1.0
CONDITION_TTA = os.environ.get("CONDITION_TTA", "1") != "0"


def _load_model():
    """Load the v5 checkpoint {state_dict, temperature, ...}. Backward compatible
    with a bare state_dict (older v4 file) -> temperature 1.0."""
    global _model, _temperature
    if _model is None:
        import torch
        from torchvision import models

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            net.load_state_dict(ckpt["state_dict"])
            _temperature = float(ckpt.get("temperature", 1.0))
        else:
            net.load_state_dict(ckpt)
            _temperature = 1.0
        net.eval()
        _model = net
    return _model, _temperature


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
        model, temp = _load_model()
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
    import torch.nn.functional as F
    from torchvision import transforms

    from worker.pipeline.roof_chip import masked_roof_pixels, roof_tiles

    arr, transform = fetched
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    def tile_p(chips: list) -> np.ndarray:
        """Per-tile P(damaged): D4 TTA (8 orientations) + temperature scaling."""
        X = torch.stack([norm(c) for c in chips])
        acc = torch.zeros(len(X), 2)
        with torch.no_grad():
            if CONDITION_TTA:
                for k in range(4):
                    rot = torch.rot90(X, k, dims=[2, 3])
                    for do_flip in (False, True):
                        v = torch.flip(rot, dims=[3]) if do_flip else rot
                        acc += F.softmax(model(v) / temp, 1)
                acc /= 8.0
            else:
                acc = F.softmax(model(X) / temp, 1)
        return acc[:, 1].numpy()

    from PIL import Image

    geoms = gdf.to_crs("EPSG:3857").geometry
    scores, tarps = [], []
    for geom in geoms:
        tiles = roof_tiles(arr, transform, geom)  # fixed-GSD, polygon-masked
        if not tiles:
            scores.append(None)
            tarps.append(0.0)
            continue
        roof_px = masked_roof_pixels(tiles)
        tarps.append(tarp_fraction(roof_px))
        p = tile_p([Image.fromarray(t) for t in tiles])
        scores.append(round(float(p.mean()), 3))  # aggregate tiles -> building score

    gdf["roof_damage_score"] = scores
    gdf["tarp_fraction"] = tarps
    gdf["condition"] = [flag(s, t) for s, t in zip(scores, tarps, strict=True)]
    return gdf
