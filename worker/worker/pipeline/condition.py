"""Stage 3.5: per-structure roof condition INDICATORS (Chapter 6).

Important framing — these are interpretable heuristics, not a validated damage
classifier. There is no free pretrained aerial roof-damage model and no
ground-truth condition labels for our area, so we compute defensible,
industry-grounded signals from each footprint's roof pixels rather than claim a
trained "damage AI". A supervised model (xBD/xView2) is the documented upgrade
path once labels + a GPU are available.

Signals per footprint:
  * tarp_fraction — share of roof pixels in the blue-tarp colour range. Blue
    tarps are a standard FEMA/insurer post-storm damage proxy — unambiguous.
  * heterogeneity — normalized spread of roof brightness; a clean uniform roof
    scores low, while missing shingles / patches / debris / staining raise it.
  * condition — a flag derived from the two: 'tarp' > 'review' > 'ok'.

The colour/threshold math is pure-numpy so it unit-tests in the slim image;
raster sampling lives in the worker stage that calls this (pipeline wiring).
"""

import numpy as np

# Blue-tarp colour rule (0-255 RGB). Real emergency tarps are vividly blue, so
# require blue to be high AND clearly dominant over red/green. Calibrated on a
# leaf-off residential AOI: a looser rule caught winter's blue-grey cast on
# ordinary roofs (tarp_fraction up to 0.39 on undamaged homes), so these are
# deliberately strict — normal roofs should score ~0.
TARP_BLUE_MIN = 120
TARP_B_OVER_R = 1.6  # blue at least 60% above red
TARP_B_OVER_G = 1.5  # and 50% above green

# condition thresholds — set to flag OUTLIERS, not the median roof. On the demo
# AOI heterogeneity ran p50 0.35 / p90 0.50, so 'review' fires on the top ~10%;
# 'tarp' needs meaningful coverage, not a few stray blue pixels. Heuristic and
# tunable — NOT a validated damage classifier (see module docstring).
TARP_FLAG_FRACTION = 0.10  # >=10% vivid-blue pixels -> 'tarp'
HETEROGENEITY_REVIEW = 0.50  # normalized std above this -> 'review'


def tarp_fraction(rgb: np.ndarray) -> float:
    """Fraction of valid roof pixels matching the blue-tarp colour rule.
    rgb: (N, 3) uint8 array of roof pixels (RGB)."""
    if rgb.size == 0:
        return 0.0
    r, g, b = rgb[:, 0].astype(float), rgb[:, 1].astype(float), rgb[:, 2].astype(float)
    is_tarp = (b >= TARP_BLUE_MIN) & (b >= TARP_B_OVER_R * r) & (b >= TARP_B_OVER_G * g)
    return round(float(is_tarp.mean()), 4)


def heterogeneity(rgb: np.ndarray) -> float:
    """Normalized brightness spread of the roof (0..~1). Clean roofs are
    uniform (low); damage/patches/debris raise it. std of luminance / 128."""
    if rgb.size == 0:
        return 0.0
    lum = rgb.astype(float) @ np.array([0.299, 0.587, 0.114])
    return round(float(min(lum.std() / 128.0, 1.0)), 4)


def classify(tarp_frac: float, hetero: float) -> str:
    """Roof-condition flag from the indicators. Order matters: a tarp is the
    strongest damage signal, then general irregularity, else ok."""
    if tarp_frac >= TARP_FLAG_FRACTION:
        return "tarp"
    if hetero >= HETEROGENEITY_REVIEW:
        return "review"
    return "ok"


def assess_pixels(rgb: np.ndarray) -> dict:
    """Compute all indicators for one footprint's roof pixels."""
    tf = tarp_fraction(rgb)
    het = heterogeneity(rgb)
    return {"tarp_fraction": tf, "heterogeneity": het, "condition": classify(tf, het)}


DEFAULT = {"tarp_fraction": 0.0, "heterogeneity": 0.0, "condition": "ok"}


def assess_footprints(gdf, raster_paths):
    """Attach condition indicators to each footprint by sampling its roof pixels
    from the source imagery. gdf is EPSG:4326; rasters may be in any CRS. When
    there's no imagery (e.g. the fake backend) every footprint defaults to 'ok'."""
    gdf = gdf.copy()
    if gdf.empty or not raster_paths:
        for k, v in DEFAULT.items():
            gdf[k] = [v] * len(gdf)
        return gdf

    # rasterio only needed on the sampling path — keeps the slim/fake path dep-free.
    import rasterio
    from rasterio.mask import mask as rio_mask

    srcs = [rasterio.open(p) for p in raster_paths]
    try:
        reproj = [gdf.to_crs(s.crs).geometry for s in srcs]  # geoms per raster CRS
        records = []
        for idx in range(len(gdf)):
            rec = dict(DEFAULT)
            for si, s in enumerate(srcs):
                try:
                    arr, _ = rio_mask(s, [reproj[si].iloc[idx]], crop=True, filled=False)
                except (ValueError, IndexError):
                    continue  # footprint doesn't overlap this raster
                if arr.shape[0] < 3:
                    continue
                valid = ~np.ma.getmaskarray(arr[0])
                rgb = np.stack([arr[0].data, arr[1].data, arr[2].data], axis=-1)[valid]
                rgb = rgb[rgb.sum(axis=1) > 0]  # drop nodata/black
                if rgb.size:
                    rec = assess_pixels(rgb.astype("uint8"))
                    break
            records.append(rec)
    finally:
        for s in srcs:
            s.close()

    for key in DEFAULT:
        gdf[key] = [r[key] for r in records]
    return gdf
