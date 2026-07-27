"""Stage 1: fetch aerial imagery for the AOI, from the best available source.

Source chain (first that yields imagery wins):

  1. mo_leafoff — Missouri statewide 6-inch **leaf-off** orthoimagery (MSDIS
     ArcGIS ImageServer). Best source for our St. Louis County real-estate use
     case: 4x NAIP resolution and flown before spring green-up, so tree canopy
     doesn't occlude roofs. We request it at NAIP-equivalent ground resolution
     (~0.6 m) so it matches what the detectors were trained on — native 0.15 m
     would make buildings 4x larger in-frame and blow past the model's window.
     Missouri only; returns blank outside the state, which we detect and skip.
  2. naip — USDA NAIP via Microsoft Planetary Computer. Nationwide, model-native,
     but leaf-ON (it's an agriculture program, flown in summer) and the provider
     has recurring multi-hour outages. Streamed as windowed COG range reads.
  3. esri — Esri World Imagery georeferenced export. Last-resort so the app keeps
     working when the above are down; mixed date/resolution, RGB only.

IMAGERY_SOURCE selects the primary (default mo_leafoff); the rest act as
fallbacks. IMAGERY_FALLBACK=0 disables falling back. Output is always a small
local GeoTIFF the segmentation backends can open.
"""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SEARCH_ATTEMPTS = 3
RETRY_DELAY_S = 15

# Missouri 6-inch statewide leaf-off orthoimagery (2023–2024), free ArcGIS
# ImageServer. Native 0.15 m / UTM 15N; requested at NAIP-equivalent res below.
MO_IMAGESERVER_URL = os.environ.get(
    "MO_IMAGESERVER_URL",
    "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
    "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer",
)
ESRI_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/" "World_Imagery/MapServer/export"
)
TARGET_MPP = 0.6  # meters/pixel — match the detectors' NAIP training resolution
MAX_PX = 4096  # ArcGIS export size cap

IMAGERY_SOURCE = os.environ.get("IMAGERY_SOURCE", "mo_leafoff")
FALLBACK_ENABLED = os.environ.get("IMAGERY_FALLBACK", "1") != "0"


def fetch_imagery(bbox: list[float], workdir: Path, year: int | None = None) -> list[Path]:
    workdir.mkdir(parents=True, exist_ok=True)

    def naip(b, wd):
        return _fetch_naip(b, wd, year)

    chain: list[tuple[str, Callable]] = [
        ("mo_leafoff", _fetch_mo_leafoff),
        ("naip", naip),
        ("esri", _fetch_esri),
    ]
    # Put the configured primary first; keep the rest as ordered fallbacks.
    chain.sort(key=lambda kv: kv[0] != IMAGERY_SOURCE)
    if not FALLBACK_ENABLED:
        chain = chain[:1]

    errors = []
    for name, fn in chain:
        try:
            tiles = fn(bbox, workdir)
            if tiles:
                logger.info("imagery source '%s' provided %d tile(s)", name, len(tiles))
                return tiles
            logger.info("imagery source '%s' has no coverage here; trying next", name)
        except Exception as exc:  # noqa: BLE001 — any source failure falls through
            errors.append(f"{name}: {exc}")
            logger.warning("imagery source '%s' failed (%s); trying next", name, exc)
    raise RuntimeError("no imagery source succeeded — " + "; ".join(errors))


# --------------------------------------------------------------------------- #
# ArcGIS image export (MO leaf-off + Esri) — bbox -> georeferenced GeoTIFF
# --------------------------------------------------------------------------- #
def _export_arcgis(endpoint: str, bbox: list[float], out: Path, image_op: str) -> Path | None:
    """Export an AOI from an ArcGIS ImageServer (exportImage) or MapServer
    (export) as a Web-Mercator GeoTIFF at TARGET_MPP. Returns None when the
    response is blank (e.g. AOI outside the service's coverage)."""
    import numpy as np
    import rasterio
    import requests
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    out.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    width = min(int(round((maxx - minx) / TARGET_MPP)), MAX_PX)
    height = min(int(round((maxy - miny) / TARGET_MPP)), MAX_PX)
    if width < 16 or height < 16:
        raise RuntimeError(f"AOI too small to export ({width}x{height}px)")

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width},{height}",
        "format": "tiff",
        "f": "image",
    }
    resp = requests.get(f"{endpoint}/{image_op}", params=params, timeout=90)
    resp.raise_for_status()
    if not resp.content or resp.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"export returned no image: {resp.text[:200]}")

    with MemoryFile(resp.content) as mem, mem.open() as src:
        data = src.read(indexes=[1, 2, 3]) if src.count >= 3 else src.read()
    # Blank/NoData tile (outside coverage) -> let the caller fall through.
    if int(data.max()) == 0 or float(data.std()) < 1.0:
        return None

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": data.shape[0],
        "height": height,
        "width": width,
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data.astype(np.uint8))
    logger.info("exported %dx%d px from %s -> %s", width, height, endpoint.split("/")[-2], out.name)
    return out


def _fetch_mo_leafoff(bbox: list[float], workdir: Path) -> list[Path] | None:
    out = _export_arcgis(MO_IMAGESERVER_URL, bbox, workdir / "mo_leafoff.tif", "exportImage")
    return [out] if out else None


def _fetch_esri(bbox: list[float], workdir: Path) -> list[Path] | None:
    out = _export_arcgis(ESRI_EXPORT_URL, bbox, workdir / "esri_worldimagery.tif", "export")
    return [out] if out else None


# --------------------------------------------------------------------------- #
# NAIP via Planetary Computer — windowed COG range reads
# --------------------------------------------------------------------------- #
def _fetch_naip(bbox: list[float], workdir: Path, year: int | None) -> list[Path]:
    items = _search_naip(bbox, year)
    clipped: list[Path] = []
    for item in items:
        out = workdir / f"clip_{item.id}.tif"
        if _read_window(item, bbox, out):
            clipped.append(out)
    if not clipped:
        raise RuntimeError(f"NAIP items matched bbox {bbox} but no pixels intersected it")
    return clipped


def _search_naip(bbox: list[float], year: int | None):
    import planetary_computer
    from pystac_client import Client

    dt = f"{year}-01-01/{year}-12-31" if year else None
    last_exc: Exception | None = None
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        try:
            catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
            items = list(
                catalog.search(collections=["naip"], bbox=tuple(bbox), datetime=dt).items()
            )
            if not items:
                raise RuntimeError(f"no NAIP imagery found for bbox {bbox} (year={year})")
            kept = _newest_year_items(items)
            logger.info(
                "STAC: %d NAIP item(s), keeping %d from %s",
                len(items),
                len(kept),
                kept[0].datetime.year,
            )
            return kept
        except RuntimeError:
            raise  # "no imagery" is a real answer, not a transient fault
        except Exception as exc:
            last_exc = exc
            if attempt < SEARCH_ATTEMPTS:
                logger.warning(
                    "STAC search attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    SEARCH_ATTEMPTS,
                    exc,
                    RETRY_DELAY_S * attempt,
                )
                time.sleep(RETRY_DELAY_S * attempt)
    raise RuntimeError(f"STAC search failed after {SEARCH_ATTEMPTS} attempts") from last_exc


def _newest_year_items(items: list) -> list:
    """A small AOI matches the same quarter-quad across several NAIP vintages;
    detecting on all of them would emit near-duplicate footprints per building.
    Keep only the most recent acquisition year."""
    newest = max(i.datetime.year for i in items)
    return [i for i in items if i.datetime.year == newest]


def _read_window(item, bbox: list[float], out: Path) -> bool:
    """Range-read the bbox window from the item's COG into a local GeoTIFF.
    Returns False when the item doesn't actually overlap the bbox."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds, intersection

    href = item.assets["image"].href  # pre-signed by the catalog modifier
    with rasterio.open(href) as src:
        wb = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*wb, transform=src.transform)
        window = intersection(window, Window(0, 0, src.width, src.height))
        window = window.round_offsets().round_lengths()
        if window.width < 1 or window.height < 1:
            return False
        data = src.read(window=window)
        profile = src.profile
        profile.update(
            driver="GTiff",
            width=window.width,
            height=window.height,
            transform=src.window_transform(window),
            compress="deflate",
        )
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
    logger.info("streamed %s window %dx%d px -> %s", item.id, window.width, window.height, out.name)
    return True
