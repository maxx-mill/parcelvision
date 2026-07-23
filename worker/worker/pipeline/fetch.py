"""Stage 1: stream NAIP imagery for the AOI from Microsoft Planetary Computer.

NAIP is served as Cloud-Optimized GeoTIFFs, so we never download whole
quarter-quads (~480 MB each): a STAC search finds the newest-vintage items,
then rasterio reads just the pixel window covering the bbox via HTTP range
requests — a few MB, seconds not minutes. Each window is materialized as a
small local GeoTIFF because the segmentation backends take file paths.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Planetary Computer's STAC API sheds load with transient gateway timeouts;
# a short retry usually clears it without failing the whole job.
SEARCH_ATTEMPTS = 3
RETRY_DELAY_S = 15


def fetch_imagery(bbox: list[float], workdir: Path, year: int | None = None) -> list[Path]:
    items = _search_naip(bbox, year)
    workdir.mkdir(parents=True, exist_ok=True)
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
                len(items), len(kept), kept[0].datetime.year,
            )
            return kept
        except RuntimeError:
            raise  # "no imagery" is a real answer, not a transient fault
        except Exception as exc:
            last_exc = exc
            if attempt < SEARCH_ATTEMPTS:
                logger.warning(
                    "STAC search attempt %d/%d failed (%s); retrying in %ds",
                    attempt, SEARCH_ATTEMPTS, exc, RETRY_DELAY_S * attempt,
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
