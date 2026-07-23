"""Stage 1: fetch NAIP imagery for the AOI from Microsoft Planetary Computer.

Tiles are clipped to the bbox before inference — NAIP quarter-quads are
~7000×7000 px and running the model on a full tile for a 1 km² AOI would waste
minutes of CPU. geoai handles STAC search, signing, and CRS-aware clipping.
"""

import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# NAIP filenames end in _YYYYMMDD.tif (e.g. m_3809022_sw_15_060_20220618.tif)
_ACQ_DATE_RE = re.compile(r"_(\d{8})\.tif$")

# Planetary Computer's STAC API sheds load with transient gateway timeouts;
# a short retry usually clears it without failing the whole job.
FETCH_ATTEMPTS = 3
RETRY_DELAY_S = 15


def fetch_imagery(bbox: list[float], workdir: Path, year: int | None = None) -> list[Path]:
    import geoai  # heavy import (geoai -> rioxarray/planetary_computer); worker-full image only

    tiles: list[str] = []
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            tiles = geoai.download_naip(
                bbox=tuple(bbox), output_dir=str(workdir / "tiles"), year=year, max_items=4
            )
            break
        except Exception as exc:
            if attempt == FETCH_ATTEMPTS:
                raise
            logger.warning(
                "NAIP fetch attempt %d/%d failed (%s); retrying in %ds",
                attempt, FETCH_ATTEMPTS, exc, RETRY_DELAY_S,
            )
            time.sleep(RETRY_DELAY_S * attempt)
    if not tiles:
        raise RuntimeError(f"no NAIP imagery found for bbox {bbox}")
    tiles = _newest_year_only(tiles)

    clipped: list[Path] = []
    for tile in tiles:
        tile = Path(tile)
        out = workdir / f"clip_{tile.name}"
        geoai.clip_raster_by_bbox(str(tile), str(out), bbox, bbox_crs="EPSG:4326")
        clipped.append(out)
    return clipped


def _newest_year_only(tiles: list[str]) -> list[str]:
    """A small AOI matches the same quarter-quad across several NAIP years;
    running inference on all of them would emit near-duplicate footprints per
    building. Keep only the most recent acquisition year."""
    dated = [(m.group(1)[:4] if (m := _ACQ_DATE_RE.search(Path(t).name)) else "", t) for t in tiles]
    years = {d for d, _ in dated if d}
    if len(years) <= 1:
        return tiles
    newest = max(years)
    kept = [t for d, t in dated if d == newest]
    logger.info("keeping %d/%d NAIP tile(s) from %s", len(kept), len(tiles), newest)
    return kept
