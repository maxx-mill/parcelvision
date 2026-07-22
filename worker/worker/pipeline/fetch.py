"""Stage 1: fetch NAIP imagery for the AOI from Microsoft Planetary Computer.

Tiles are clipped to the bbox before inference — NAIP quarter-quads are
~7000×7000 px and running the model on a full tile for a 1 km² AOI would waste
minutes of CPU. geoai handles STAC search, signing, and CRS-aware clipping.
"""

from pathlib import Path


def fetch_imagery(bbox: list[float], workdir: Path, year: int | None = None) -> list[Path]:
    import geoai  # heavy import (geoai -> rioxarray/planetary_computer); worker-full image only

    tiles = geoai.download_naip(
        bbox=tuple(bbox), output_dir=str(workdir / "tiles"), year=year, max_items=4
    )
    if not tiles:
        raise RuntimeError(f"no NAIP imagery found for bbox {bbox}")

    clipped: list[Path] = []
    for tile in tiles:
        tile = Path(tile)
        out = workdir / f"clip_{tile.name}"
        geoai.clip_raster_by_bbox(str(tile), str(out), bbox, bbox_crs="EPSG:4326")
        clipped.append(out)
    return clipped
