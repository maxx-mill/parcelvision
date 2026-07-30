"""Fixed-GSD, polygon-masked, multi-scale roof chipping.

Shared by the roof-condition classifier's TRAINING and its INFERENCE so the two
pixel pipelines can never drift (a classic train/serve-skew bug source).

Why fixed ground-sampling distance (GSD)
    A CNN keys on texture at a *pixel* scale. v4 cropped each footprint's bbox
    and resized it to a fixed 128 px, so a 10 m house and a 60 m warehouse were
    squashed by very different factors — the model then shortcut on "how
    compressed the texture looks", which is why it over-flagged big flat /
    institutional roofs. Here every emitted tile always covers
    ``CHIP_PX * CHIP_MPP`` metres of ground (19.2 m at 0.15 m/px), so roof
    texture sits at a constant real-world scale for every building, large or
    small. (Refs: shortcut-learning + object-zoomed-training literature.)

Why polygon masking
    Zeroing pixels outside the footprint strips the street / lawn / parking-lot
    background the classifier could otherwise latch onto — object-zoom pushes it
    toward shape/texture of the roof itself rather than its surroundings.

Why tiling
    A roof larger than one 19.2 m tile is split into several fixed-scale tiles;
    the caller aggregates the per-tile scores (mean). A big intact roof is then
    "many intact tiles", not "one weird squashed chip" — directly attacking the
    v4 institutional confound.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

CHIP_PX = 128
CHIP_MPP = 0.15  # metres/pixel of every emitted tile (constant by construction)
MARGIN_M = 1.5  # a little roof-edge context beyond the footprint (EPSG:3857 metres)
MIN_TILE_COVER = 0.08  # drop tiles whose masked roof area is under this fraction
MEAN_RGB = (124, 116, 104)  # neutral fill for masked-out pixels (~ImageNet mean)

TILE_M = CHIP_PX * CHIP_MPP  # ground extent of one tile, metres


def _world_to_px(transform, x, y):
    """(x, y) EPSG:3857 -> fractional (col, row) in the source array. Assumes a
    north-up affine (rasterio ``from_bounds`` output): b == d == 0."""
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return col, row


def _rings(geom):
    """Yield (exterior_coords, [hole_coords, ...]) for Polygon / MultiPolygon."""
    if geom.geom_type == "Polygon":
        yield list(geom.exterior.coords), [list(r.coords) for r in geom.interiors]
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            yield list(g.exterior.coords), [list(r.coords) for r in g.interiors]


def _tile_mask(geom, win_minx, win_maxy):
    """Boolean CHIP_PX x CHIP_PX mask: True where the tile pixel is inside the
    roof polygon. Tile top-left is (win_minx, win_maxy) at CHIP_MPP m/px."""
    img = Image.new("L", (CHIP_PX, CHIP_PX), 0)
    draw = ImageDraw.Draw(img)

    def to_px(coords):
        return [((x - win_minx) / CHIP_MPP, (win_maxy - y) / CHIP_MPP) for x, y in coords]

    for ext, holes in _rings(geom):
        if len(ext) >= 3:
            draw.polygon(to_px(ext), fill=1)
        for hole in holes:
            if len(hole) >= 3:
                draw.polygon(to_px(hole), fill=0)
    return np.asarray(img, dtype=bool)


def roof_tiles(arr: np.ndarray, transform, geom, *, mask: bool = True) -> list[np.ndarray]:
    """Fixed-GSD, polygon-masked tiles (uint8 CHIP_PX x CHIP_PX x 3) for one roof.

    ``arr``       source RGB tile (H, W, 3) uint8.
    ``transform`` its north-up affine (rasterio ``from_bounds``), EPSG:3857.
    ``geom``      building polygon in EPSG:3857 (shapely Polygon/MultiPolygon).

    Every tile covers exactly TILE_M metres, so after resampling to CHIP_PX the
    effective resolution is CHIP_MPP for every building. Roofs bigger than one
    tile are split into a grid; corner tiles that barely touch the roof are
    dropped. Always returns at least one tile so every building gets a score.
    """
    if geom is None or geom.is_empty:
        return []
    H, W, _ = arr.shape
    minx, miny, maxx, maxy = geom.buffer(MARGIN_M).bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    ncols = max(1, int(np.ceil((maxx - minx) / TILE_M)))
    nrows = max(1, int(np.ceil((maxy - miny) / TILE_M)))
    # centre the grid on the footprint so coverage is symmetric
    grid_minx = cx - ncols * TILE_M / 2.0
    grid_maxy = cy + nrows * TILE_M / 2.0

    tiles: list[np.ndarray] = []
    covers: list[float] = []
    for i in range(nrows):
        for j in range(ncols):
            win_minx = grid_minx + j * TILE_M
            win_maxy = grid_maxy - i * TILE_M
            win_maxx = win_minx + TILE_M
            win_miny = win_maxy - TILE_M
            # source pixel window (fractional), then clamp to the array
            c0, r0 = _world_to_px(transform, win_minx, win_maxy)
            c1, r1 = _world_to_px(transform, win_maxx, win_miny)
            c0i, c1i = int(np.floor(min(c0, c1))), int(np.ceil(max(c0, c1)))
            r0i, r1i = int(np.floor(min(r0, r1))), int(np.ceil(max(r0, r1)))
            c0i, c1i = max(0, c0i), min(W, c1i)
            r0i, r1i = max(0, r0i), min(H, r1i)
            if c1i - c0i < 4 or r1i - r0i < 4:
                continue
            # every window spans exactly TILE_M ground -> resize to CHIP_PX gives
            # a constant effective GSD regardless of the source tile's own mpp
            patch = Image.fromarray(arr[r0i:r1i, c0i:c1i]).resize((CHIP_PX, CHIP_PX))
            chip = np.asarray(patch, dtype="uint8").copy()
            if mask:
                m = _tile_mask(geom, win_minx, win_maxy)
                cover = float(m.mean())
                chip[~m] = MEAN_RGB
            else:
                cover = 1.0
            tiles.append(chip)
            covers.append(cover)

    if not tiles:
        return []
    keep = [t for t, cov in zip(tiles, covers, strict=True) if cov >= MIN_TILE_COVER]
    if keep:
        return keep
    # nothing cleared the coverage bar (tiny/thin building) -> keep the best one
    return [tiles[int(np.argmax(covers))]]


def masked_roof_pixels(tiles: list[np.ndarray]) -> np.ndarray:
    """(N, 3) uint8 of the non-background pixels across tiles, for the tarp rule.
    Background was set to MEAN_RGB, so drop exact matches of it."""
    if not tiles:
        return np.empty((0, 3), dtype="uint8")
    flat = np.concatenate([t.reshape(-1, 3) for t in tiles], axis=0)
    bg = np.all(flat == np.array(MEAN_RGB, dtype="uint8"), axis=1)
    return flat[~bg]
