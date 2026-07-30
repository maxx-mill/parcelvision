import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("affine")

from affine import Affine  # noqa: E402
from shapely.geometry import box  # noqa: E402

from worker.pipeline.roof_chip import (  # noqa: E402
    CHIP_PX,
    MEAN_RGB,
    TILE_M,
    masked_roof_pixels,
    roof_tiles,
)


def _scene(mpp=0.15, size=800):
    """A synthetic north-up 3857 scene: uniform red, transform origin at (0, 0)."""
    arr = np.zeros((size, size, 3), dtype="uint8")
    arr[..., 0] = 200  # red everywhere so we can see masking (bg -> MEAN_RGB)
    # from_bounds-style north-up affine: col->+x, row->-y, origin top-left (0, 0)
    transform = Affine(mpp, 0, 0, 0, -mpp, 0)
    return arr, transform


def test_small_roof_single_tile_masked():
    arr, transform = _scene()
    # a ~10 m house well inside one 19.2 m tile, centred at (100, -100)
    geom = box(95, -105, 105, -95)
    tiles = roof_tiles(arr, transform, geom)
    assert len(tiles) == 1
    t = tiles[0]
    assert t.shape == (CHIP_PX, CHIP_PX, 3)
    # centre is roof (red), corners are masked to the neutral fill
    assert tuple(t[CHIP_PX // 2, CHIP_PX // 2]) == (200, 0, 0)
    assert tuple(t[0, 0]) == MEAN_RGB


def test_large_roof_tiles_into_grid():
    arr, transform = _scene(size=1200)
    # a ~40 m x 40 m warehouse -> needs a >=3x3 grid of 19.2 m tiles
    span = 40
    geom = box(100, -(100 + span), 100 + span, -100)
    tiles = roof_tiles(arr, transform, geom)
    assert len(tiles) >= 4  # tiled, not one squashed chip
    assert all(t.shape == (CHIP_PX, CHIP_PX, 3) for t in tiles)


def test_fixed_gsd_constant_across_sizes():
    # tile ground extent is constant regardless of building size -> constant GSD
    assert TILE_M == CHIP_PX * 0.15


def test_masked_roof_pixels_drops_background():
    arr, transform = _scene()
    geom = box(95, -105, 105, -95)
    tiles = roof_tiles(arr, transform, geom)
    px = masked_roof_pixels(tiles)
    assert px.shape[0] > 0
    # every returned pixel is roof (red), never the neutral background fill
    assert np.all(px[:, 0] == 200)
    assert not np.any(np.all(px == np.array(MEAN_RGB), axis=1))
