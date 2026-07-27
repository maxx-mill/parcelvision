"""Chapter 5 — fine-tune a building detector on LEAF-OFF residential imagery.

The eval (worker/scripts/eval_detectors.py) exposed the core gap: the pretrained
Mask R-CNN (`building_footprints_usa`) was trained on leaf-ON summer NAIP, so on
Missouri's leaf-off orthoimagery it over-detects (residential F1 0.25). This
pipeline closes that gap the honest way — retrain on the imagery we actually use:

  1. prepare  — fetch leaf-off imagery + Overture footprints (weak but
                authoritative labels) for a training region distinct from the
                eval AOI, and export aligned image/label tiles.
  2. train    — fine-tune Mask R-CNN from COCO-pretrained weights on those tiles.
  3. evaluate — score the fine-tuned checkpoint vs the pretrained baseline on a
                held-out leaf-off AOI (per-structure precision/recall/F1).

CPU trains this slowly; a full run wants a GPU (see notebooks/train_building_model.ipynb
and the README). `--smoke` runs a few tiles / 2 epochs just to prove the pipeline
executes end to end. This is a distinct deliverable — not wired into the live app.

    docker compose run --rm --no-deps worker python scripts/finetune_buildings.py --smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.pipeline.fetch import _fetch_mo_leafoff  # noqa: E402

# Training region: a residential swath of St. Louis County, kept clear of the
# eval AOI (-90.3167..-90.3111) so we measure generalization, not memorization.
TRAIN_BBOX = [-90.3600, 38.6600, -90.3400, 38.6750]
TEST_BBOX = [-90.3167, 38.6465, -90.3111, 38.6501]  # the residential eval AOI
WORK = Path("/data/imagery/finetune")
PRETRAINED = "building_footprints_usa.pth"  # geoai baseline checkpoint


def _grid(bbox: list[float], nx: int, ny: int) -> list[list[float]]:
    """Split a bbox into an nx*ny grid of sub-bboxes."""
    minx, miny, maxx, maxy = bbox
    dx, dy = (maxx - minx) / nx, (maxy - miny) / ny
    return [
        [minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy]
        for i in range(nx)
        for j in range(ny)
    ]


def prepare(bbox: list[float], name: str, grid: tuple[int, int] = (3, 3)) -> Path:
    """Leaf-off imagery + Overture footprints -> geoai image/label tiles.

    The ArcGIS ImageServer 500s on very large exports, so we fetch the region as
    a grid of smaller leaf-off exports and merge each one's tiles (prefixed to
    avoid name collisions) into a single images/ + labels/ set."""
    import shutil

    import geoai
    import geopandas as gpd

    out = WORK / name
    tiles_dir = out / "tiles"
    images, labels = tiles_dir / "images", tiles_dir / "labels"
    if images.exists() and any(images.glob("*.tif")):
        print(f"  [{name}] tiles already prepared ({len(list(images.glob('*.tif')))})")
        return tiles_dir
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    vpath = out / "overture.geojson"
    if not vpath.exists():
        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(vpath))
    gdf = gpd.read_file(vpath)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf["class"] = 1  # single foreground class for the label rasterization
    lpath = out / "labels.geojson"
    gdf[["class", "geometry"]].to_file(lpath)
    print(f"  [{name}] {len(gdf)} Overture footprints as labels")

    total = 0
    for k, sub in enumerate(_grid(bbox, *grid)):
        cell = out / f"cell_{k}"
        try:
            raster = _fetch_mo_leafoff(sub, cell)
        except Exception as exc:
            print(f"    cell {k}: imagery export failed ({str(exc)[:50]}); skipping")
            continue
        if not raster:
            continue
        geoai.export_geotiff_tiles(
            in_raster=str(raster[0]),
            out_folder=str(cell / "tiles"),
            in_class_data=str(lpath),
            tile_size=512,
            stride=256,
            buffer_radius=0,
            skip_empty_tiles=True,  # residential AOIs have lots of empty yard/road
        )
        for img in (cell / "tiles" / "images").glob("*.tif"):
            lab = cell / "tiles" / "labels" / img.name
            if not lab.exists():
                continue
            shutil.move(str(img), str(images / f"c{k}_{img.name}"))
            shutil.move(str(lab), str(labels / f"c{k}_{img.name}"))
            total += 1
        shutil.rmtree(cell, ignore_errors=True)
    if total == 0:
        raise RuntimeError(f"[{name}] no tiles exported — check imagery/label coverage")
    print(f"  [{name}] exported {total} image/label tiles -> {tiles_dir}")
    return tiles_dir


def train(tiles_dir: Path, epochs: int, batch_size: int) -> Path:
    import geoai

    models_dir = WORK / "models"
    geoai.train_MaskRCNN_model(
        images_dir=str(tiles_dir / "images"),
        labels_dir=str(tiles_dir / "labels"),
        output_dir=str(models_dir),
        num_channels=3,
        pretrained=True,  # COCO-pretrained backbone -> fine-tune on our tiles
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=0.005,
        val_split=0.2,
    )
    ckpt = next(models_dir.glob("*.pth"), None)
    print(f"  trained checkpoint -> {ckpt}")
    return ckpt


def evaluate(model_path: Path, bbox: list[float]) -> None:
    """Before/after: fine-tuned vs pretrained on a held-out leaf-off AOI."""
    import geopandas as gpd
    from geoai import BuildingFootprintExtractor
    from worker.pipeline.postprocess import postprocess

    tiles = _fetch_mo_leafoff(bbox, WORK / "test")
    ref_path = WORK / "test" / "overture.geojson"
    if not ref_path.exists():
        import geoai

        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ref_path))
    ref = gpd.read_file(ref_path)
    ref = ref.to_crs(ref.estimate_utm_crs())

    def run(model):
        ex = BuildingFootprintExtractor(model_path=model) if model else None
        # fall back to default pretrained when model is None
        ex = ex or BuildingFootprintExtractor()
        gdf = ex.process_raster(
            str(tiles[0]), batch_size=2, confidence_threshold=0.5, min_object_area=50
        )
        clean = postprocess(gdf, bbox)
        return _score(clean, ref)

    print("\n== before/after on held-out leaf-off AOI ==")
    print("  pretrained :", run(None))
    print("  fine-tuned :", run(str(model_path)))


def _score(pred, ref) -> dict:
    if pred is None or pred.empty:
        return {"detections": 0, "precision": 0, "recall": 0, "f1": 0}
    pred = pred.to_crs(ref.crs)
    tp = 0
    used = set()
    for pg in pred.geometry:
        pg = pg if pg.is_valid else pg.buffer(0)
        for ri, rg in enumerate(ref.geometry):
            if ri in used:
                continue
            inter = pg.intersection(rg).area
            if inter and inter / (pg.area + rg.area - inter) >= 0.5:
                used.add(ri)
                tp += 1
                break
    p = tp / max(len(pred), 1)
    r = tp / max(len(ref), 1)
    return {
        "detections": len(pred),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / max(p + r, 1e-9), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the pipeline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    epochs = 2 if args.smoke else args.epochs

    WORK.mkdir(parents=True, exist_ok=True)
    print("STEP 1/3 prepare training data")
    # Smaller grid for the smoke run so it's fast; full run tiles more densely.
    tiles_dir = prepare(TRAIN_BBOX, "train", grid=(2, 2) if args.smoke else (3, 3))
    if args.smoke:
        _trim_to(tiles_dir, keep=6)  # keep the run fast on CPU

    print(f"STEP 2/3 train (epochs={epochs}, pretrained=COCO)")
    ckpt = train(tiles_dir, epochs, args.batch_size)

    print("STEP 3/3 evaluate vs pretrained baseline")
    if ckpt:
        evaluate(ckpt, TEST_BBOX)


def _trim_to(tiles_dir: Path, keep: int) -> None:
    imgs = sorted((tiles_dir / "images").glob("*.tif"))
    for p in imgs[keep:]:
        p.unlink(missing_ok=True)
        (tiles_dir / "labels" / p.name).unlink(missing_ok=True)
    print(f"  [smoke] trimmed to {min(keep, len(imgs))} tiles")


if __name__ == "__main__":
    main()
