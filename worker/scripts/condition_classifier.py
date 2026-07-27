"""In-domain roof-condition classifier (damage v3).

Off-the-shelf models (CLIP, xView2 SegFormer, RescueNet YOLO) all failed to
transfer to our leaf-off imagery. This trains a small classifier ON our own
imagery instead. Clean per-building vacancy labels weren't reachable from this
environment, so labels are GEOGRAPHIC weak supervision: roof chips from
known-derelict north St. Louis City areas = 'damaged', intact suburban areas =
'intact'. The honest test is held-out per-building validation on 2020 Palm St
(a mixed block with both damaged and intact roofs) — if it separates those, it
learned roof condition, not just neighbourhood style.

    docker run ... python scripts/condition_classifier.py
"""

import sys
import time
import warnings
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geoai  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.transform import from_bounds, rowcol  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from torchvision import models, transforms  # noqa: E402

OUT = Path("/data/imagery/condition_clf")
OUT.mkdir(parents=True, exist_ok=True)
CHIP = 128

# Training AOIs (weak geographic labels). Palm St + demo are held out for eval.
TRAIN = {
    "damaged": [
        [-90.2260, 38.6560, -90.2210, 38.6590],
        [-90.2150, 38.6620, -90.2100, 38.6650],
        [-90.2350, 38.6520, -90.2300, 38.6550],
        [-90.2080, 38.6600, -90.2030, 38.6630],
    ],
    "intact": [
        [-90.4450, 38.5900, -90.4400, 38.5930],
        [-90.3550, 38.5950, -90.3500, 38.5980],
        [-90.4000, 38.6100, -90.3950, 38.6130],
        [-90.4600, 38.6000, -90.4550, 38.6030],
    ],
}
EVAL = {
    "palm_damaged": [-90.2035, 38.6548, -90.1999, 38.6568],
    "demo_intact": [-90.3150, 38.6478, -90.3120, 38.6498],
}


def hires(bbox, mpp=0.3, maxpx=1600):
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    sx, sy = maxx - minx, maxy - miny
    scale = min(maxpx / (sx / mpp), maxpx / (sy / mpp), 1.0)
    w, h = int(sx / mpp * scale), int(sy / mpp * scale)
    url = (
        "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
        "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer/exportImage"
    )
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{w},{h}",
        "format": "png",
        "f": "image",
    }
    for a in range(4):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            arr = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
            return arr, from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
        except Exception:
            if a == 3:
                raise
            time.sleep(8 * (a + 1))


def roof_chips(bbox):
    """Crop each Overture footprint's roof to a fixed-size chip."""
    ov = OUT / f"ov_{bbox[0]:.4f}_{bbox[1]:.4f}.geojson"
    if not ov.exists():
        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
    try:
        gdf = gpd.read_file(ov).to_crs("EPSG:3857")
    except Exception:
        return []
    arr, tf = hires(bbox)
    H, W, _ = arr.shape
    chips = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        r0, c0 = rowcol(tf, minx, maxy)
        r1, c1 = rowcol(tf, maxx, miny)
        r0, r1 = max(0, min(r0, r1)), min(H, max(r0, r1))
        c0, c1 = max(0, min(c0, c1)), min(W, max(c0, c1))
        if r1 - r0 < 16 or c1 - c0 < 16:
            continue
        chips.append(Image.fromarray(arr[r0:r1, c0:c1]).resize((CHIP, CHIP)))
    return chips


NORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


def build_dataset():
    X, y = [], []
    for label, aois in TRAIN.items():
        lab = 1 if label == "damaged" else 0
        for bbox in aois:
            cs = roof_chips(bbox)
            for c in cs:
                X.append(NORM(c))
                y.append(lab)
            print(f"  {label} {bbox[0]:.3f},{bbox[1]:.3f}: {len(cs)} chips")
    return torch.stack(X), torch.tensor(y)


def train(X, y, epochs=12):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, (y == 0).sum() / max((y == 1).sum(), 1)]).float()
    )
    n = len(X)
    idx = torch.randperm(n)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for i in range(0, n, 16):
            b = idx[i : i + 16]
            opt.zero_grad()
            out = model(X[b])
            loss = lossf(out, y[b])
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs} loss {tot / (n / 16):.3f}")
    model.eval()
    return model


def evaluate(model):
    print("\n== held-out per-building validation ==")
    for name, bbox in EVAL.items():
        chips = roof_chips(bbox)
        if not chips:
            print(f"  {name}: no chips")
            continue
        with torch.no_grad():
            X = torch.stack([NORM(c) for c in chips])
            p = torch.softmax(model(X), 1)[:, 1].numpy()  # P(damaged)
        print(
            f"  {name}: n={len(p)} P(damaged) mean={p.mean():.2f} "
            f"median={np.median(p):.2f} >0.5={int((p > 0.5).sum())}"
        )
        order = np.argsort(-p)
        for rank, i in enumerate(list(order[:4]) + list(order[-4:])):
            tag = "HI" if rank < 4 else "LO"
            chips[i].save(OUT / f"{name}_{tag}_{p[i]:.2f}_{i}.png")


def main():
    print("STEP 1 build dataset (weak geographic labels)")
    X, y = build_dataset()
    print(f"dataset: {len(X)} chips, {int((y == 1).sum())} damaged / {int((y == 0).sum())} intact")
    print("STEP 2 train resnet18")
    model = train(X, y)
    torch.save(model.state_dict(), OUT / "roof_condition_resnet18.pt")
    print("STEP 3 evaluate on held-out Palm St + demo")
    evaluate(model)
    print("\nsaved model + eval chips to", OUT)


if __name__ == "__main__":
    main()
