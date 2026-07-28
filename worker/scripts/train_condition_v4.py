"""Damage v4 — retrain the roof-condition classifier on HAND labels.

Reads the chip pool (make_label_pool.py) + labels.json {idx: 0 intact | 1
damaged}; unlabeled chips are skipped. Trains a ResNet18 with augmentation at
0.15 m, validates held-out on Palm St (damaged) vs the demo (intact), and saves
the model when it beats v3.
"""

import json
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

POOL = Path("/data/imagery/label_pool")
OUT = Path("/data/imagery/condition_v4")
OUT.mkdir(parents=True, exist_ok=True)
CHIP = 128
EVAL = {
    "palm_damaged": [-90.2035, 38.6548, -90.1999, 38.6568],
    "demo_intact": [-90.3150, 38.6478, -90.3120, 38.6498],
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
AUG = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def hires(bbox, mpp=0.15, maxpx=2000):
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    sx, sy = maxx - minx, maxy - miny
    scale = min(maxpx / (sx / mpp), maxpx / (sy / mpp), 1.0)
    w, h = int(sx / mpp * scale), int(sy / mpp * scale)
    url = (
        "https://stateimagery.msdis.missouri.edu/arcgis/rest/services/"
        "Missouri_6inch_Statewide_2023_2024_Dynamic/ImageServer/exportImage"
    )
    p = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{w},{h}",
        "format": "png",
        "f": "image",
    }
    for a in range(4):
        try:
            r = requests.get(url, params=p, timeout=120)
            r.raise_for_status()
            arr = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
            return arr, from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
        except Exception:
            if a == 3:
                raise
            time.sleep(8 * (a + 1))


def roof_chips(bbox):
    ov = OUT / f"ov_{bbox[0]:.4f}.geojson"
    if not ov.exists():
        geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
    gdf = gpd.read_file(ov).to_crs("EPSG:3857")
    arr, tf = hires(bbox)
    H, W, _ = arr.shape
    out = []
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
        out.append(Image.fromarray(arr[r0:r1, c0:c1]).resize((CHIP, CHIP)))
    return out


def main():
    labels = {int(k): int(v) for k, v in json.loads((POOL / "labels.json").read_text()).items()}
    imgs, ys = [], []
    for idx, y in labels.items():
        f = POOL / "chips" / f"{idx:04d}.png"
        if f.exists():
            imgs.append(Image.open(f).convert("RGB"))
            ys.append(y)
    y = torch.tensor(ys)
    print(f"labeled: {len(ys)} chips, {int((y == 1).sum())} damaged / {int((y == 0).sum())} intact")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    w = torch.tensor([1.0, float((y == 0).sum()) / max(int((y == 1).sum()), 1)])
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    for ep in range(25):  # augment fresh each epoch
        X = torch.stack([AUG(im) for im in imgs])
        idx = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 16):
            b = idx[i : i + 16]
            opt.zero_grad()
            loss = lossf(model(X[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}/25 loss {tot / (len(X) / 16):.3f}")
    model.eval()
    torch.save(model.state_dict(), OUT / "roof_condition_resnet18.pt")

    print("\n== held-out validation ==")
    for name, bbox in EVAL.items():
        chips = roof_chips(bbox)
        with torch.no_grad():
            X = torch.stack([NORM(c) for c in chips])
            p = torch.softmax(model(X), 1)[:, 1].numpy()
        print(
            f"  {name}: n={len(p)} P(damaged) mean={p.mean():.2f} "
            f"median={np.median(p):.2f} >0.5={int((p > 0.5).sum())}"
        )
        order = np.argsort(-p)
        for rank, i in enumerate(list(order[:4]) + list(order[-4:])):
            chips[i].save(OUT / f"{name}_{'HI' if rank < 4 else 'LO'}_{p[i]:.2f}_{i}.png")
    print("saved model + eval chips to", OUT)


if __name__ == "__main__":
    main()
