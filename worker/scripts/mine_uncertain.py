"""Damage v5 — active-learning miner (uncertainty + hard-negative sampling).

Given a trained v5 checkpoint, score every pool building the human hasn't
labelled yet and surface the two kinds of chips that teach the model the most
per label (active-learning literature):

  * UNCERTAIN — building score nearest 0.5 (the decision boundary).
  * HARD-NEGATIVE — LARGE roofs the model calls damaged with high confidence.
    These are exactly the flat/institutional false positives v4 suffered from,
    so labelling them (almost all "intact") directly attacks the confound.

Writes montages (index-labelled cells) + a labels_todo.json stub to
/data/imagery/label_pool for a human to fill in; merge the result into
labels.json and retrain.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torchvision import models, transforms  # noqa: E402

from scripts.pool_lib import building_tiles, iter_buildings  # noqa: E402

POOL = Path("/data/imagery/label_pool")
CKPT = Path("/data/imagery/condition_v5/roof_condition_resnet18.pt")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
N_UNCERTAIN = 36
N_HARDNEG = 24


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    temp = ckpt.get("temperature", 1.0)
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, temp


def score(model, temp, chips):
    with torch.no_grad():
        X = torch.stack([NORM(c) for c in chips])
        acc = torch.zeros(len(X), 2)
        for k in range(4):
            r = torch.rot90(X, k, dims=[2, 3])
            for flip in (False, True):
                v = torch.flip(r, dims=[3]) if flip else r
                acc += F.softmax(model(v) / temp, 1)
        return float((acc / 8.0)[:, 1].mean())


def montage(items, path, title):
    """items: list of (idx, preview_img, score). 6-wide grid, index + score."""
    cols, cell = 6, 150
    rows = (len(items) + cols - 1) // cols
    m = Image.new("RGB", (cols * cell, rows * cell + 22), (18, 18, 22))
    dr = ImageDraw.Draw(m)
    dr.text((6, 5), title, fill=(255, 240, 0))
    for j, (idx, img, s) in enumerate(items):
        c = img.resize((cell - 6, cell - 6))
        x, y = (j % cols) * cell, (j // cols) * cell + 22
        m.paste(c, (x + 3, y + 3))
        dr.rectangle([x + 2, y + 2, x + 78, y + 16], fill=(0, 0, 0))
        dr.text((x + 4, y + 4), f"{idx} p={s:.2f}", fill=(255, 240, 0))
    m.save(path)


def main():
    labels = {int(k) for k in json.loads((POOL / "labels.json").read_text())}
    model, temp = load_model()
    rows = []
    for idx, ai, geom in iter_buildings():
        if idx in labels:
            continue
        ch = building_tiles(ai, geom)
        if not ch:
            continue
        chips = [Image.fromarray(c) for c in ch]
        s = score(model, temp, chips)
        area = geom.area  # m^2 in EPSG:3857
        rows.append((idx, chips[0], s, area, len(ch)))
    print(f"scored {len(rows)} unlabelled buildings")

    uncertain = sorted(rows, key=lambda r: abs(r[2] - 0.5))[:N_UNCERTAIN]
    big = [r for r in rows if r[3] >= 250]  # roughly >250 m^2 footprint
    hardneg = sorted(big, key=lambda r: -r[2])[:N_HARDNEG]

    montage(
        [(r[0], r[1], r[2]) for r in uncertain],
        POOL / "mine_uncertain.png",
        "UNCERTAIN (score near 0.5) — label 1=damaged 0=intact",
    )
    montage(
        [(r[0], r[1], r[2]) for r in hardneg],
        POOL / "mine_hardneg.png",
        "HARD-NEG (big roofs called damaged) — expect mostly 0=intact",
    )
    todo = {str(r[0]): -1 for r in uncertain + hardneg}
    (POOL / "labels_todo.json").write_text(json.dumps(todo, indent=0))
    print(
        f"wrote mine_uncertain.png ({len(uncertain)}), mine_hardneg.png "
        f"({len(hardneg)}), labels_todo.json ({len(todo)} idx) -> {POOL}"
    )


if __name__ == "__main__":
    main()
