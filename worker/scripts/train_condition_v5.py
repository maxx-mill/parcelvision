"""Damage v5 — roof-condition classifier with a research-backed training recipe.

Over v4 this changes five things, each grounded in the building-damage /
shortcut-learning literature:

  1. Fixed-GSD, polygon-masked, multi-scale chips (worker.pipeline.roof_chip),
     used identically at train and inference time. Kills the v4 scale shortcut
     that over-flagged big flat/institutional roofs.
  2. Focal loss with inverse-frequency alpha (xBD winners) instead of plain
     weighted cross-entropy — focuses on the hard, rare damaged tiles.
  3. Strong augmentation: scale-jitter (RandomResizedCrop), full rotation,
     colour jitter, and RandomErasing (occlusion) — the biggest lever on texture
     bias.
  4. Honest evaluation: grouped k-fold CV (tiles from one building never split
     across folds) reporting precision/recall/F1/AP + confusion, plus the
     held-out Palm St (damaged) vs demo (intact) sanity check with D4 TTA.
  5. Temperature scaling on out-of-fold logits so P(damaged) is calibrated and
     the deployed thresholds actually mean something.

Reads /data/imagery/label_pool/labels.json ({idx: 0 intact | 1 damaged}), where
idx is the make_label_pool building index (pool_lib replays the same order).
Saves a checkpoint dict {state_dict, temperature, chip_px, chip_mpp, arch}.
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scripts.pool_lib import building_tiles, iter_buildings  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from torchvision import models, transforms  # noqa: E402
from worker.pipeline.roof_chip import CHIP_MPP, CHIP_PX  # noqa: E402

POOL = Path("/data/imagery/label_pool")
OUT = Path("/data/imagery/condition_v5")
OUT.mkdir(parents=True, exist_ok=True)
EVAL = {
    "palm_damaged": [-90.2035, 38.6548, -90.1999, 38.6568],
    "demo_intact": [-90.3150, 38.6478, -90.3120, 38.6498],
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MAX_TILES = 16  # cap so one huge building can't dominate the loss
CV_EPOCHS = int(os.environ.get("CV_EPOCHS", "12"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "26"))
SKIP_CV = os.environ.get("SKIP_CV", "0") == "1"  # fast final-only rerun
TEMP_OVERRIDE = float(os.environ.get("TEMP", "0.823"))  # reuse CV-fitted T when skipping
SEED = 0

AUG = transforms.Compose(
    [
        transforms.RandomResizedCrop(CHIP_PX, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.12)),
    ]
)
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        w = self.alpha[target]
        return (-w * (1 - pt) ** self.gamma * logpt).mean()


def new_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def train(imgs, ys, epochs, alpha, log=None):
    torch.manual_seed(SEED)
    model = new_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lossf = FocalLoss(alpha)
    y = torch.tensor(ys)
    n = len(imgs)
    model.train()
    for ep in range(epochs):
        order = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 32):
            b = order[i : i + 32]
            # augment only the current batch -> low, flat peak memory
            X = torch.stack([AUG(imgs[j]) for j in b.tolist()])
            opt.zero_grad()
            loss = lossf(model(X), y[b])
            loss.backward()
            opt.step()
            tot += loss.item()
        if log and (ep + 1) % 4 == 0:
            print(f"    {log} epoch {ep + 1}/{epochs} loss {tot / (n / 32):.3f}", flush=True)
    model.eval()
    return model


def tile_logits(model, chips):
    with torch.no_grad():
        X = torch.stack([NORM(c) for c in chips])
        return model(X)


def building_score(model, chips, temp=1.0, tta=True):
    """Mean P(damaged) over a building's tiles, temperature-applied per tile."""
    if not chips:
        return None
    with torch.no_grad():
        X = torch.stack([NORM(c) for c in chips])
        if tta:
            acc = torch.zeros(len(X), 2)
            for k in range(4):
                r = torch.rot90(X, k, dims=[2, 3])
                for flip in (False, True):
                    v = torch.flip(r, dims=[3]) if flip else r
                    acc += F.softmax(model(v) / temp, 1)
            p = (acc / 8.0)[:, 1]
        else:
            p = F.softmax(model(X) / temp, 1)[:, 1]
    return float(p.mean())


def fit_temperature(logits, labels):
    """One scalar T minimising NLL on held-out (out-of-fold) tile logits."""
    logits = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def main():
    labels = {int(k): int(v) for k, v in json.loads((POOL / "labels.json").read_text()).items()}
    # Extract fixed-GSD tiles once per labelled building (pool_lib caches imagery).
    tiles_by_b, y_by_b = {}, {}
    for idx, ai, geom in iter_buildings():
        if idx not in labels:
            continue
        ch = building_tiles(ai, geom)[:MAX_TILES]
        if ch:
            tiles_by_b[idx] = [Image.fromarray(c) for c in ch]
            y_by_b[idx] = labels[idx]
    bids = sorted(tiles_by_b)
    n_dmg = sum(y_by_b[b] for b in bids)
    n_tiles = sum(len(tiles_by_b[b]) for b in bids)
    print(
        f"buildings: {len(bids)} ({n_dmg} damaged / {len(bids) - n_dmg} intact); "
        f"tiles: {n_tiles}; chip {CHIP_PX}px @ {CHIP_MPP} m/px"
    )

    # Flatten to tiles carrying their building's label + group id.
    imgs, ys, groups = [], [], []
    for b in bids:
        for im in tiles_by_b[b]:
            imgs.append(im)
            ys.append(y_by_b[b])
            groups.append(b)
    ys = np.array(ys)
    groups = np.array(groups)
    n1 = int((ys == 1).sum())
    n0 = int((ys == 0).sum())
    alpha = torch.tensor([len(ys) / (2 * n0), len(ys) / (2 * n1)], dtype=torch.float32)

    # ---- grouped k-fold CV: honest building-level metrics + OOF logits for T ----
    cv_metrics = None
    if SKIP_CV:
        temp = TEMP_OVERRIDE
        print(f"\n== SKIP_CV: reusing temperature T={temp:.3f} ==", flush=True)
    else:
        print("\n== grouped 4-fold CV (building-level) ==", flush=True)
        gkf = GroupKFold(n_splits=4)
        oof_logits, oof_labels = [], []
        b_true, b_pred, b_score = [], [], []
        for fold, (tr, te) in enumerate(gkf.split(imgs, ys, groups)):
            m = train([imgs[i] for i in tr], ys[tr].tolist(), CV_EPOCHS, alpha)
            # OOF tile logits (for temperature) + building-level aggregation
            te_chips = [imgs[i] for i in te]
            lg = tile_logits(m, te_chips).numpy()
            oof_logits.extend(lg.tolist())
            oof_labels.extend(ys[te].tolist())
            for b in sorted(set(groups[te])):
                s = building_score(m, tiles_by_b[b], temp=1.0, tta=True)
                b_score.append(s)
                b_pred.append(int(s >= 0.5))
                b_true.append(y_by_b[b])
            print(
                f"  fold {fold}: trained {len(tr)} tiles, scored {len(set(groups[te]))} bldgs",
                flush=True,
            )

        p, r, f1, _ = precision_recall_fscore_support(
            b_true, b_pred, average="binary", zero_division=0
        )
        ap = average_precision_score(b_true, b_score)
        cm = confusion_matrix(b_true, b_pred)
        acc = float((np.array(b_true) == np.array(b_pred)).mean())
        print(
            f"  building CV: acc={acc:.2f} precision={p:.2f} recall={r:.2f} F1={f1:.2f} AP={ap:.2f}"
        )
        print(f"  confusion [[TN FP][FN TP]] = {cm.tolist()}", flush=True)
        cv_metrics = {"acc": acc, "precision": p, "recall": r, "f1": f1, "ap": ap}
        temp = fit_temperature(oof_logits, oof_labels)
        print(f"  fitted temperature T={temp:.3f}", flush=True)

    # ---- final model on ALL labelled tiles ----
    print("\n== final model (all labels) ==", flush=True)
    model = train(imgs, ys.tolist(), FINAL_EPOCHS, alpha, log="final")
    ckpt = {
        "state_dict": model.state_dict(),
        "temperature": temp,
        "chip_px": CHIP_PX,
        "chip_mpp": CHIP_MPP,
        "arch": "resnet18",
    }
    torch.save(ckpt, OUT / "roof_condition_resnet18.pt")

    # ---- held-out sanity check: Palm St (damaged) vs demo (intact) ----
    print("\n== held-out validation (building-level, D4 TTA, calibrated) ==")
    report = {"cv": cv_metrics, "temp": temp}
    import geoai
    import geopandas as gpd
    from scripts.pool_lib import hires
    from worker.pipeline.roof_chip import roof_tiles

    for name, bbox in EVAL.items():
        ov = OUT / f"ov_{name}.geojson"
        if not ov.exists():
            geoai.download_overture_buildings(bbox=tuple(bbox), output=str(ov))
        gdf = gpd.read_file(ov).to_crs("EPSG:3857")
        arr, tf = hires(bbox)
        scores = []
        for geom in gdf.geometry:
            ch = roof_tiles(arr, tf, geom)[:MAX_TILES]
            s = building_score(model, [Image.fromarray(c) for c in ch], temp=temp, tta=True)
            if s is not None:
                scores.append(s)
        scores = np.array(scores)
        report[name] = {
            "n": int(len(scores)),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "frac_gt_0.5": float((scores > 0.5).mean()),
        }
        print(
            f"  {name}: n={len(scores)} P(damaged) mean={scores.mean():.2f} "
            f"median={np.median(scores):.2f} >0.5={(scores > 0.5).mean():.2f}"
        )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nsaved checkpoint + report to", OUT)


if __name__ == "__main__":
    main()
