"""Merge an active-learning label batch into the pool's labels.json.

mine_uncertain.py writes labels_todo.json ({idx: -1}); a human replaces each -1
with 0 (intact) or 1 (damaged) after reading mine_uncertain.png / mine_hardneg.png.
Any idx still -1 is skipped. Run inside the worker:

    python -m scripts.merge_labels
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POOL = Path("/data/imagery/label_pool")


def main():
    labels = {int(k): int(v) for k, v in json.loads((POOL / "labels.json").read_text()).items()}
    todo = json.loads((POOL / "labels_todo.json").read_text())
    added = 0
    for k, v in todo.items():
        v = int(v)
        if v in (0, 1):
            labels[int(k)] = v
            added += 1
    (POOL / "labels.json").write_text(json.dumps({str(k): labels[k] for k in sorted(labels)}))
    n1 = sum(labels.values())
    print(
        f"merged {added} new labels -> {len(labels)} total ({n1} damaged / {len(labels) - n1} intact)"
    )


if __name__ == "__main__":
    main()
