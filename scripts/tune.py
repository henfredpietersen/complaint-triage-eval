"""How the training settings were picked, using only training data.

    python scripts/tune.py

A fifth of the training complaints (chosen by hash) is held back as a validation slice.
Each setting is trained on the rest and scored on that slice. The test set is not touched.
Picked: 12 epochs, learning rate 2.0, threshold 0.6 (the lowest threshold where the
validation slice's routed accuracy was at least 93%).
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage import model as m  # noqa: E402


def main() -> int:
    snap = json.loads((ROOT / "data" / "snapshot.json").read_text(encoding="utf-8"))
    train, _ = m.split(snap["complaints"])
    val = [c for c in train if int(hashlib.sha1(("v" + c["complaint_id"]).encode()).hexdigest(), 16) % 5 == 0]
    val_ids = {c["complaint_id"] for c in val}
    fit = [c for c in train if c["complaint_id"] not in val_ids]
    print(f"fit {len(fit)}, validation {len(val)}")
    for epochs in (6, 12, 20):
        for lr in (0.5, 1.0, 2.0):
            m.EPOCHS, m.LEARNING_RATE = epochs, lr
            r = m.evaluate(m.Model(fit), val)
            sweep = ", ".join(f"{s['threshold']}: {s['routed_share']:.0%} routed at {s['accuracy_on_routed']:.1%}"
                              for s in r["sweep"][2:5])
            print(f"epochs {epochs:2d} lr {lr}: accuracy {r['accuracy_all']:.1%} | {sweep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
