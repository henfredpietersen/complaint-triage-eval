"""Glue between the demo page and triage/model.py.

The page loads data/snapshot.json, then runs this file and triage/model.py unchanged in the
visitor's browser (Pyodide). The model is trained there, on the training split, and scored on
the held-out split. tests/test_web_demo.py makes the same calls under CPython.
"""

from __future__ import annotations

import json

from triage import model as m

_state: dict = {}


def setup(snapshot_json: str) -> str:
    snap = json.loads(snapshot_json)
    train, test = m.split(snap["complaints"])
    model = m.Model(train)
    report = m.evaluate(model, test)
    _state.update(model=model, train=train, test=test)
    examples = []
    seen = set()
    for c in test:  # one held-out complaint per product, in a fixed order
        if c["product"] not in seen:
            seen.add(c["product"])
            examples.append({"product": c["product"], "issue": c["issue"], "text": c["narrative"]})
    return json.dumps({
        "report": report,
        "threshold": m.DEFAULT_THRESHOLD,
        "queues": m.QUEUES,
        "examples": sorted(examples, key=lambda e: e["product"]),
        "source": snap["source"],
        "method": snap["method"],
        "rows_in_export": snap["rows_in_export"],
        "rows_with_narrative": snap["rows_with_narrative"],
        "product_counts": snap["product_counts"],
    })


def classify(text: str, threshold: float = m.DEFAULT_THRESHOLD) -> str:
    model = _state["model"]
    out = model.predict(text, threshold)
    out["similar"] = m.similar(model, _state["train"], text)
    out["threshold"] = threshold
    return json.dumps(out)
