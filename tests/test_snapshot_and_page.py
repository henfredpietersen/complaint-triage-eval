"""The real snapshot, the published scores and the page."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

from triage import model as m

ROOT = Path(__file__).resolve().parents[1]
SNAP = json.loads((ROOT / "data" / "snapshot.json").read_text(encoding="utf-8"))


def test_snapshot_is_what_it_says():
    c = SNAP["complaints"]
    meth = SNAP["method"]
    assert len(c) == meth["per_product"] * meth["products"]
    assert set(Counter(x["product"] for x in c).values()) == {meth["per_product"]}
    assert len({x["complaint_id"] for x in c}) == len(c)
    assert all(x["narrative"].strip() for x in c)
    assert all(len(x["narrative"]) <= meth["max_chars"] for x in c)
    assert SNAP["source"]["file_url"].startswith("https://files.consumerfinance.gov/")
    assert len(SNAP["source"]["file_sha256"]) == 64
    # ZIP code and state are dropped
    assert not {"zip_code", "state"} & set(c[0])


def test_readme_numbers_match_a_fresh_run():
    """The README quotes these. If the model or data changes, this fails until the README is updated."""
    train, test = m.split(SNAP["complaints"])
    r = m.evaluate(m.Model(train), test)
    at = r["at_threshold"]
    assert (r["train_size"], r["test_size"]) == (845, 355)
    assert (r["accuracy_all"], r["chance_accuracy"], r["macro_f1"]) == (0.7944, 0.2085, 0.7882)
    assert (at["routed_share"], at["accuracy_on_routed"], at["wrong_queue"]) == (0.7211, 0.9023, 25)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for s in ("79.4%", "20.9%", "72.1%", "90.2%", "25 of 355", "845", "619,940", "17,528"):
        assert s in readme, s


def test_published_page_embeds_current_code():
    spec = importlib.util.spec_from_file_location("build", ROOT / "web" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "demo.js").read_text(encoding="utf-8") == mod.build(), \
        "docs/demo.js is stale: run python web/build.py"


def test_driver_runs_like_the_page():
    import sys
    sys.path.insert(0, str(ROOT / "web"))
    import driver

    info = json.loads(driver.setup(json.dumps(SNAP)))
    assert info["report"]["test_size"] == 355 and len(info["examples"]) == 6
    test_ids = {c["complaint_id"] for c in m.split(SNAP["complaints"])[1]}
    texts = {c["narrative"]: c["complaint_id"] for c in SNAP["complaints"]}
    assert all(texts[e["text"]] in test_ids for e in info["examples"]), "examples must be held-out complaints"
    out = json.loads(driver.classify("I was charged two overdraft fees on my checking account"))
    assert out["product"] == "Checking or savings account" and out["routed"]
    assert len(out["similar"]) == 3
