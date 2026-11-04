"""Model rules, tested on a small hand-made set where the right answers are obvious."""

import pytest

from triage import model as m

WORDS = {
    "Mortgage": "mortgage escrow servicer foreclosure loan modification payment house",
    "Credit card": "credit card statement charge dispute merchant interest limit card",
    "Debt collection": "collector collection agency debt validation calls owe letter",
}


def fixture(n_per=12):
    out = []
    for p, words in WORDS.items():
        w = words.split()
        for i in range(n_per):
            out.append({"complaint_id": f"{p[:3]}{i}", "product": p, "issue": "x", "company_response": "Closed",
                        "narrative": " ".join(w[(i + k) % len(w)] for k in range(6)) + f" XXXX XX/XX/2026 note{i}"})
    return out


@pytest.fixture(scope="module")
def model():
    return m.Model(fixture())


def test_tokens_drop_masking_stopwords_and_add_pairs():
    t = m.tokens("The XXXX bank charged me XX/XX/2026 an overdraft fee")
    assert "xxxx" not in t and "xx" not in t and "the" not in t
    assert "overdraft_fee" in t and "bank_charged" in t


def test_split_is_stable_and_disjoint():
    data = fixture()
    a_train, a_test = m.split(data)
    b_train, b_test = m.split(list(reversed(data)))
    assert {c["complaint_id"] for c in a_test} == {c["complaint_id"] for c in b_test}
    assert not {c["complaint_id"] for c in a_train} & {c["complaint_id"] for c in a_test}


def test_clear_text_is_routed_to_the_right_queue(model):
    p = model.predict("my mortgage servicer added escrow and started foreclosure")
    assert p["product"] == "Mortgage" and p["routed"] and p["queue"] == "Mortgage servicing team"
    assert "mortgage" in p["evidence"]
    assert abs(sum(r["p"] for r in p["ranking"]) - 1) < 1e-3


def test_unknown_words_go_to_a_person(model):
    p = model.predict("zebra quantum lighthouse")
    assert not p["routed"] and p["queue"] == m.HUMAN and p["reason"] == "no known words"


def test_low_confidence_goes_to_a_person(model):
    p = model.predict("my mortgage servicer added escrow", threshold=0.9999)
    assert not p["routed"] and "below" in p["reason"]


def test_training_is_deterministic():
    a, b = m.Model(fixture()), m.Model(fixture())
    assert a.W == b.W and a.b == b.b


def test_evaluate_counts_add_up(model):
    test = fixture(4)
    r = m.evaluate(model, test, threshold=0.5)
    assert sum(sum(row.values()) for row in r["confusion"].values()) == len(test)
    for s in r["sweep"]:
        assert s["routed"] <= len(test)
    assert r["sweep"][0]["routed"] == len(test)  # threshold 0 routes everything with known words
    assert len(r["mistakes"]) == len(test) - sum(r["confusion"][lab][lab] for lab in r["labels"])


def test_similar_returns_same_product_first(model):
    s = m.similar(model, fixture(), "credit card charge dispute with the merchant")
    assert s and s[0]["product"] == "Credit card"
