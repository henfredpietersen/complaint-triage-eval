"""Complaint triage: sort a consumer complaint into a product queue, or hand it to a person.

Standard library only, so the same file runs in the browser (Pyodide) and in CI.

- Text features: words and word pairs, TF-IDF weighted. The CFPB's XXXX masking is dropped.
- Model: multinomial logistic regression trained with plain SGD and a fixed seed.
- Abstain: if the top class's probability is below the threshold, the complaint goes to a
  person instead of a queue. Accuracy is reported on the complaints it did route, next to
  how many it routed, so the trade-off is visible.
- Settings: epochs, learning rate and the 0.6 threshold were picked on a validation slice of the
  training data (scripts/tune.py), never on the test set.
- Split: a complaint is in the test set if a hash of its ID says so (30%). Test complaints
  are never seen in training.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter

TEST_SHARE = 30  # percent
SEED = 7
EPOCHS = 12
LEARNING_RATE = 2.0
L2 = 1e-5
MIN_DF = 2
DEFAULT_THRESHOLD = 0.6

STOP = set(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves also am im ive dont didnt cant wont us""".split()
)
TOKEN = re.compile(r"[a-z]+")

# Product -> the queue that would own it. Plain routing table, not learned.
QUEUES = {
    "Checking or savings account": "Deposit accounts team",
    "Credit card": "Card services team",
    "Credit reporting or other personal consumer reports": "Credit reporting disputes team",
    "Debt collection": "Collections compliance team",
    "Money transfer, virtual currency, or money service": "Payments and transfers team",
    "Mortgage": "Mortgage servicing team",
    "Vehicle loan or lease": "Auto finance team",
    "Student loan": "Student loan servicing team",
    "Payday loan, title loan, personal loan, or advance loan": "Consumer lending team",
}
HUMAN = "Person to review"


def tokens(text: str) -> list[str]:
    words = [w for w in TOKEN.findall(text.lower()) if len(w) > 1 and w not in STOP and not set(w) <= {"x"}]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]


def is_test(complaint_id: str) -> bool:
    return int(hashlib.sha1(str(complaint_id).encode()).hexdigest(), 16) % 100 < TEST_SHARE


def split(complaints: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [c for c in complaints if not is_test(c["complaint_id"])]
    test = [c for c in complaints if is_test(c["complaint_id"])]
    return train, test


class Vectorizer:
    def __init__(self, docs: list[list[str]], min_df: int = MIN_DF):
        df = Counter()
        for d in docs:
            df.update(set(d))
        terms = sorted(t for t, n in df.items() if n >= min_df)
        self.index = {t: i for i, t in enumerate(terms)}
        self.terms = terms
        n = len(docs)
        self.idf = [math.log((1 + n) / (1 + df[t])) + 1 for t in terms]

    def __call__(self, toks: list[str]) -> list[tuple[int, float]]:
        tf = Counter(self.index[t] for t in toks if t in self.index)
        vec = [(i, (1 + math.log(c)) * self.idf[i]) for i, c in sorted(tf.items())]
        norm = math.sqrt(sum(v * v for _, v in vec)) or 1.0
        return [(i, v / norm) for i, v in vec]


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    t = sum(e)
    return [x / t for x in e]


class Model:
    def __init__(self, train: list[dict]):
        self.labels = sorted({c["product"] for c in train})
        docs = [tokens(c["narrative"]) for c in train]
        self.vec = Vectorizer(docs)
        X = [self.vec(d) for d in docs]
        y = [self.labels.index(c["product"]) for c in train]
        k, n = len(self.labels), len(self.vec.terms)
        self.W = [[0.0] * n for _ in range(k)]
        self.b = [0.0] * k
        order = list(range(len(X)))
        rng = random.Random(SEED)
        for epoch in range(EPOCHS):
            rng.shuffle(order)
            lr = LEARNING_RATE / (1 + epoch)
            for j in order:
                x, target = X[j], y[j]
                p = softmax(self._scores(x))
                for c in range(k):
                    g = p[c] - (1.0 if c == target else 0.0)
                    if g == 0.0:
                        continue
                    w = self.W[c]
                    for i, v in x:
                        w[i] -= lr * (g * v + L2 * w[i])
                    self.b[c] -= lr * g
        self.train_size = len(train)

    def _scores(self, x: list[tuple[int, float]]) -> list[float]:
        return [self.b[c] + sum(w[i] * v for i, v in x) for c, w in enumerate(self.W)]

    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD, top_words: int = 6) -> dict:
        toks = tokens(text)
        x = self.vec(toks)
        probs = softmax(self._scores(x))
        ranked = sorted(range(len(probs)), key=lambda c: -probs[c])
        best = ranked[0]
        product = self.labels[best]
        routed = probs[best] >= threshold and bool(x)
        w = self.W[best]
        evidence = sorted(((w[i] * v, self.vec.terms[i]) for i, v in x if w[i] > 0), reverse=True)[:top_words]
        return {
            "product": product,
            "confidence": round(probs[best], 4),
            "routed": routed,
            "queue": QUEUES.get(product, product) if routed else HUMAN,
            "reason": None if routed else ("no known words" if not x else
                                            f"top confidence {probs[best]:.0%} is below {threshold:.0%}"),
            "ranking": [{"product": self.labels[c], "p": round(probs[c], 4)} for c in ranked],
            "evidence": [t.replace("_", " ") for _, t in evidence],
            "known_words": len(x),
        }


def evaluate(model: Model, test: list[dict], threshold: float = DEFAULT_THRESHOLD,
             sweep=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    preds = [model.predict(c["narrative"], threshold) for c in test]
    labels = model.labels
    confusion = {a: dict.fromkeys(labels, 0) for a in labels}
    for c, p in zip(test, preds, strict=True):
        confusion[c["product"]][p["product"]] += 1

    def rate(th):
        routed = [(c, p) for c, p in zip(test, preds, strict=True) if p["confidence"] >= th and p["known_words"]]
        right = sum(c["product"] == p["product"] for c, p in routed)
        return {"threshold": th, "routed": len(routed), "routed_share": round(len(routed) / len(test), 4),
                "accuracy_on_routed": round(right / len(routed), 4) if routed else None,
                "wrong_queue": len(routed) - right}

    per_class = {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[o][lab] for o in labels if o != lab)
        fn = sum(confusion[lab][o] for o in labels if o != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"support": tp + fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    correct = sum(c["product"] == p["product"] for c, p in zip(test, preds, strict=True))
    mistakes = [
        {"complaint_id": c["complaint_id"], "actual": c["product"], "predicted": p["product"],
         "confidence": p["confidence"], "routed": p["routed"], "issue": c["issue"],
         "excerpt": c["narrative"][:280]}
        for c, p in zip(test, preds, strict=True) if c["product"] != p["product"]
    ]
    mistakes.sort(key=lambda m: -m["confidence"])
    by_label = Counter(c["product"] for c in test)
    return {
        "train_size": model.train_size,
        "test_size": len(test),
        "vocabulary": len(model.vec.terms),
        "labels": labels,
        "chance_accuracy": round(max(by_label.values()) / len(test), 4),
        "accuracy_all": round(correct / len(test), 4),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(labels), 4),
        "at_threshold": rate(threshold),
        "sweep": [rate(t) for t in sweep],
        "per_class": per_class,
        "confusion": confusion,
        "mistakes": mistakes,
    }


def similar(model: Model, train: list[dict], text: str, n: int = 3) -> list[dict]:
    """Most similar training complaints by cosine similarity, to show a reviewer past cases."""
    q = dict(model.vec(tokens(text)))
    if not q:
        return []
    if not hasattr(model, "_train_vecs"):
        model._train_vecs = [dict(model.vec(tokens(c["narrative"]))) for c in train]
    scored = []
    for c, v in zip(train, model._train_vecs, strict=True):
        s = sum(w * v.get(i, 0.0) for i, w in q.items())
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda t: (-t[0], t[1]["complaint_id"]))
    return [{"similarity": round(s, 3), "product": c["product"], "issue": c["issue"],
             "company_response": c["company_response"], "excerpt": c["narrative"][:220]} for s, c in scored[:n]]
