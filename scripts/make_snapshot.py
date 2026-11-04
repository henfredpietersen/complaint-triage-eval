"""Build data/snapshot.json from the CFPB's archived complaint export.

    python scripts/make_snapshot.py --zip CCDB_Export_21_August_2026.zip

The CFPB stopped publishing complaint narratives on 14 August 2026 and moved the published
ones to a FOIA archive. This script reads one archived monthly export, keeps only complaints
that have a narrative, picks the most common products, and draws a fixed-seed sample of the
same size for each product. The workflow in .github/workflows/snapshot.yml downloads the zip
and runs this, so the snapshot can be rebuilt by anyone from the public source.

Kept per complaint: id, date received, product, issue, company, company response, timely
flag and the narrative. ZIP code and state are dropped. The CFPB already replaces personal
details in narratives with XXXX before publishing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
SEED = 20260814
PER_PRODUCT = 200
PRODUCTS = 6
MAX_CHARS = 4000  # longer narratives are cut here; the cut is recorded per complaint

# CSV header (lower-cased, stripped) -> our field. The export's header names are matched loosely.
COLUMNS = {
    "complaint_id": ("complaint id", "complaint_id"),
    "date_received": ("date received", "date_received"),
    "product": ("product",),
    "issue": ("issue",),
    "company": ("company",),
    "company_response": ("company response to consumer", "company_response"),
    "timely": ("timely response?", "timely"),
    "narrative": ("consumer complaint narrative", "complaint_what_happened", "narrative"),
}


csv.field_size_limit(sys.maxsize)


def _map_header(header: list[str]) -> dict[str, int]:
    norm = [h.strip().lower().lstrip("﻿") for h in header]
    out = {}
    for field, names in COLUMNS.items():
        for n in names:
            if n in norm:
                out[field] = norm.index(n)
                break
    missing = [f for f in COLUMNS if f not in out]
    if missing:
        raise SystemExit(f"Export is missing column(s) {missing}. Header was: {header}")
    return out


def read_rows(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV inside {zip_path}: {z.namelist()}")
        for name in names:
            with z.open(name) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
                cols = _map_header(next(reader))
                for row in reader:
                    if len(row) <= max(cols.values()):
                        continue
                    yield {f: row[i].strip() for f, i in cols.items()}


def describe(zip_path: Path, n: int = 20000) -> None:
    """Print the header and how often each column is filled in the first n rows."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(x for x in z.namelist() if x.lower().endswith(".csv"))
        with z.open(name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
            header = next(reader)
            filled = Counter()
            lengths = Counter()
            rows = 0
            for row in reader:
                rows += 1
                lengths[len(row)] += 1
                for i, v in enumerate(row):
                    if v.strip():
                        filled[i] += 1
                if rows >= n:
                    break
    print(f"{name}: {len(header)} columns; row lengths in first {rows}: {dict(lengths.most_common(5))}")
    for i, h in enumerate(header):
        print(f"  {i:2d} {h!r}: filled {filled[i]}/{rows}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(zip_path: Path, source_url: str) -> dict:
    total = 0
    by_product: dict[str, list[dict]] = defaultdict(list)
    for r in read_rows(zip_path):
        total += 1
        if r["narrative"] and r["product"]:
            by_product[r["product"]].append(r)
    with_text = sum(len(v) for v in by_product.values())
    counts = Counter({p: len(v) for p, v in by_product.items()})
    chosen = [p for p, n in counts.most_common(PRODUCTS) if n >= PER_PRODUCT]
    rng = random.Random(SEED)
    sample = []
    for p in sorted(chosen):
        rows = sorted(by_product[p], key=lambda r: r["complaint_id"])
        for r in rng.sample(rows, PER_PRODUCT):
            text = r["narrative"]
            sample.append({**r, "narrative": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS})
    sample.sort(key=lambda r: r["complaint_id"])
    return {
        "source": {
            "archive_page": ARCHIVE_PAGE,
            "file_url": source_url,
            "file_sha256": sha256(zip_path),
            "publisher": "Consumer Financial Protection Bureau (CFPB), Consumer Complaint Database",
            "licence": "Public domain (CFPB complaint data is published as CC0)",
        },
        "method": {
            "seed": SEED,
            "per_product": PER_PRODUCT,
            "products": len(chosen),
            "max_chars": MAX_CHARS,
            "note": "Equal sample per product. The real mix is far more uneven: see product_counts.",
        },
        "rows_in_export": total,
        "rows_with_narrative": with_text,
        "product_counts": dict(counts.most_common()),
        "complaints": sample,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, type=Path)
    p.add_argument("--source-url", required=True)
    p.add_argument("--out", default="data/snapshot.json", type=Path)
    p.add_argument("--describe", action="store_true", help="print the header and fill rates, then stop")
    a = p.parse_args(argv)
    if a.describe:
        describe(a.zip)
        return 0
    snap = build(a.zip, a.source_url)
    if snap["method"]["products"] < 3:
        print(json.dumps(snap["product_counts"], indent=1))
        raise SystemExit(f"Only {snap['rows_with_narrative']} rows have a narrative: not enough for a snapshot.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{snap['rows_in_export']:,} rows, {snap['rows_with_narrative']:,} with a narrative; "
          f"kept {len(snap['complaints'])} across {snap['method']['products']} products -> {a.out}")
    print(json.dumps(snap["product_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
