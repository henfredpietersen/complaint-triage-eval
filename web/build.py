"""Build the live demo page.

    python web/build.py

Writes docs/demo.js (what the case-study page loads) and docs/demo.html (open it in a browser).
The page embeds triage/__init__.py, triage/model.py and web/driver.py byte for byte and runs
them with Pyodide. It fetches data/snapshot.json from the same commit on jsDelivr, or from
../data/snapshot.json when demo.html is opened from a checkout. CI rebuilds the page and fails
if the committed copy differs.

Embed on any page:
  <div id="ov-complaint-triage" data-snapshot="https://cdn.jsdelivr.net/gh/henfredpietersen/complaint-triage-eval@<commit>/data/snapshot.json"></div>
  <script defer src="https://cdn.jsdelivr.net/gh/henfredpietersen/complaint-triage-eval@<commit>/docs/demo.js"></script>
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED = {
    "/home/pyodide/triage/__init__.py": "triage/__init__.py",
    "/home/pyodide/triage/model.py": "triage/model.py",
    "/home/pyodide/driver.py": "web/driver.py",
}

STANDALONE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complaint triage, scored</title>
<style>html,body{margin:0;background:#05070d}</style></head>
<body>
<div id="ov-complaint-triage" data-snapshot="../data/snapshot.json"></div>
<script src="demo.js"></script>
</body></html>
"""


def build() -> str:
    files = {dest: (ROOT / src).read_text(encoding="utf-8") for dest, src in EMBED.items()}
    template = (ROOT / "web" / "page.js").read_text(encoding="utf-8")
    assert template.count("__FILES__") == 1
    return template.replace("__FILES__", json.dumps(files).replace("</", "<\\/"))


def main() -> int:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    script = build()
    (docs / "demo.js").write_text(script, encoding="utf-8")
    (docs / "demo.html").write_text(STANDALONE, encoding="utf-8")
    print(f"Wrote docs/demo.js ({len(script):,} bytes) and docs/demo.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
