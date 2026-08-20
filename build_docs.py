#!/usr/bin/env python3
"""
Renders the same page run_explorer.py serves at "/" to docs/index.html, for
publishing as a static GitHub Pages site (Settings -> Pages -> Deploy from
branch -> main -> /docs). Reuses run_explorer's data loading and HTML
template directly, so the static output is guaranteed to match the live
server -- run this again any time the data or template changes.

Usage: python build_docs.py
"""
import json
from pathlib import Path

from flask import render_template_string
from markupsafe import Markup

import run_explorer

OUT_DIR = Path(__file__).parent / "docs"
OUT_FILE = OUT_DIR / "index.html"


def main() -> None:
    with run_explorer.app.app_context():
        html = render_template_string(
            run_explorer.HTML_TEMPLATE,
            stats=Markup(json.dumps(run_explorer.DATASET_STATS)),
            raw=Markup(json.dumps(run_explorer.RAW_DATA)),
            my_model=Markup(json.dumps(run_explorer.load_my_model_stats())),
        )
    OUT_DIR.mkdir(exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_FILE}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
