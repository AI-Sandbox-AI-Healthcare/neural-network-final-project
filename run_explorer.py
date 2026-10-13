#!/usr/bin/env python3
"""
Final Project — AI-Sandbox Model Explorer
Run from the repo root:  python run_explorer.py
Opens http://localhost:3002
"""
import csv
import json
import logging
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template_string

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
DATA_FILE  = DATA_DIR / "synthea_enriched_features.csv"
RAW_DIR    = DATA_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the enriched patient dataset if present; otherwise generate a small,
# honestly-labeled sample from the real raw CSVs and save it, so the page
# always shows real (if limited) data instead of a hardcoded placeholder.
# ---------------------------------------------------------------------------
PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]
_PAIN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b"
)
_SAMPLE_SIZE = 20
_AGE_REFERENCE = date(2026, 1, 1)


def _age_at_reference(birthdate: str) -> float:
    y, m, d = (int(p) for p in birthdate.split("-"))
    years = _AGE_REFERENCE.year - y - ((_AGE_REFERENCE.month, _AGE_REFERENCE.day) < (m, d))
    return float(years)


def _generate_sample_dataset() -> list:
    """Builds a small (_SAMPLE_SIZE-row) enriched-style sample from the real
    patients.csv + conditions.csv, since the full engineered feature set
    used by the real pipeline isn't part of this repo. binary_label is a
    keyword-match stand-in (same approach as the course's assignment-1),
    not the real pipeline's label -- good enough for the explorer UI's
    aggregate stats, not a substitute for the real dataset."""
    with open(DATA_DIR / "patients.csv", encoding="utf-8", newline="") as f:
        patients = list(csv.DictReader(f))[:_SAMPLE_SIZE]

    pain_patient_ids = set()
    with open(DATA_DIR / "conditions.csv", encoding="utf-8", newline="") as f:
        wanted_ids = {p["Id"] for p in patients}
        for row in csv.DictReader(f):
            if row["PATIENT"] in wanted_ids and _PAIN_PATTERN.search(row["DESCRIPTION"].lower()):
                pain_patient_ids.add(row["PATIENT"])

    rows = []
    for p in patients:
        rows.append({
            "id": p["Id"],
            "age": str(_age_at_reference(p["BIRTHDATE"])),
            "is_female": "1" if p["GENDER"] == "F" else "0",
            "race": p["RACE"],
            "ethnicity": p["ETHNICITY"],
            "healthcare_expenses": p["HEALTHCARE_EXPENSES"],
            "healthcare_coverage": p["HEALTHCARE_COVERAGE"],
            "binary_label": "1" if p["Id"] in pain_patient_ids else "0",
        })
    return rows


def _load_or_generate_enriched_dataset() -> list:
    if DATA_FILE.exists():
        with open(DATA_FILE, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        log.info("Loaded %d patients from %s", len(rows), DATA_FILE.name)
        return rows

    log.warning(
        "%s not found -- generating a %d-row sample from patients.csv/conditions.csv "
        "and saving it there (not the real engineered feature set).",
        DATA_FILE.name, _SAMPLE_SIZE,
    )
    rows = _generate_sample_dataset()
    with open(DATA_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved generated sample to %s", DATA_FILE)
    return rows


all_rows = _load_or_generate_enriched_dataset()
PATIENTS_SAMPLE = all_rows[:200]
log.info("Loaded %d patients (showing first %d)", len(all_rows), len(PATIENTS_SAMPLE))


def _compute_label_stats() -> dict:
    """Applies the keyword label to every patient in the raw CSVs -- the same
    rule the benchmark pipeline uses -- so the Dataset tab shows the real
    class balance instead of the small sample's."""
    with open(DATA_DIR / "patients.csv", encoding="utf-8", newline="") as f:
        patient_ids = {row["Id"] for row in csv.DictReader(f)}
    positive_ids = set()
    with open(DATA_DIR / "conditions.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _PAIN_PATTERN.search(row["DESCRIPTION"].lower()):
                positive_ids.add(row["PATIENT"])
    n_total = len(patient_ids)
    n_pos = len(positive_ids & patient_ids)
    return {
        "n_total":    n_total,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "pos_rate":   round(n_pos / n_total * 100, 1),
        "keywords":   PAIN_KEYWORDS,
    }


DATASET_STATS = _compute_label_stats()
log.info("Labels: %d positive / %d negative (%.1f%% positive)",
         DATASET_STATS["n_positive"], DATASET_STATS["n_negative"], DATASET_STATS["pos_rate"])

# ---------------------------------------------------------------------------
# Load the first rows of every raw Synthea CSV and profile each column (EDA)
# ---------------------------------------------------------------------------
RAW_FILES = [
    "patients", "encounters", "conditions", "medications", "observations",
    "procedures", "immunizations", "allergies", "careplans", "devices",
    "imaging_studies", "supplies", "providers", "organizations", "payers",
    "payer_transitions",
]
RAW_PREVIEW_ROWS = 25

# Identifier columns, and the file each one points to.
_ID_REFS = {
    "PATIENT": "patients", "ENCOUNTER": "encounters", "ORGANIZATION": "organizations",
    "PROVIDER": "providers", "PAYER": "payers",
}
# Number-like columns that are really labels (codes, ZIPs, phone numbers).
_CATEGORICAL_COLS = {
    "CODE", "REASONCODE", "ZIP", "SSN", "PHONE", "BODYSITE_CODE",
    "MODALITY_CODE", "SOP_CODE", "UDI", "DRIVERS", "PASSPORT",
}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def _profile_column(name: str, counts: Counter, n_rows: int, file_key: str) -> dict:
    """Summarises one column from its value counts: missing share, distinct
    values, and a kind-specific summary (date range, numeric spread, or the
    most common values)."""
    filled = {v: c for v, c in counts.items() if v != ""}
    n_filled = sum(filled.values())
    prof = {"name": name, "missing": n_rows - n_filled, "unique": len(filled)}

    if name == "Id" or name in _ID_REFS:
        prof.update(kind="id", ref=file_key if name == "Id" else _ID_REFS[name])
        return prof
    if not filled:
        prof["kind"] = "empty"
        return prof
    if all(_DATE_RE.match(v) for v in filled):
        prof.update(kind="date", min=min(filled)[:10], max=max(filled)[:10])
        return prof

    if name not in _CATEGORICAL_COLS:
        numbers = []
        for value, count in filled.items():
            number = _parse_float(value)
            if number is not None:
                numbers.append((number, count))
        n_numeric = sum(count for _, count in numbers)
        if n_numeric >= 0.5 * n_filled:
            numbers.sort()
            seen, median = 0, numbers[-1][0]
            for number, count in numbers:
                seen += count
                if seen >= n_numeric / 2:
                    median = number
                    break
            prof.update(kind="numeric", min=numbers[0][0], median=median, max=numbers[-1][0])
            if n_numeric < n_filled:   # e.g. observations.VALUE mixes numbers and text
                text_counts = Counter({v: c for v, c in filled.items() if _parse_float(v) is None})
                prof["text_rows"] = n_filled - n_numeric
                prof["top"] = [[v, c] for v, c in text_counts.most_common(3)]
            return prof

    prof.update(kind="category", top=[[v, c] for v, c in Counter(filled).most_common(3)])
    return prof


# EDA charts: the birth-date column behind "patients by birth year", and the column
# behind "most common values" (payers is small enough to chart a value per insurer).
_EDA_YEAR_COLUMN = {"patients": "BIRTHDATE"}
_EDA_TOP_COLUMN = {
    "patients": ["RACE", "COUNTY"], "encounters": "ENCOUNTERCLASS", "conditions": "DESCRIPTION",
    "medications": "DESCRIPTION", "observations": "DESCRIPTION", "procedures": "DESCRIPTION",
    "immunizations": "DESCRIPTION", "allergies": "DESCRIPTION", "careplans": "DESCRIPTION",
    "devices": "DESCRIPTION", "imaging_studies": "BODYSITE_DESCRIPTION", "supplies": "DESCRIPTION",
    "providers": "SPECIALITY", "organizations": "CITY", "payer_transitions": "OWNERSHIP",
}
_EDA_VALUE_BY = {"payers": ("NAME", "UNIQUE_CUSTOMERS")}
_PER_PATIENT_BINS = [   # (low, high, label); high=None means open-ended
    (0, 0, "0"), (1, 1, "1"), (2, 4, "2–4"), (5, 9, "5–9"), (10, 24, "10–24"),
    (25, 49, "25–49"), (50, 99, "50–99"), (100, 249, "100–249"),
    (250, 499, "250–499"), (500, None, "500+"),
]


def _eda_charts(key: str, cols: list, counters: list, value_pairs: list) -> dict:
    """Builds the data behind each file's EDA charts from its value counts."""
    by_name = dict(zip(cols, counters))
    eda = {}

    year_col = _EDA_YEAR_COLUMN.get(key)
    if year_col in by_name:
        years = Counter()
        for value, count in by_name[year_col].items():
            if value:
                years[int(value[:4])] += count
        eda["by_year"] = {"column": year_col, "counts": sorted(years.items())}

    if "PATIENT" in by_name:
        per_patient = [count for value, count in by_name["PATIENT"].items() if value]
        bins = [0] * len(_PER_PATIENT_BINS)
        bins[0] = DATASET_STATS["n_total"] - len(per_patient)   # patients with no rows here
        for n in per_patient:
            for i, (low, high, _) in enumerate(_PER_PATIENT_BINS[1:], start=1):
                if n >= low and (high is None or n <= high):
                    bins[i] += 1
                    break
        eda["per_patient"] = [[label, n] for (_, _, label), n in zip(_PER_PATIENT_BINS, bins)]

    top_cols = _EDA_TOP_COLUMN.get(key, [])
    if isinstance(top_cols, str):
        top_cols = [top_cols]
    eda["top"] = []
    for top_col in top_cols:
        filled = Counter({v: c for v, c in by_name[top_col].items() if v})
        eda["top"].append({"column": top_col, "values": filled.most_common(10), "total": sum(filled.values())})
    if value_pairs:
        name_col, value_col = _EDA_VALUE_BY[key]
        eda["value_by"] = {"name": name_col, "value": value_col,
                           "values": sorted(value_pairs, key=lambda p: -p[1])}
    return eda


def _load_raw_file(key: str) -> dict:
    """Reads one raw CSV in a single pass: keeps the first rows for the
    preview grid and counts every value for the column profile and charts."""
    value_by = _EDA_VALUE_BY.get(key)
    with open(RAW_DIR / f"{key}.csv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        cols = next(reader)
        counters = [Counter() for _ in cols]
        rows, total, value_pairs = [], 0, []
        for row in reader:
            total += 1
            if len(rows) < RAW_PREVIEW_ROWS:
                rows.append(dict(zip(cols, row)))
            for counter, value in zip(counters, row):
                counter[value] += 1
            if value_by:
                record = dict(zip(cols, row))
                value_pairs.append([record[value_by[0]], float(record[value_by[1]] or 0)])
    profile = [_profile_column(c, counts, total, key) for c, counts in zip(cols, counters)]
    eda = _eda_charts(key, cols, counters, value_pairs)
    return {"cols": cols, "rows": rows, "total": total, "profile": profile, "eda": eda}


RAW_DATA = {}   # {key: {"cols": [...], "rows": [...], "total": N, "profile": [...]}}

for key in RAW_FILES:
    try:
        RAW_DATA[key] = _load_raw_file(key)
        log.info("Raw %-22s  %8d rows  %2d cols", f"{key}.csv", RAW_DATA[key]["total"], len(RAW_DATA[key]["cols"]))
    except Exception as exc:
        log.error("Could not load %s.csv: %s", key, exc)
        RAW_DATA[key] = {"cols": [], "rows": [], "total": 0, "profile": []}

# ---------------------------------------------------------------------------
# My Model — loaded fresh on every request so edits to the CSV are reflected
# on page refresh without restarting the server.
# ---------------------------------------------------------------------------
# One row per iteration (5 rows). Performance columns default to 0.0 in the
# template; computational-cost columns (named like ai-sandbox's ResourceLogger)
# default to null, meaning "not reported".
_MY_MODEL_METRIC_MAP = {
    "MacroAUC":   "auc",
    "Accuracy":   "accuracy",
    "F1_1":       "f1",
    "Precision_1":"precision",
    "Recall_1":   "recall",
}
_MY_MODEL_COST_COLS = ["elapsed_hr", "gpu_hrs", "cpu_pct"]
_MY_MODEL_CSV = ROOT / "my_model.csv"
_MISSING_VALUES = {"", "null", "none", "na", "nan"}


def load_my_model_stats():
    """Averages the student's per-iteration results file. Metrics with no
    values stay 0.0; cost columns with no values become None (not reported)."""
    columns = {**_MY_MODEL_METRIC_MAP, **{c: c for c in _MY_MODEL_COST_COLS}}
    values = {key: [] for key in columns.values()}
    n_iterations = 0
    try:
        with open(_MY_MODEL_CSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                n_iterations += 1
                for col, key in columns.items():
                    raw = (row.get(col) or "").strip()
                    if raw.lower() in _MISSING_VALUES:
                        continue
                    try:
                        values[key].append(float(raw))
                    except ValueError:
                        log.warning("my_model CSV: ignoring non-numeric %s=%r in iteration %s",
                                    col, raw, row.get("iteration"))
    except Exception as exc:
        log.warning("Could not load my_model CSV (%s): %s — using defaults", _MY_MODEL_CSV, exc)

    mean = lambda vals: sum(vals) / len(vals) if vals else None
    result = {key: round(mean(values[key]) or 0.0, 4) for key in _MY_MODEL_METRIC_MAP.values()}
    result.update({key: mean(values[key]) for key in _MY_MODEL_COST_COLS})
    result["n_iterations"] = n_iterations
    log.info("My Model  F1=%.4f  AUC=%.4f  over %d iterations (from %s)",
             result["f1"], result["auc"], n_iterations, _MY_MODEL_CSV.name)
    return result

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    from markupsafe import Markup
    return render_template_string(
        HTML_TEMPLATE,
        stats=Markup(json.dumps(DATASET_STATS)),
        raw=Markup(json.dumps(RAW_DATA)),
        my_model=Markup(json.dumps(load_my_model_stats())),
    )


@app.route("/api/patients")
def api_patients():
    return jsonify(PATIENTS_SAMPLE)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI-Sandbox — Final Project Explorer</title>
<style>
/* ── Reset & base ── */
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f0e8;color:#1e1208;min-height:100vh}
a{color:inherit;text-decoration:none}

/* ── Layout ── */
#app-header{background:#2e1f10;color:#f5f0e8;padding:0 24px;display:flex;align-items:center;gap:16px;height:56px;border-bottom:3px solid #7c4f2f}
#app-header .logo{font-size:1.16rem;font-weight:800;letter-spacing:.5px}
#app-header .subtitle{font-size:0.84rem;color:#c4a882;font-weight:500}
#app-header .badge{background:#7c4f2f;color:#f5f0e8;font-size:0.75rem;padding:2px 8px;border-radius:20px;margin-left:auto;font-weight:700;letter-spacing:.5px}

/* ── Tab nav ── */
#tab-nav{background:#3d2a18;display:flex;gap:0;border-bottom:2px solid #5c3d22}
.tab-btn{padding:12px 22px;font-size:0.86rem;font-weight:700;color:#c4a882;cursor:pointer;border:none;background:none;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .15s;letter-spacing:.3px;text-transform:uppercase}
.tab-btn:hover{color:#f5f0e8;background:rgba(255,255,255,.05)}
.tab-btn.active{color:#f5f0e8;border-bottom-color:#e8a250}
.tab-btn.tab-deliverables{color:#e8a250}
.tab-btn.tab-deliverables:hover{color:#f5c27a}
.tab-btn.tab-deliverables.active{color:#ffcf8a}
.tab-btn .tab-sub{text-transform:none;letter-spacing:0;font-weight:600}
.tab-btn .tab-badge{display:inline-flex;align-items:center;justify-content:center;background:#e8a250;color:#2e1f10;border-radius:50%;width:16px;height:16px;font-size:0.69rem;font-weight:900;margin-left:5px}

/* ── Page content ── */
#page{max-width:1440px;margin:0 auto;padding:28px clamp(36px,6.75vw,108px)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* ── Section titles ── */
.section-title{font-size:1.26rem;font-weight:800;color:#2e1f10;margin-bottom:6px}
.section-sub{font-size:0.89rem;color:#7a5c3c;margin-bottom:22px;line-height:1.5}
.section-header{font-size:1.15rem;font-weight:800;color:#2e1f10;margin-bottom:6px}

/* ── Cards ── */
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);border:1px solid #e8ddd0}
.card-grid{display:grid;gap:16px}
.card-grid.cols-3{grid-template-columns:repeat(3,1fr)}
.card-grid.cols-4{grid-template-columns:repeat(4,1fr)}
.card-grid.cols-2{grid-template-columns:repeat(2,1fr)}

/* ── Stat cards ── */
.stat-card{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 6px rgba(0,0,0,.07);border:1px solid #e8ddd0;text-align:center}
.stat-card .val{font-size:2rem;font-weight:900;color:#2e1f10;line-height:1}
.stat-card .lbl{font-size:0.78rem;color:#7a5c3c;font-weight:600;margin-top:5px;text-transform:uppercase;letter-spacing:.5px}

/* ── Problem tab ── */
.problem-hero{background:linear-gradient(135deg,#2e1f10 0%,#5c3d22 100%);color:#f5f0e8;border-radius:14px;padding:36px;margin-bottom:24px}
.problem-hero h1{font-size:1.6rem;font-weight:900;margin-bottom:10px}
.problem-hero p{font-size:0.95rem;color:#d4b896;line-height:1.7;max-width:1000px}
.pipeline-steps{display:flex;gap:0;margin:20px 0}
.pipeline-step{flex:1;background:#fff;border:1px solid #e8ddd0;border-radius:8px;padding:14px 16px;position:relative;text-align:center}
.pipeline-step+.pipeline-step::before{content:"→";position:absolute;left:-14px;top:50%;transform:translateY(-50%);color:#b8996c;font-size:1.16rem;font-weight:900;z-index:1}
.pipeline-step .step-icon{font-size:1.5rem;margin-bottom:6px}
.pipeline-step .step-title{font-size:0.81rem;font-weight:800;color:#2e1f10;text-transform:uppercase;letter-spacing:.4px}
.pipeline-step .step-desc{font-size:0.78rem;color:#7a5c3c;margin-top:3px;line-height:1.4}
.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.78rem;font-weight:700;margin:2px}
.pill.cpu{background:#dcfce7;color:#15803d}
.pill.gpu{background:#ede9fe;color:#5b21b6}
.pill.bert{background:#dbeafe;color:#1d4ed8}

/* ── Dataset tab ── */
.table-wrap{overflow-x:auto;border-radius:10px;border:1px solid #e8ddd0;background:#fff;margin-top:16px}
table{width:100%;border-collapse:collapse;font-size:0.81rem}
thead tr{background:#3d2a18;color:#f5f0e8}
thead th{padding:10px 12px;font-weight:700;white-space:nowrap;text-align:left;font-size:0.78rem;letter-spacing:.3px}
tbody tr:nth-child(even){background:#faf7f2}
tbody tr:hover{background:#f0e8d8}
tbody td{padding:8px 12px;border-bottom:1px solid #f0e8d8;white-space:nowrap;color:#2e1f10}
.label-badge-1{display:inline-block;background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-weight:700;font-size:0.75rem}
.label-badge-0{display:inline-block;background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-weight:700;font-size:0.75rem}
.table-pager{display:flex;align-items:center;gap:10px;padding:12px 16px;background:#faf7f2;border-top:1px solid #e8ddd0;font-size:0.81rem;color:#7a5c3c}
.pager-btn{background:#3d2a18;color:#f5f0e8;border:none;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:0.81rem;font-weight:700}
.pager-btn:disabled{opacity:.4;cursor:not-allowed}

/* ── Raw data sub-tabs ── */
.raw-tab-btn{padding:6px 12px;font-size:0.78rem;font-weight:700;color:#7a5c3c;cursor:pointer;border:1px solid #e8ddd0;background:#fff;border-radius:20px;transition:all .15s;white-space:nowrap}
.raw-tab-btn:hover{color:#2e1f10;background:#f0e8d8}
.raw-tab-btn.active{color:#f5f0e8;background:#3d2a18;border-color:#3d2a18}
.raw-tab-btn .rtb-count{font-size:0.69rem;color:#86674a;font-weight:500;margin-left:4px}
.raw-tab-btn.active .rtb-count{color:#c4a882}
.raw-panel{display:none}
.raw-panel.active{display:block}
.raw-file-meta{background:#fdf8f2;border:1px solid #e8ddd0;border-bottom:none;border-radius:10px 10px 0 0;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.raw-file-meta .rfm-name{font-size:0.86rem;font-weight:800;color:#2e1f10}
.raw-file-meta .rfm-desc{font-size:0.81rem;color:#6b5440;flex-basis:100%}
.raw-file-meta .rfm-count{font-size:0.78rem;color:#86674a;margin-left:auto;white-space:nowrap}
.raw-col-count{font-size:0.75rem;background:#e8ddd0;color:#5c3d22;border-radius:4px;padding:2px 6px;font-weight:700;flex-shrink:0}
.raw-table-wrap{overflow-x:auto;border-radius:0 0 10px 10px;border:1px solid #e8ddd0;background:#fff}
.raw-table-wrap table{width:100%;border-collapse:collapse;font-size:0.78rem}
.raw-table-wrap thead tr{background:#3d2a18;color:#f5f0e8}
.raw-table-wrap thead th{padding:8px 12px;font-weight:700;white-space:nowrap;text-align:left;font-size:0.75rem;letter-spacing:.2px}
.raw-table-wrap tbody tr:nth-child(even){background:#faf7f2}
.raw-table-wrap tbody tr:hover{background:#f0e8d8}
.raw-table-wrap tbody td{padding:7px 12px;border-bottom:1px solid #f5f0e8;white-space:nowrap;color:#2e1f10;max-width:220px;overflow:hidden;text-overflow:ellipsis}
.raw-table-wrap tbody td:first-child{color:#86674a;font-size:0.71rem}
.raw-table-wrap tbody td.pid{max-width:none;overflow:visible;font-family:monospace;font-size:0.76rem}
.col-guide tbody td{white-space:normal}
.eda-title{font-size:0.86rem;font-weight:800;color:#2e1f10;margin:18px 0 8px}
.kind-chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.71rem;font-weight:700;white-space:nowrap}
.kind-id{background:#ede9fe;color:#5b21b6}
.kind-date{background:#ccfbf1;color:#0f766e}
.kind-numeric{background:#dbeafe;color:#1e40af}
.kind-category{background:#fef3c7;color:#92400e}
.kind-empty{background:#f0e8d8;color:#7a5c3c}
/* EDA charts: one series per chart in #b45309 (passes lightness, chroma, and 3:1 contrast on white) */
.eda-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.eda-card{background:#fff;border:1px solid #e8ddd0;border-radius:10px;padding:14px 16px;min-width:0}
.eda-card.wide{grid-column:1/-1}
.eda-card-title{font-size:0.86rem;font-weight:800;color:#2e1f10;margin-bottom:3px}
.eda-card-sub{font-size:0.78rem;color:#7a5c3c;line-height:1.5;margin-bottom:12px}
.eda-plot{display:flex;align-items:stretch;gap:2px;height:150px;border-bottom:1px solid #e8ddd0;padding-top:16px}
.eda-col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;min-width:0;cursor:default}
.eda-bar{width:100%;max-width:24px;background:#b45309;border-radius:4px 4px 0 0}
.eda-col:hover .eda-bar,.eda-hrow:hover .eda-hbar{opacity:.75}
.eda-cap{font-size:0.71rem;color:#5c3d22;font-weight:700;white-space:nowrap;line-height:1;margin-bottom:3px}
.eda-xaxis{position:relative;height:18px;margin-top:4px}
.eda-xaxis span{position:absolute;transform:translateX(-50%);font-size:0.69rem;color:#86674a;white-space:nowrap;font-variant-numeric:tabular-nums}
.eda-hrow{display:flex;align-items:center;gap:10px;padding:3px 0}
.eda-hlabel{width:42%;flex-shrink:0;font-size:0.78rem;color:#2e1f10;line-height:1.3;overflow-wrap:anywhere}
.eda-hplot{flex:1;display:flex;align-items:center;gap:6px;min-width:0}
.eda-hbar{height:12px;background:#b45309;border-radius:0 4px 4px 0;min-width:1px}
.eda-hval{font-size:0.76rem;color:#5c3d22;font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}
.eda-table{margin-top:8px;font-size:0.78rem;color:#7a5c3c}
.eda-table summary{cursor:pointer;font-weight:700}
.eda-table div{max-height:200px;overflow-y:auto;margin-top:6px;border:1px solid #e8ddd0;border-radius:6px}
.eda-table table{font-size:0.78rem}
.eda-table td{white-space:nowrap}
.eda-tip{position:fixed;display:none;z-index:50;pointer-events:none;background:#2e1f10;color:#f5f0e8;border-radius:6px;padding:6px 10px;font-size:0.78rem;line-height:1.4;max-width:260px;box-shadow:0 4px 12px rgba(0,0,0,.2)}
.eda-tip strong{font-size:0.85rem}
.eda-tip div{color:#d4b896}

/* ── Dataset tab: labels ── */
.ds-text{font-size:0.87rem;color:#6b5440;line-height:1.6;margin-bottom:8px}
.ds-text:last-child{margin-bottom:0}
.ds-card-title{font-weight:800;font-size:0.89rem;color:#2e1f10;margin-bottom:8px}
.ds-note{font-size:0.78rem;color:#86674a;line-height:1.5;margin-top:8px}
.ds-legend{display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;font-size:0.83rem;color:#2e1f10}
.ds-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle}
.ds-age-row{display:flex;align-items:center;gap:10px;font-size:0.81rem;color:#2e1f10;margin-top:6px}
.ds-age-row span:first-child{width:80px;flex-shrink:0}
.ds-age-row span:last-child{width:42px;text-align:right;font-weight:700}
.ds-age-track{flex:1;height:12px;background:#f0e8d8;border-radius:4px;overflow:hidden}
.ds-age-track div{height:100%;background:#b45309}

/* ── Feature table ── */
.src-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;letter-spacing:.2px}
.src-patients{background:#fef3c7;color:#92400e}
.src-conditions{background:#ede9fe;color:#5b21b6}
.src-encounters{background:#ccfbf1;color:#0f766e}
.src-medications{background:#ffedd5;color:#c2410c}
.src-procedures{background:#dbeafe;color:#1e40af}
.src-observations{background:#dcfce7;color:#166534}
.feat-table-wrap{max-height:460px;overflow-y:auto;border-radius:10px;border:1px solid #e8ddd0;background:#fff}
.feat-table{width:100%;border-collapse:collapse;font-size:0.81rem}
.feat-table thead{position:sticky;top:0;z-index:2}
.feat-table thead tr{background:#3d2a18;color:#f5f0e8}
.feat-table thead th{padding:9px 14px;font-weight:700;text-align:left;font-size:0.77rem;letter-spacing:.3px;white-space:nowrap}
.feat-table tbody tr:nth-child(even){background:#faf7f2}
.feat-table tbody tr:hover{background:#f0e8d8}
.feat-table tbody td{padding:8px 14px;border-bottom:1px solid #f5f0e8;vertical-align:top;color:#2e1f10;line-height:1.45}
.feat-table tbody td:first-child{color:#86674a;font-size:0.75rem;text-align:center}
.feat-name{font-family:monospace;font-size:0.81rem;font-weight:700;color:#1e1208;background:#f0e8d8;padding:2px 6px;border-radius:4px;white-space:nowrap}
.feat-group-row td{background:#f0e8d8!important;font-size:0.77rem;font-weight:800;color:#5c3d22;text-transform:uppercase;letter-spacing:.5px;padding:5px 14px!important}
.feature-group{background:#fff;border-radius:10px;padding:16px 18px;border:1px solid #e8ddd0;margin-bottom:12px}
.feature-group .fg-title{font-size:0.86rem;font-weight:800;color:#2e1f10;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.feature-group .fg-tags{display:flex;flex-wrap:wrap;gap:5px}
.feature-tag{background:#f0e8d8;color:#5c3d22;border-radius:6px;padding:3px 9px;font-size:0.75rem;font-weight:600}

/* ── Feature engineering tab ── */
.pipeline-flow{display:flex;flex-direction:column;gap:12px;margin:16px 0}
.flow-row{display:flex;align-items:stretch;gap:12px}
.flow-node{background:#fff;border:2px solid #e8ddd0;border-radius:10px;padding:14px 18px;flex:1;position:relative}
.flow-node.source{border-color:#b8996c;background:#fdf8f2}
.flow-node.process{border-color:#7c3aed;background:#faf5ff}
.flow-node.output{border-color:#16a34a;background:#f0fdf4}
.flow-node .fn-label{font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.flow-node.source .fn-label{color:#b8996c}
.flow-node.process .fn-label{color:#7c3aed}
.flow-node.output .fn-label{color:#16a34a}
.flow-node .fn-title{font-size:0.92rem;font-weight:800;color:#1e1208;margin-bottom:4px}
.flow-node .fn-desc{font-size:0.81rem;color:#6b5440;line-height:1.5}
.flow-arrow{display:flex;align-items:center;justify-content:center;font-size:1.4rem;color:#b8996c;font-weight:900;flex-shrink:0;width:28px}
.model-uses{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.model-use-tag{font-size:0.71rem;font-weight:700;padding:2px 7px;border-radius:12px}
.mtu-cpu{background:#dcfce7;color:#15803d}
.mtu-gpu{background:#ede9fe;color:#5b21b6}
.mtu-bert{background:#dbeafe;color:#1d4ed8}
.fe-files-row{display:flex;align-items:flex-start;gap:12px;margin-top:10px}
.fe-files-lbl{width:150px;flex-shrink:0;font-size:0.81rem;font-weight:700;color:#5c3d22;padding-top:3px}
.fe-chips{display:flex;flex-wrap:wrap;gap:5px}
.fe-chip{border-radius:6px;padding:3px 9px;font-size:0.75rem;font-weight:700}
.fe-chip-used{background:#dcfce7;color:#15803d}
.fe-chip-label{background:#fef3c7;color:#92400e}
.fe-models td,.fe-table tbody td{white-space:normal}
.fe-table .model-use-tag{white-space:nowrap;display:inline-block;margin:1px 0}
.fe-models td{font-size:0.83rem;line-height:1.55}

/* ── Model cards ── */
.models-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-top:16px}
.model-card{background:#fff;border-radius:14px;padding:20px;border:2px solid #e8ddd0;cursor:pointer;transition:all .2s;position:relative;user-select:none}
.model-card:hover{border-color:#b8996c;box-shadow:0 4px 16px rgba(0,0,0,.1)}
.model-card.selected{border-color:#e8a250;box-shadow:0 0 0 3px rgba(232,162,80,.25)}
.model-card .mc-check{position:absolute;top:14px;right:14px;width:22px;height:22px;border-radius:50%;border:2px solid #e8ddd0;display:flex;align-items:center;justify-content:center;font-size:0.84rem;transition:all .2s;background:#fff}
.model-card.selected .mc-check{background:#e8a250;border-color:#e8a250;color:#fff}
.model-card .mc-icon{font-size:2rem;margin-bottom:10px}
.model-card .mc-category{font-size:0.71rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;padding:2px 8px;border-radius:12px;display:inline-block}
.mc-cat-cpu{background:#dcfce7;color:#15803d}
.mc-cat-gpu{background:#ede9fe;color:#5b21b6}
.mc-cat-bert{background:#dbeafe;color:#1d4ed8}
.mc-cat-mymodel{background:#fee2e2;color:#991b1b}
.model-section-hdr{font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.8px;color:#7a5c3c;margin:22px 0 10px;display:flex;align-items:center;gap:10px}
.model-section-hdr::after{content:'';flex:1;height:1px;background:#e8ddd0}
.model-card.locked-card{opacity:.45;pointer-events:none;filter:grayscale(.6)}
.tier-lock-banner{background:linear-gradient(135deg,#1e1208 0%,#3d2a18 100%);border-radius:12px;padding:18px 22px;margin:4px 0 18px;display:flex;align-items:center;gap:16px;border:1px solid #6b4c2e}
.tlb-icon{font-size:2rem;flex-shrink:0}
.tlb-body{flex:1}
.tlb-title{font-size:0.97rem;font-weight:800;color:#f5f0e8;margin-bottom:4px}
.tlb-sub{font-size:0.84rem;color:#b8996c;line-height:1.5}
.tlb-need{font-weight:800;color:#e8a250}
.tlb-got{font-weight:800;color:#dc2626}
.tlb-badge{background:#e8a250;color:#2e1f10;font-weight:900;font-size:0.75rem;padding:4px 12px;border-radius:20px;flex-shrink:0}
.tlb-badge.locked{background:#4b3320;color:#b8996c}
.model-card .mc-name{font-size:1.05rem;font-weight:800;color:#1e1208;margin-bottom:6px}
.model-card .mc-auc{font-size:1.6rem;font-weight:900;margin-bottom:4px}
.model-card .mc-auc-lbl{font-size:0.71rem;font-weight:600;color:#7a5c3c;text-transform:uppercase;letter-spacing:.4px}
.model-card .mc-desc{font-size:0.81rem;color:#6b5440;line-height:1.5;margin-top:8px;border-top:1px solid #f0e8d8;padding-top:8px}
.model-card .mc-input{font-size:0.75rem;color:#86674a;margin-top:6px;font-style:italic}
.compare-bar{background:#3d2a18;color:#f5f0e8;padding:12px 20px;display:flex;align-items:center;gap:14px;border-radius:10px;margin-bottom:20px;flex-wrap:wrap}
.compare-bar .cb-title{font-size:0.86rem;font-weight:700}
.selected-pills{display:flex;gap:6px;flex-wrap:wrap;flex:1}
.selected-pill{display:flex;align-items:center;gap:4px;background:rgba(255,255,255,.15);padding:4px 10px;border-radius:20px;font-size:0.78rem;font-weight:600}
.selected-pill .sp-remove{cursor:pointer;opacity:.7;font-size:0.86rem;margin-left:2px}
.selected-pill .sp-remove:hover{opacity:1}
.compare-go-btn{background:#e8a250;color:#2e1f10;border:none;padding:8px 18px;border-radius:8px;font-weight:800;font-size:0.86rem;cursor:pointer;flex-shrink:0}
.compare-go-btn:hover{background:#f0b864}
.compare-go-btn:disabled{opacity:.4;cursor:not-allowed}
.dl-splits td:first-child{font-weight:700}
.dl-total td{font-weight:800;background:#faf7f2}
.dl-share{display:flex;align-items:center;gap:8px}
.dl-share-bar{height:12px;background:#b45309;border-radius:0 4px 4px 0}
.dl-share span{font-size:0.81rem;font-weight:700;color:#5c3d22;font-variant-numeric:tabular-nums}
.dl-steps{margin:0 0 12px;padding-left:20px;font-size:0.87rem;color:#2e1f10;line-height:1.8}
.dl-report-chip{display:inline-block;vertical-align:middle;margin-left:8px;padding:2px 9px;border-radius:10px;background:#ede9fe;color:#5b21b6;font-size:0.7rem;font-weight:700;letter-spacing:.2px}
.dl-placeholder{opacity:.6;border-style:dashed}
.dl-columns{margin:10px 0 14px}
.dl-columns td{white-space:normal}
.dl-checklist{list-style:none;margin:0;padding:0;font-size:0.86rem;color:#2e1f10;line-height:2}
.dl-checklist li::before{content:"☐";margin-right:8px;color:#b45309;font-weight:800}
.mm-upload{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:6px 0 8px}
.mm-btn{display:inline-block;background:#fff;color:#3d2a18;border:1px solid #d6c3a8;padding:7px 14px;border-radius:8px;font-weight:700;font-size:0.83rem;cursor:pointer;font-family:inherit}
.mm-btn:hover{background:#f0e8d8}
.mm-btn-primary{background:#3d2a18;color:#f5f0e8;border-color:#3d2a18}
.mm-btn-primary:hover{background:#5c3d22}
.mm-upload-status{flex:1;min-width:220px;font-size:0.81rem;color:#6b5440;line-height:1.5}
.mm-upload-status.mm-error{flex-basis:100%;color:#991b1b;background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;border-radius:8px;padding:10px 14px}
.mm-error-title{font-weight:700;margin-bottom:4px}
.mm-upload-status.mm-error ul{margin:0;padding-left:18px;line-height:1.6}
.mm-link{background:none;border:none;padding:0;color:#b45309;font-weight:700;text-decoration:underline;cursor:pointer;font-size:inherit;font-family:inherit}
.goal-chip{display:block;width:fit-content;max-width:100%;margin-top:9px;background:rgba(232,162,80,.16);border:1px solid rgba(232,162,80,.75);border-radius:8px;padding:7px 13px;color:#fde7c2;font-size:.86rem;line-height:1.5}
.goal-chip .goal-num{color:#ffb85c;font-weight:800}
.goal-chip.reached{background:rgba(74,222,128,.14);border-color:rgba(74,222,128,.6);color:#dcfce7}
.goal-chip.reached .goal-num{color:#4ade80}
.mission-callout{display:flex;align-items:flex-start;gap:12px;background:linear-gradient(135deg,#fff7ed 0%,#fef3c7 100%);border:1px solid #f5d6a8;border-left:4px solid #e8a250;border-radius:10px;padding:14px 18px;margin:4px 0 14px;box-shadow:0 2px 8px rgba(232,162,80,.12)}
.mission-icon{font-size:1.3rem;line-height:1.3;flex-shrink:0}
.mission-text{font-size:0.89rem;color:#5c3d22;line-height:1.65}
.mission-hl{background:linear-gradient(transparent 58%,#fcd9a0 58%);padding:0 2px;color:#3d2a18}
.compare-tip{display:flex;align-items:center;gap:14px;background:#fff7ed;border:1px solid #fdba74;border-left:4px solid #e8a250;border-radius:10px;padding:14px 18px;margin-bottom:20px;flex-wrap:wrap}
.compare-tip-icon{font-size:1.5rem;flex-shrink:0}
.compare-tip-body{flex:1;min-width:220px}
.compare-tip-title{font-size:0.92rem;font-weight:800;color:#2e1f10;margin-bottom:2px}
.compare-tip-text{font-size:0.85rem;color:#6b5440;line-height:1.5}

/* ── Compare tab ── */
#compare-empty{text-align:center;padding:60px 20px;color:#86674a}
#compare-empty .ce-icon{font-size:3rem;margin-bottom:14px}
#compare-empty h3{font-size:1.05rem;font-weight:700;margin-bottom:6px}
#compare-empty p{font-size:0.86rem;line-height:1.5}
#compare-content{display:none}
.compare-section{background:#fff;border-radius:12px;padding:22px;border:1px solid #e8ddd0;margin-bottom:20px}
.compare-section-title{font-size:0.92rem;font-weight:800;color:#2e1f10;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #f0e8d8;display:flex;align-items:center;gap:8px}
.metric-tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.metric-tab{padding:5px 14px;border-radius:20px;font-size:0.81rem;font-weight:700;cursor:pointer;border:2px solid #e8ddd0;background:#fff;color:#6b5440;transition:all .15s}
.metric-tab.active{background:#3d2a18;color:#f5f0e8;border-color:#3d2a18}
#chart-svg-wrap{width:100%;min-height:200px}
.comp-table{width:100%;border-collapse:collapse;font-size:0.85rem}
.comp-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;text-align:left;color:#3d2a18;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.comp-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10;vertical-align:top}
.comp-table tr:last-child td{border-bottom:none}
.comp-table tr:hover td{background:#faf7f2}
.yes-chip{display:inline-flex;align-items:center;gap:4px;background:#dcfce7;color:#15803d;border-radius:6px;padding:2px 8px;font-size:0.78rem;font-weight:700}
.no-chip{display:inline-flex;align-items:center;gap:4px;background:#fee2e2;color:#b91c1c;border-radius:6px;padding:2px 8px;font-size:0.78rem;font-weight:700}
.na-chip{display:inline-flex;align-items:center;background:#f0e8d8;color:#7a5c3c;border-radius:6px;padding:2px 8px;font-size:0.78rem;font-weight:700}
.pros-cons-grid{display:grid;gap:16px}
.pc-model-title{font-size:0.86rem;font-weight:800;color:#1e1208;padding:10px 14px;border-radius:8px 8px 0 0;margin-bottom:0}
.pro-item,.con-item{display:flex;align-items:flex-start;gap:8px;font-size:0.84rem;line-height:1.5;padding:6px 0;border-bottom:1px solid #f5f0e8;color:#2e1f10}
.pro-item:last-child,.con-item:last-child{border-bottom:none}
.pro-dot{color:#16a34a;font-weight:900;flex-shrink:0;font-size:0.97rem}
.con-dot{color:#dc2626;font-weight:900;flex-shrink:0;font-size:0.97rem}
.pc-section-label{font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;padding:6px 0 4px;color:#7a5c3c}
.model-color-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;display:inline-block;margin-right:4px}

/* ── Reproducibility tab ── */
.repro-hero{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:28px}
.repro-hero-left{background:#2e1f10;border-radius:14px;padding:26px 28px;display:flex;flex-direction:column;justify-content:center}
.repro-hero-eyebrow{font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:#7a5c3c;margin-bottom:6px}
.repro-hero-title{font-size:2rem;font-weight:900;color:#e8a250;margin-bottom:12px;line-height:1}
.repro-hero-body{font-size:0.87rem;color:#c4a882;line-height:1.75;margin:0}
.repro-hero-right{display:flex;flex-direction:column;gap:10px}
.repro-concept-card{background:#fff;border:1px solid #e8ddd0;border-radius:10px;padding:14px 16px;display:flex;align-items:flex-start;gap:12px}
.repro-concept-icon{font-size:1.3rem;flex-shrink:0;line-height:1;margin-top:1px}
.repro-concept-label{font-size:0.85rem;font-weight:800;color:#2e1f10;margin-bottom:3px}
.repro-concept-desc{font-size:0.78rem;color:#5c3d22;line-height:1.5}
/* Range chart */
#repro-range-chart{background:#fff;border:1px solid #e8ddd0;border-radius:12px;padding:20px 24px}
/* Slider */
.repro-slider-wrap{background:#fff;border:1px solid #e8ddd0;border-radius:12px;padding:20px 24px;margin-bottom:14px}
.repro-slider-top{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.repro-slider-label{font-size:0.81rem;font-weight:700;color:#86674a;text-transform:uppercase;letter-spacing:.4px}
.repro-iter-display{font-size:2rem;font-weight:900;color:#2e1f10;background:#f5f0e8;border-radius:8px;padding:4px 18px;min-width:70px;text-align:center;line-height:1.2}
.repro-slider{width:100%;accent-color:#e8a250;height:6px;cursor:pointer}
.repro-slider-ticks{display:flex;justify-content:space-between;font-size:0.75rem;color:#86674a;margin-top:6px;padding:0 2px}
/* Insight box */
.repro-insight-box{background:#faf7f2;border-left:4px solid #e8a250;border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:18px;font-size:0.87rem;color:#2e1f10;line-height:1.65;min-height:44px}
.repro-insight-winner{color:#b45309;font-weight:800}
.repro-insight-delta{font-weight:700}
.repro-insight-delta.up{color:#15803d}
.repro-insight-delta.down{color:#b91c1c}
/* Iteration table */
.repro-iter-table-wrap{overflow-x:auto;border-radius:12px;border:1px solid #e8ddd0}
.repro-iter-table{width:100%;border-collapse:collapse;font-size:0.84rem}
.repro-iter-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;color:#3d2a18;text-align:left;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.repro-iter-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10;white-space:nowrap}
.repro-iter-table tr:last-child td{border-bottom:none}
.repro-iter-table tr.iter-winner td{background:#fffbeb}
.repro-iter-table tr.iter-winner td:first-child{border-left:3px solid #e8a250}
.repro-winner-badge{background:#e8a250;color:#1e1208;font-weight:900;font-size:0.69rem;border-radius:10px;padding:2px 7px;margin-left:6px;text-transform:uppercase}
.repro-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle;flex-shrink:0}
.repro-winner-row{background:rgba(234,179,8,.08)!important}
.repro-winner-row td{font-weight:700}
.repro-crown{color:#f59e0b;margin-left:4px;font-size:0.92rem}
.repro-delta{font-size:0.78rem;font-weight:700}
.repro-delta.up{color:#15803d}
.repro-delta.down{color:#dc2626}
.repro-delta.flat{color:#86674a}
/* Summary table */
.repro-summary-wrap{overflow-x:auto;border-radius:12px;border:1px solid #e8ddd0;margin-bottom:24px}
.repro-summary-table{width:100%;border-collapse:collapse;font-size:0.84rem}
.repro-summary-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;color:#3d2a18;text-align:left;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.repro-summary-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10}
.repro-summary-table tr:last-child td{border-bottom:none}
.repro-summary-table tr:hover td{background:#faf7f2}
.repro-range-bar{display:inline-flex;align-items:center;gap:6px;font-size:0.78rem;font-weight:700}
.repro-range-pill{background:#f0e8d8;border-radius:4px;padding:1px 7px;color:#5c3d22}
/* Takeaway */
.repro-takeaway{background:#f5f0e8;border-radius:12px;padding:22px 24px}
.repro-takeaway-title{font-size:0.81rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#7a5c3c;margin-bottom:16px}
.repro-takeaway-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.repro-takeaway-item{display:flex;gap:12px;align-items:flex-start;font-size:0.87rem;color:#2e1f10;line-height:1.65}
.repro-takeaway-num{background:#e8a250;color:#1e1208;font-weight:900;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:0.78rem;flex-shrink:0;margin-top:2px}
/* ── Tab intro banner ── */
.rq-intro-banner{display:flex;align-items:center;justify-content:space-between;gap:20px;background:#2e1f10;border-radius:14px;padding:22px 26px;margin-bottom:28px}
.rq-intro-title{font-size:1.05rem;font-weight:900;color:#f5f0e8;margin-bottom:5px}
.rq-intro-sub{font-size:0.84rem;color:#c4a882;line-height:1.6;max-width:820px}
.rq-intro-pills{display:flex;gap:10px;flex-shrink:0}
.rq-pill{display:flex;align-items:center;gap:8px;background:rgba(255,255,255,.08);border-radius:20px;padding:7px 14px;font-size:0.81rem;font-weight:700;color:#c4a882;white-space:nowrap}
.rq-pill.accent{background:#e8a250;color:#1e1208}
.rq-pill-n{font-size:1.05rem;font-weight:900}
/* ── Section label ── */
.rq-section-label{font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:#86674a;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #e8ddd0}
.rq-section-note{font-weight:500;text-transform:none;letter-spacing:0;color:#9a7a5c;font-size:0.78rem}
/* ── New RQ cards ── */
.rq2-card{display:flex;gap:0;background:#fff;border-radius:12px;border:1px solid #e8ddd0;margin-bottom:14px;overflow:hidden;transition:box-shadow .15s}
.rq2-card:hover{box-shadow:0 4px 16px rgba(46,31,16,.08)}
.rq2-left{width:72px;flex-shrink:0;background:#faf7f2;display:flex;flex-direction:column;align-items:center;padding:20px 10px;gap:10px;border-right:1px solid #e8ddd0}
.rq2-num{font-size:1.6rem;font-weight:900;color:#c2410c;line-height:1}
.rq2-diff{padding:3px 7px;border-radius:10px;font-size:0.67rem;font-weight:800;text-transform:uppercase;letter-spacing:.3px;text-align:center}
.rq2-diff.hard{background:#fee2e2;color:#b91c1c}
.rq2-diff.medium{background:#fef9c3;color:#92400e}
.rq2-body{flex:1;padding:18px 20px}
.rq2-title{font-size:0.95rem;font-weight:800;color:#1e1208;margin-bottom:8px}
.rq2-context{font-size:0.86rem;color:#5c3d22;line-height:1.65;margin:0 0 12px}
.rq2-context strong{color:#2e1f10}
.rq2-question{font-size:0.87rem;color:#2e1f10;line-height:1.75;background:#f5f0e8;border-left:3px solid #e8a250;padding:11px 15px;border-radius:0 8px 8px 0;margin-bottom:10px}
.rq2-question em{color:#7a5c3c;font-style:italic}
.rq2-where{font-size:0.78rem;color:#86674a;line-height:1.5}
.rq2-where-icon{color:#e8a250;font-weight:900;margin-right:4px}
/* ── Final Project 2.0 ── */
.fp2-hero{background:linear-gradient(135deg,#1e1208 0%,#2e1f10 60%,#3d2a18 100%);border-radius:14px;padding:2px;margin-bottom:4px}
.fp2-hero-inner{background:linear-gradient(135deg,#2e1f10 0%,#3d2a18 100%);border-radius:13px;padding:26px 28px}
.fp2-hero-top{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:22px}
.fp2-hero-title{font-size:1.26rem;font-weight:900;color:#e8a250;margin-bottom:8px}
.fp2-hero-sub{font-size:0.87rem;color:#c4a882;line-height:1.7;max-width:480px}
.fp2-hero-sub code{background:rgba(255,255,255,.12);padding:2px 6px;border-radius:4px;color:#f5f0e8;font-size:0.83rem}
.fp2-badge-col{display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex-shrink:0}
.fp2-file-chip{background:#e8a250;color:#1e1208;font-weight:800;font-family:monospace;font-size:0.83rem;padding:9px 16px;border-radius:8px;white-space:nowrap}
.fp2-path{font-size:0.75rem;color:#7a5c3c;text-align:right}
.fp2-stats-row{display:flex;align-items:center;gap:0;background:rgba(0,0,0,.2);border-radius:10px;overflow:hidden}
.fp2-stat{flex:1;text-align:center;padding:14px 10px}
.fp2-stat-val{font-size:1.21rem;font-weight:900;color:#f5f0e8}
.fp2-stat-lbl{font-size:0.69rem;color:#7a5c3c;margin-top:3px;text-transform:uppercase;letter-spacing:.3px}
.fp2-stat-div{width:1px;background:rgba(255,255,255,.06);align-self:stretch}
/* ── Challenge cards ── */
.challenge-card{background:#fff;border-radius:14px;padding:26px 28px;border:1px solid #e8ddd0;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
.ch-header{display:flex;align-items:flex-start;gap:16px;margin-bottom:18px;padding-bottom:16px;border-bottom:2px solid #f0e8d8}
.ch-num{font-size:2rem;font-weight:900;color:#e8ddd0;line-height:1;flex-shrink:0;font-variant-numeric:tabular-nums}
.ch-title{font-size:1.03rem;font-weight:800;color:#2e1f10;margin-bottom:3px}
.ch-file{font-size:0.81rem;color:#7a5c3c;font-family:monospace;background:#f5f0e8;padding:2px 8px;border-radius:4px;display:inline-block}
.ch-badge{margin-left:auto;flex-shrink:0;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px}
.cpu-badge{background:#dbeafe;color:#1d4ed8}
.hard-badge{background:#fef9c3;color:#92400e;border:1px solid #fde68a}
.ch-section-label{font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#7a5c3c;margin:16px 0 6px}
.ch-text{font-size:0.86rem;color:#2e1f10;line-height:1.7;margin:0 0 4px}
.ch-task-list{margin:0 0 18px;padding-left:20px;font-size:0.86rem;color:#2e1f10;line-height:1.9}
.ch-task-list li{margin-bottom:2px}
.ch-task-list code{font-size:0.86rem;background:#f0e8d8;padding:1px 5px;border-radius:3px;color:#5c3d22}
/* Feature ideas grid */
.ch-ideas-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.ch-idea{background:#faf7f2;border:1px solid #e8ddd0;border-radius:8px;padding:10px 12px}
.ch-idea.ch-idea-stretch{border-color:#f6d860;background:#fffbeb}
.ch-idea-name{font-size:0.84rem;font-weight:800;color:#2e1f10;margin-bottom:3px;font-family:monospace}
.ch-idea-desc{font-size:0.78rem;color:#5c3d22;line-height:1.5;margin-bottom:4px}
.ch-idea-src{font-size:0.71rem;color:#86674a;font-style:italic}
/* Stacking diagram */
.ch-stacking-diagram{display:flex;align-items:center;gap:12px;background:#faf7f2;border-radius:10px;padding:16px 20px;margin-bottom:16px;flex-wrap:wrap}
.ch-stack-col{display:flex;flex-direction:column;gap:8px}
.ch-stack-model,.ch-stack-meta,.ch-stack-out{padding:10px 14px;border-radius:8px;font-size:0.81rem;font-weight:700;text-align:center;line-height:1.3}
.ch-stack-model span,.ch-stack-meta span{font-weight:400;font-size:0.75rem;color:inherit;opacity:.8;display:block}
.ch-stack-model.rf{background:#dbeafe;color:#1e40af}
.ch-stack-model.tfidf{background:#dcfce7;color:#15803d}
.ch-stack-model.xgb{background:#fce7f3;color:#9d174d}
.ch-stack-meta{background:#3d2a18;color:#f5f0e8;min-width:110px}
.ch-stack-meta span{color:#c4a882}
.ch-stack-out{background:#e8a250;color:#1e1208}
.ch-stack-arrow{font-size:0.78rem;color:#86674a;font-weight:700;text-align:center;white-space:nowrap}
/* Warning box */
.ch-warn-box{background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:12px 16px;font-size:0.84rem;color:#92400e;line-height:1.6;margin-bottom:18px}
.ch-warn-box strong{display:block;margin-bottom:3px}
/* Cost matrix table */
.ch-cost-table{overflow-x:auto;margin:0 0 18px}
.ch-cost-table table{border-collapse:collapse;width:100%;font-size:0.85rem}
.ch-cost-table th,.ch-cost-table td{padding:10px 14px;border:1px solid #e8ddd0;text-align:center}
.ch-cost-table th{background:#f0e8d8;font-weight:800;color:#3d2a18}
.ch-cost-table td:first-child{background:#faf7f2;font-weight:700;color:#3d2a18;text-align:left}
.cost-tp,.cost-tn{background:#f0fdf4;color:#15803d}
.cost-fp{background:#fef2f2;color:#b91c1c}
.cost-tp span,.cost-tn span,.cost-fp span,.cost-fn span{display:block;font-size:0.75rem;font-weight:400;margin-top:3px}
.cost-fn{background:#fef2f2;color:#b91c1c;font-weight:700}
/* Reflect box */
.ch-reflect-box{background:#f5f0e8;border-radius:8px;padding:14px 16px;margin-bottom:18px}
.ch-reflect-title{font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.4px;color:#7a5c3c;margin-bottom:8px}
.ch-reflect-list{margin:0;padding-left:18px;font-size:0.87rem;color:#2e1f10;line-height:1.8}
/* Baseline → target row */
.ch-metrics-row{display:flex;align-items:center;gap:12px;background:#faf7f2;border-radius:10px;padding:14px 18px;flex-wrap:wrap}
.ch-metric-box{text-align:center;padding:8px 16px}
.ch-metric-box.ch-target{background:#fff;border-radius:8px;border:2px solid #e8a250}
.ch-metric-val{font-size:1.16rem;font-weight:900;color:#2e1f10}
.ch-target .ch-metric-val{color:#b45309}
.ch-metric-lbl{font-size:0.71rem;color:#7a5c3c;margin-top:2px}
.ch-arrow{font-size:1.26rem;color:#c4a882;font-weight:700}

/* ── Responsive ── */
@media(max-width:700px){
  #page{padding:20px 16px}
  .card-grid.cols-3,.card-grid.cols-4{grid-template-columns:1fr 1fr}
  .pipeline-steps{flex-direction:column}
  .pipeline-step+.pipeline-step::before{content:"↓";left:50%;top:-14px;transform:translateX(-50%)}
  .eda-grid{grid-template-columns:1fr}
  .fe-files-row{flex-direction:column;gap:4px}
}
@media(max-width:480px){
  .ds-legend{grid-template-columns:1fr}
  .card-grid.cols-3,.card-grid.cols-4,.card-grid.cols-2{grid-template-columns:1fr}
}
</style>
</head>
<body>

<!-- ── Header ── -->
<div id="app-header">
  <div>
    <div class="logo">AI-Sandbox &mdash; Final Project Explorer</div>
    <div class="subtitle">Healthcare ML Benchmark &bull; Synthea Patient Dataset &bull; 30-Iteration Study</div>
  </div>
  <div class="badge">READ-ONLY EXPLORER</div>
</div>

<!-- ── Tab navigation ── -->
<div id="tab-nav">
  <button class="tab-btn active" data-tab="problem">Problem</button>
  <button class="tab-btn" data-tab="dataset">Dataset</button>
  <button class="tab-btn" data-tab="features">Feature Engineering</button>
  <button class="tab-btn" data-tab="models">Models <span class="tab-badge" id="cart-count" style="display:none">0</span></button>
  <button class="tab-btn" data-tab="compare">Compare</button>
  <button class="tab-btn" data-tab="reproducibility">Reproducibility</button>
  <button class="tab-btn tab-deliverables" data-tab="challenges">Deliverables <span class="tab-sub">(Tasks &amp; Submission)</span></button>
</div>

<!-- ══════════════════════════════════════════════════════ -->
<!-- PROBLEM TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-problem" class="tab-panel active">
<div id="page">

  <div class="problem-hero">
    <h1>Benchmarking Machine Learning Models on Synthetic Clinical Records</h1>
    <p>This project benchmarks seven machine learning models on a binary patient classification task using 12,352 synthetic patients generated by Synthea. The classification task is to predict whether a patient has chronic pain based on their clinical history. We compare models ranging from logistic regression to a fine-tuned ClinicalBERT transformer. Each model uses patient information, including demographics, vitals, medications, and encounter notes. We evaluate performance and computational cost across 30 iterations with different random seeds.</p>
  </div>

  <!-- Context cards -->
  <div class="card-grid cols-3" style="margin-bottom:24px">
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🏥</div>
      <div class="section-title" style="font-size:0.97rem;margin-bottom:6px">Clinical Context</div>
      <div style="font-size:0.87rem;color:#6b5440;line-height:1.6">Chronic pain affects roughly 20% of US adults and is a leading cause of disability and healthcare spending. Electronic health records hold the diagnoses, medications, and vitals needed to study it at scale, but privacy rules limit access to real patient data. Synthetic records offer a safe way to build and test methods.</div>
    </div>
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🎯</div>
      <div class="section-title" style="font-size:0.97rem;margin-bottom:6px">Research Question</div>
      <div style="font-size:0.87rem;color:#6b5440;line-height:1.6">Healthcare teams must choose between simple, interpretable models and large deep learning models that need GPUs and specialized infrastructure. Which machine learning architecture offers the best trade-off between classification performance and computational cost, and when is the added complexity actually worth it?</div>
    </div>
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🔬</div>
      <div class="section-title" style="font-size:0.97rem;margin-bottom:6px">Scientific Rigor</div>
      <div style="font-size:0.87rem;color:#6b5440;line-height:1.6">Many machine learning results in healthcare come from a single train/test split and are hard to reproduce. A model can look better because of a lucky split, a leaky feature, or noisy labels. Repeating every experiment across 30 random seeds on shared splits shows which differences are real and which are chance.</div>
    </div>
  </div>

  <!-- Pipeline overview -->
  <div class="section-title">End-to-End Pipeline</div>
  <div class="section-sub">Every model sees the same patients and the same split in each iteration &mdash; only the input representation and architecture differ.</div>
  <div class="pipeline-steps">
    <div class="pipeline-step" style="flex:1.8">
      <div class="step-icon">📋</div>
      <div class="step-title">Raw Data</div>
      <div class="step-desc">16 Synthea CSVs: patients, encounters, conditions, medications, observations, procedures, immunizations, allergies, careplans, devices, imaging studies, supplies, providers, organizations, payers, payer transitions</div>
    </div>
    <div class="pipeline-step">
      <div class="step-icon">🏷️</div>
      <div class="step-title">Keyword Labels</div>
      <div class="step-desc">31 pain-related keywords matched against condition descriptions</div>
    </div>
    <div class="pipeline-step">
      <div class="step-icon">⚙️</div>
      <div class="step-title">Feature Engineering</div>
      <div class="step-desc">Demographics + 7 vitals, plus medication text: 12 drug TF-IDF terms, 5 LDA topics, PCA/UMAP, sentiment, clusters</div>
    </div>
    <div class="pipeline-step">
      <div class="step-icon">✂️</div>
      <div class="step-title">Shared Split</div>
      <div class="step-desc">Stratified 70/15/15 train/val/test (8,646 / 1,853 / 1,853), new seed each iteration</div>
    </div>
    <div class="pipeline-step">
      <div class="step-icon">🤖</div>
      <div class="step-title">7 Models Trained</div>
      <div class="step-desc">CPU: RF, XGBoost, TF-IDF + LogReg. GPU: LSTM, GRU, Transformer, ClinicalBERT</div>
    </div>
    <div class="pipeline-step">
      <div class="step-icon">📊</div>
      <div class="step-title">Benchmark</div>
      <div class="step-desc">Test-set AUC, accuracy, precision, recall, F1 + run time, GPU hours, CPU use &times; 30 seeds</div>
    </div>
  </div>

  <!-- Model families -->
  <div class="section-title" style="margin-top:24px">Three Model Families</div>
  <div class="section-sub">Models differ in what data they consume and what hardware they need. F1 values are 30-iteration means on the held-out test set.</div>
  <div class="card-grid cols-3">
    <div class="card" style="border-left:4px solid #16a34a">
      <div style="font-size:1.26rem;margin-bottom:8px">🖥️ <span class="pill cpu">CPU Baseline</span></div>
      <div style="font-weight:800;font-size:0.97rem;margin-bottom:8px">Tabular &amp; Text</div>
      <div style="font-size:0.85rem;color:#6b5440;line-height:1.6;margin-bottom:10px">Random Forest and XGBoost use 13 structured features (demographics, costs, vitals). TF-IDF Logistic Regression reads each patient&rsquo;s raw medication text. All three train in a few seconds per iteration on CPU.</div>
      <div style="font-size:0.78rem;color:#16a34a;font-weight:700">F1 range: 0.649 &ndash; 0.699</div>
    </div>
    <div class="card" style="border-left:4px solid #7c3aed">
      <div style="font-size:1.26rem;margin-bottom:8px">⚡ <span class="pill gpu">GPU Sequence</span></div>
      <div style="font-weight:800;font-size:0.97rem;margin-bottom:8px">Sequence Models</div>
      <div style="font-size:0.85rem;color:#6b5440;line-height:1.6;margin-bottom:10px">LSTM, GRU, and a custom Transformer take structured and medication-text features as 10-step sequences. The feature table has one row per patient, so each sequence is left-padded and holds a single real time step. 14&ndash;22 s per iteration on an A100.</div>
      <div style="font-size:0.78rem;color:#7c3aed;font-weight:700">F1 range: 0.679 &ndash; 0.692</div>
    </div>
    <div class="card" style="border-left:4px solid #1d4ed8">
      <div style="font-size:1.26rem;margin-bottom:8px">🧠 <span class="pill bert">Pretrained BERT</span></div>
      <div style="font-weight:800;font-size:0.97rem;margin-bottom:8px">Clinical Language Model</div>
      <div style="font-size:0.85rem;color:#6b5440;line-height:1.6;margin-bottom:10px">Bio_ClinicalBERT, pre-trained on real clinical notes (MIMIC-III), turns each patient&rsquo;s medication text into an embedding once. A small Transformer head learns from that embedding plus structured features. Scored on the 1,445 test patients who have medication records. ~36 s per iteration.</div>
      <div style="font-size:0.78rem;color:#1d4ed8;font-weight:700">F1: 0.707 (highest mean F1)</div>
    </div>
  </div>

</div>
</div><!-- /tab-problem -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- DATASET TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-dataset" class="tab-panel">
<div id="page">

  <!-- The problem -->
  <div class="section-title">The Problem We Are Solving</div>
  <div class="card" style="margin-bottom:24px">
    <div class="ds-text">Every model in this project answers the same yes-or-no question: does this synthetic patient have chronic pain or not? Supervised learning needs that answer, called the label, for every patient. Health records do not include a ready-made &ldquo;has chronic pain&rdquo; field, so we build the label ourselves from the diagnoses in conditions.csv. How we build it shapes what every model learns and what its scores really mean.</div>
  </div>

  <!-- How labels are made -->
  <div class="section-title">How We Create Labels</div>
  <div class="card-grid cols-2" style="margin-bottom:16px">
    <div class="card">
      <div class="ds-text">We create patient-level labels by searching diagnosis descriptions for 31 pain-related keywords, such as &ldquo;pain,&rdquo; &ldquo;arthritis,&rdquo; &ldquo;migraine,&rdquo; &ldquo;fracture,&rdquo; and &ldquo;injury.&rdquo;</div>
      <div class="ds-text"><span class="label-badge-1">Positive (1)</span> At least one diagnosis contains a matching keyword.</div>
      <div class="ds-text"><span class="label-badge-0">Negative (0)</span> No diagnosis contains a matching keyword.</div>
      <div class="ds-text">This produces 6,235 positive patients (50.5%) and 6,117 negative patients (49.5%).</div>
      <div class="ds-text">The labels are simple and reproducible but imperfect. They include chronic conditions, temporary injuries, and some unrelated diagnoses. Because the model features come from the same patient history used to create the labels, they may contain information that directly reveals the label.</div>
    </div>
    <div class="card">
      <div class="ds-card-title">The 31 Keywords</div>
      <div id="label-keywords" style="display:flex;flex-wrap:wrap;gap:5px"></div>
    </div>
  </div>

  <!-- Class balance bar -->
  <div class="card" style="margin-bottom:28px;padding:14px 20px">
    <div class="ds-card-title">Label Distribution</div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <div style="flex:1;min-width:200px;height:22px;border-radius:5px;overflow:hidden;display:flex">
        <div id="pos-bar" style="background:#16a34a;height:100%;transition:width .6s"></div>
        <div id="neg-bar" style="background:#dc2626;height:100%;transition:width .6s"></div>
      </div>
      <div style="font-size:0.81rem;color:#6b5440;white-space:nowrap">
        <span style="color:#16a34a;font-weight:800" id="bar-pos-pct"></span> &nbsp;|&nbsp;
        <span style="color:#dc2626;font-weight:800" id="bar-neg-pct"></span>
      </div>
    </div>
  </div>

  <!-- Label flaws -->
  <div class="section-title">More Insights on the Classification Label</div>
  <div class="section-sub">Keyword labels are quick and transparent, which is why they are common in health-records research. A closer look at ours shows a few things to keep in mind before you trust any model score.</div>

  <div class="card-grid cols-3" style="margin-bottom:28px">
    <div class="card">
      <div class="ds-card-title">🫁 Some matches are not about pain</div>
      <div class="ds-text">Keyword matching is noisy. Some diagnoses contain a keyword but have nothing to do with pain, so those patients get the wrong label. For example, &ldquo;chronic&rdquo; also matches chronic sinusitis, chronic kidney disease, and chronic heart failure.</div>
    </div>
    <div class="card">
      <div class="ds-card-title">🩹 Temporary problems count as much as chronic pain</div>
      <div class="ds-text">A sprained ankle that has healed counts the same as fibromyalgia. &ldquo;Joint pain&rdquo; and &ldquo;muscle pain&rdquo; each appear in 1,352 patients, all of whom had COVID-19, and 93% of those records are marked resolved. 1,225 positive patients (19.6%) have only short-term problems.</div>
    </div>
    <div class="card">
      <div class="ds-card-title">📈 The label grows with age</div>
      <div class="ds-text">Every diagnosis in a patient&rsquo;s lifetime counts, so older patients have more chances to match a keyword. A model can score well partly by learning age.</div>
      <div class="ds-age-row"><span>Ages 0&ndash;17</span><div class="ds-age-track"><div style="width:19.5%"></div></div><span>19.5%</span></div>
      <div class="ds-age-row"><span>Ages 18&ndash;39</span><div class="ds-age-track"><div style="width:32.1%"></div></div><span>32.1%</span></div>
      <div class="ds-age-row"><span>Ages 40&ndash;64</span><div class="ds-age-track"><div style="width:50.5%"></div></div><span>50.5%</span></div>
      <div class="ds-age-row"><span>Ages 65+</span><div class="ds-age-track"><div style="width:73.2%"></div></div><span>73.2%</span></div>
    </div>
  </div>

  <!-- Raw data explorer -->
  <div class="section-title">Raw Data Explorer</div>
  <div class="section-sub">The Synthea COVID-19 10K export has 16 CSV files. Pick a file to see its first 25 rows, a quick profile of every column, and what each column means. All records are synthetic, so no real patient data is shown.</div>

  <!-- Raw file sub-tabs -->
  <div id="raw-tab-nav" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px"></div>

  <!-- Raw table panels (one per file) -->
  <div id="raw-panels"></div>

</div>
</div><!-- /tab-dataset -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- FEATURES TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-features" class="tab-panel">
<div id="page">

  <!-- A small baseline on purpose -->
  <div class="section-title">A Small Baseline, on Purpose</div>
  <div class="card" style="margin-bottom:24px">
    <div class="ds-text">We intentionally use a small, simple set of features. The benchmark models are a baseline for you to beat, not the best this data can do. Their features come from just 3 of the 16 CSV files, and conditions.csv is used only to build the label. The other 12 files are untouched, which leaves plenty of room for your own feature engineering.</div>
    <div class="fe-files-row"><span class="fe-files-lbl">Used for features</span><div class="fe-chips"><span class="fe-chip fe-chip-used">patients.csv</span><span class="fe-chip fe-chip-used">observations.csv</span><span class="fe-chip fe-chip-used">medications.csv</span></div></div>
    <div class="fe-files-row"><span class="fe-files-lbl">Used only for the label</span><div class="fe-chips"><span class="fe-chip fe-chip-label">conditions.csv</span></div></div>
    <div class="fe-files-row"><span class="fe-files-lbl">Not used yet</span><div class="fe-chips"><span class="feature-tag">encounters.csv</span><span class="feature-tag">procedures.csv</span><span class="feature-tag">immunizations.csv</span><span class="feature-tag">allergies.csv</span><span class="feature-tag">careplans.csv</span><span class="feature-tag">devices.csv</span><span class="feature-tag">imaging_studies.csv</span><span class="feature-tag">supplies.csv</span><span class="feature-tag">providers.csv</span><span class="feature-tag">organizations.csv</span><span class="feature-tag">payers.csv</span><span class="feature-tag">payer_transitions.csv</span></div></div>
  </div>

  <!-- Pipeline -->
  <div class="section-title">Feature Engineering Pipeline</div>
  <div class="section-sub">Raw records become one row of numbers per patient in two steps: structured features from patients.csv and observations.csv, and text features from medications.csv.</div>

  <div class="pipeline-flow">

    <div class="flow-row">
      <div class="flow-node source">
        <div class="fn-label">Stage 0 — Raw Input</div>
        <div class="fn-title">3 of the 16 Synthea CSV Files</div>
        <div class="fn-desc">patients.csv &bull; observations.csv &bull; medications.csv. Apart from patients.csv, each file has many rows per patient, so every feature below is a per-patient summary.</div>
      </div>
    </div>

    <div class="flow-row"><div class="flow-arrow">↓</div></div>

    <div class="flow-row">
      <div class="flow-node process">
        <div class="fn-label">Stage 1 — Structured Features</div>
        <div class="fn-title">13 Features from Demographics, Costs, and Vital Signs</div>
        <div class="fn-desc">From patients.csv: race and ethnicity (converted to numbers), age, gender, and lifetime healthcare expenses and coverage. From observations.csv: the average of every recorded height, weight, BMI, systolic and diastolic blood pressure, heart rate, and respiratory rate. Missing vitals are filled with the median.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-cpu">Random Forest</span>
          <span class="model-use-tag mtu-cpu">XGBoost</span>
          <span class="model-use-tag mtu-gpu">LSTM</span>
          <span class="model-use-tag mtu-gpu">GRU</span>
          <span class="model-use-tag mtu-gpu">Transformer</span>
          <span class="model-use-tag mtu-bert">ClinicalBERT</span>
        </div>
      </div>
    </div>

    <div class="flow-row"><div class="flow-arrow">↓</div></div>

    <div class="flow-row">
      <div class="flow-node process">
        <div class="fn-label">Stage 2 — Medication Text Features</div>
        <div class="fn-title">19 Features from Medication Names</div>
        <div class="fn-desc">Join each patient&rsquo;s medication names into one &ldquo;medication note.&rdquo; From it, compute TF-IDF weights for 12 pain-related drug names, 5 LDA topics (themes in medication use), a sentiment score, and a K-means cluster number. The 2,786 patients with no medications get 0 for all of these.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-gpu">LSTM</span>
          <span class="model-use-tag mtu-gpu">GRU</span>
          <span class="model-use-tag mtu-gpu">Transformer</span>
          <span class="model-use-tag mtu-bert">ClinicalBERT</span>
        </div>
      </div>
    </div>

  </div><!-- /pipeline-flow -->

  <!-- Model inputs -->
  <div class="section-title" style="margin-top:28px">What Each Model Receives</div>
  <div class="section-sub">Every model sees the same patients, but each one gets a different slice of the features.</div>
  <div class="card" style="padding:0;overflow:hidden;margin-bottom:24px">
    <table class="comp-table fe-models">
      <thead><tr><th style="width:24%">Model</th><th style="width:30%">Input</th><th>How it is shaped</th></tr></thead>
      <tbody>
        <tr>
          <td><span class="model-use-tag mtu-cpu">Random Forest</span> <span class="model-use-tag mtu-cpu">XGBoost</span></td>
          <td>13 structured features</td>
          <td>One row per patient.</td>
        </tr>
        <tr>
          <td><span class="model-use-tag mtu-cpu">TF-IDF + LogReg</span></td>
          <td>Raw medication text</td>
          <td>Its own TF-IDF over up to 20,000 words and word pairs. Patients with no medications are left out.</td>
        </tr>
        <tr>
          <td><span class="model-use-tag mtu-gpu">LSTM</span> <span class="model-use-tag mtu-gpu">GRU</span> <span class="model-use-tag mtu-gpu">Transformer</span></td>
          <td>32 features: the 13 structured features plus the 19 medication text features</td>
          <td>Shaped as a 10-step sequence, the format these models expect. In the baseline, each patient is summarized in a single step, so the rest of the sequence is left empty. Adding visit-by-visit history is an open opportunity.</td>
        </tr>
        <tr>
          <td><span class="model-use-tag mtu-bert">ClinicalBERT</span></td>
          <td>A 768-number embedding of the medication text, plus the same 32 features</td>
          <td>Each patient&rsquo;s medication names are joined, cut to the first 256 tokens, and run once through Bio_ClinicalBERT. The embedding is computed once and reused in all 30 runs; only a small Transformer head on top is trained.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Leakage risk -->
  <div class="section-title">⚠️ Label Leakage Risk</div>
  <div class="card" style="margin-bottom:24px">
    <div class="ds-text">Condition text is left out of every model because it defines the label. Medication text is kept, but Synthea prescribes medications for the same diagnoses, so it can partly reveal the label. The Dataset tab explains how the label is built.</div>
  </div>

  <!-- All features table -->
  <div class="section-title">All 32 Features Used by the Models</div>
  <div class="section-sub">13 structured features and 19 medication text features. Each row shows the column name, what it means, where it comes from, how it is calculated, and which models use it. Sequence models means LSTM, GRU, and Transformer.</div>

  <!-- Source legend -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
    <span class="src-badge src-patients">patients.csv</span>
    <span class="src-badge src-observations">observations.csv</span>
    <span class="src-badge src-medications">medications.csv</span>
  </div>

  <div class="feat-table-wrap">
    <table class="feat-table fe-table" id="feat-table">
      <thead>
        <tr>
          <th style="width:32px">#</th>
          <th style="min-width:170px">Feature Name</th>
          <th style="min-width:200px">Description</th>
          <th style="min-width:120px">Source File</th>
          <th style="min-width:130px">Raw Column(s) Used</th>
          <th style="min-width:240px">How Calculated</th>
          <th style="min-width:150px">Used By</th>
        </tr>
      </thead>
      <tbody id="feat-tbody"></tbody>
    </table>
  </div>

  <!-- Feature importance -->
  <div class="section-title" style="margin-top:28px">Which Baseline Features Matter?</div>
  <div class="section-sub">How much the Random Forest&rsquo;s test AUC drops when one feature&rsquo;s values are shuffled, averaged over the 30 seeded splits. A bigger drop means the model relies on that feature more.</div>
  <div id="feat-importance"></div>
  <div class="ds-note" style="margin-bottom:24px">Age and insurance coverage carry most of the signal. Age matters partly because older patients have more diagnoses and are more often labeled positive, as the Dataset tab shows.</div>

  <!-- Room to improve -->
  <div class="section-title">Room to Improve</div>
  <div class="section-sub">Because the baseline is deliberately simple, there are clear places to start.</div>
  <div class="card-grid cols-2">
    <div class="card">
      <div class="ds-card-title">🧩 Add more features</div>
      <div class="ds-text">The baseline is deliberately simple, so there is plenty of room to improve it. Explore the data, follow your own ideas, and build features you can explain. Every feature you add is a chance to beat the benchmark.</div>
    </div>
    <div class="card">
      <div class="ds-card-title">⚠️ Watch for leakage</div>
      <div class="ds-text">Avoid features built from conditions.csv or from REASONDESCRIPTION columns, which repeat the diagnoses behind the label. If a new feature makes the score jump, check whether it is quietly encoding the label.</div>
    </div>
  </div>

</div>
</div><!-- /tab-features -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- MODELS TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-models" class="tab-panel">
<div id="page">

  <div class="section-title">Model Gallery</div>
  <div class="mission-callout">
    <div class="mission-icon">🎯</div>
    <div class="mission-text">Complete your exploration, run your best model 5 times, and upload its performance (my_model.csv) on the Deliverables tab to update your model card here. <span class="mission-hl">Beat the best CPU model</span> to unlock the GPU tier, then <span class="mission-hl">beat the GPU models</span> to complete the task. F1-Score is the primary metric.</div>
  </div>

  <div class="mm-upload" style="margin-bottom:18px">
    <button class="mm-btn mm-btn-primary" onclick="goToDeliverables()">Upload on Deliverables &rarr;</button>
    <div class="mm-upload-status"></div>
  </div>

  <!-- Selected models bar -->
  <div class="compare-bar" id="compare-bar" style="display:none">
    <span class="cb-title">Comparing:</span>
    <div class="selected-pills" id="selected-pills"></div>
    <button class="compare-go-btn" onclick="goToCompare()">Compare Selected &rarr;</button>
  </div>

  <div id="models-grid"></div>

</div>
</div><!-- /tab-models -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- COMPARE TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-compare" class="tab-panel">
<div id="page">

  <div id="compare-empty">
    <div class="ce-icon">⚖️</div>
    <h3>No models selected yet</h3>
    <p>Go to the <strong>Models</strong> tab, select 2 or more model cards,<br/>then click <strong>Compare Selected</strong> to see results here.</p>
  </div>

  <div id="compare-content">

    <!-- Back to Models -->
    <div class="compare-tip">
      <div class="compare-tip-icon">🔁</div>
      <div class="compare-tip-body">
        <div class="compare-tip-title">Want to compare other models?</div>
        <div class="compare-tip-text">Go back to the Models tab, select or deselect model cards, and click Compare Selected. This tab updates with your new selection.</div>
      </div>
      <button class="compare-go-btn" onclick="goToModels()">&larr; Back to Models</button>
    </div>

    <!-- Performance section -->
    <div class="compare-section">
      <div class="compare-section-title">📈 Performance (mean across runs)</div>
      <div class="metric-tabs" id="metric-tabs">
        <button class="metric-tab" onclick="setMetric(this,'auc')">AUC-ROC</button>
        <button class="metric-tab" onclick="setMetric(this,'accuracy')">Accuracy</button>
        <button class="metric-tab active" onclick="setMetric(this,'f1')">F1-Score ★</button>
        <button class="metric-tab" onclick="setMetric(this,'precision')">Precision</button>
        <button class="metric-tab" onclick="setMetric(this,'recall')">Recall</button>
      </div>
      <div id="chart-svg-wrap"></div>
      <div style="margin-top:14px;overflow-x:auto">
        <table class="comp-table" id="perf-table">
          <thead>
            <tr id="perf-thead-row"><th>Metric</th></tr>
          </thead>
          <tbody id="perf-tbody"></tbody>
        </table>
      </div>
      <div style="font-size:0.78rem;color:#86674a;margin-top:10px">
        Benchmark values are means across 30 random-seed runs on shared 70/15/15 train/validation/test splits. My Model values are the mean of the 5 iterations in your results file.
      </div>
    </div>

    <!-- Computational section -->
    <div class="compare-section">
      <div class="compare-section-title">⚙️ Computational Cost</div>
      <div style="overflow-x:auto">
        <table class="comp-table" id="compute-table">
          <thead>
            <tr id="comp-thead-row"><th>Attribute</th></tr>
          </thead>
          <tbody id="comp-tbody"></tbody>
        </table>
      </div>
      <div style="font-size:0.78rem;color:#86674a;margin-top:10px">
        Benchmark times, GPU hours, and CPU utilization are 30-run means measured on an NVIDIA A100 (40 GB) via the NAIRR Classroom Pilot allocation. ClinicalBERT&rsquo;s time covers training its small head; its BERT embeddings are computed once beforehand. My Model values are the mean of your 5 iterations; &mdash; means not reported.<br/>Cost in dollars is an estimate: run time &times; Google Cloud on-demand price (us-central1, checked Oct 12, 2026). GPU models are priced at $3.6734/hr for an a2-highgpu-1g machine (1&times; A100 40 GB), and CPU models at $0.3885/hr for an n2-standard-8 machine. Jetstream2, where the benchmark ran, has no dollar pricing; it charges ACCESS credits. ClinicalBERT&rsquo;s one-time embedding step is not included, and Random Forest and XGBoost were timed together, so each shows their combined time.
      </div>
    </div>

    <!-- Pros / cons section -->
    <div class="compare-section">
      <div class="compare-section-title">✅ Strengths &amp; ⚠️ Limitations</div>
      <div class="pros-cons-grid" id="pros-cons-grid"></div>
    </div>


  </div><!-- /compare-content -->

</div>
</div><!-- /tab-compare -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- REPRODUCIBILITY TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-reproducibility" class="tab-panel">
<div id="page">

  <!-- Hero -->
  <div class="repro-hero">
    <div class="repro-hero-left">
      <div class="repro-hero-eyebrow">Core ML Practice</div>
      <div class="repro-hero-title">Reproducibility</div>
      <p class="repro-hero-body">
        ClinicalBERT reached F1 0.730 in its luckiest run, but across all 30 runs it
        averages 0.707 ± 0.014. Reporting only the lucky run would overstate it by 0.023.
        The 30-run average is the number you can trust.
        Reproducibility is the difference between a result and a finding.
      </p>
    </div>
    <div class="repro-hero-right">
      <div class="repro-concept-card">
        <div class="repro-concept-icon">🎲</div>
        <div class="repro-concept-label">Random Seed</div>
        <div class="repro-concept-desc">Controls how data is split and how models initialize. Different seeds = different results.</div>
      </div>
      <div class="repro-concept-card">
        <div class="repro-concept-icon">📊</div>
        <div class="repro-concept-label">Standard Deviation</div>
        <div class="repro-concept-desc">How much results vary across runs. Low std = stable model. High std = sensitive to split luck.</div>
      </div>
      <div class="repro-concept-card">
        <div class="repro-concept-icon">🔁</div>
        <div class="repro-concept-label">30 Iterations</div>
        <div class="repro-concept-desc">Each iteration uses a fresh random seed, shared across all 7 models so comparisons are fair.</div>
      </div>
    </div>
  </div>

  <!-- Variation range chart -->
  <div class="section-header" style="margin-top:0">F1-Score Range Across 30 Iterations</div>
  <p style="color:#5c3d22;font-size:0.87rem;line-height:1.6;margin-bottom:16px">
    Each bar shows the <strong>full spread</strong> (min → max) of F1 values across 30 runs.
    The filled circle marks the <strong>mean</strong>. The ranges overlap heavily: only XGBoost and
    ClinicalBERT never overlap, so a single lucky run can make most models look like the best.
  </p>
  <div id="repro-range-chart"></div>

  <!-- Divider -->
  <div style="height:28px"></div>

  <!-- Interactive slider section -->
  <div class="section-header">Explore Each Iteration</div>
  <p style="color:#5c3d22;font-size:0.87rem;line-height:1.6;margin-bottom:20px">
    Drag the slider to see how every model performs on a specific iteration.
    Watch the winner change. This is why a single run cannot be trusted.
  </p>

  <div class="repro-slider-wrap">
    <div class="repro-slider-top">
      <span class="repro-slider-label">Iteration</span>
      <div class="repro-iter-display" id="repro-iter-num">1</div>
      <span class="repro-slider-label" style="color:#86674a">of 30</span>
    </div>
    <input type="range" id="repro-slider" min="1" max="30" value="1" class="repro-slider">
    <div class="repro-slider-ticks">
      <span>1</span><span>5</span><span>10</span><span>15</span><span>20</span><span>25</span><span>30</span>
    </div>
  </div>

  <!-- Insight box -->
  <div class="repro-insight-box" id="repro-insight"></div>

  <!-- Per-iteration table -->
  <div class="repro-iter-table-wrap">
    <table class="repro-iter-table" id="repro-iter-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>F1 ★</th>
          <th>AUC</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody id="repro-iter-tbody"></tbody>
    </table>
  </div>

  <!-- Summary stats table -->
  <div style="height:28px"></div>
  <div class="section-header">30-Iteration Summary Statistics</div>
  <p style="color:#5c3d22;font-size:0.87rem;line-height:1.6;margin-bottom:16px">
    Mean and standard deviation across all 30 runs. A high std means the model's apparent
    performance depends heavily on which patients ended up in the test set.
  </p>
  <div class="repro-summary-wrap">
    <table class="repro-summary-table" id="repro-summary-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>F1 Mean ★</th>
          <th>F1 Std</th>
          <th>F1 Min</th>
          <th>F1 Max</th>
          <th>F1 Range</th>
          <th>AUC Mean</th>
          <th>AUC Std</th>
        </tr>
      </thead>
      <tbody id="repro-summary-tbody"></tbody>
    </table>
  </div>

  <!-- Takeaway -->
  <div class="repro-takeaway">
    <div class="repro-takeaway-title">The core lesson</div>
    <div class="repro-takeaway-grid">
      <div class="repro-takeaway-item">
        <div class="repro-takeaway-num">1</div>
        <div><strong>Never report a single run.</strong> Here, each model's best run beats its 30-run average by 0.02 to 0.035.</div>
      </div>
      <div class="repro-takeaway-item">
        <div class="repro-takeaway-num">2</div>
        <div><strong>Share your random seed.</strong> Anyone re-running your code should get the same number.</div>
      </div>
      <div class="repro-takeaway-item">
        <div class="repro-takeaway-num">3</div>
        <div><strong>Report mean ± std.</strong> The spread tells readers how confident they should be in your result.</div>
      </div>
      <div class="repro-takeaway-item">
        <div class="repro-takeaway-num">4</div>
        <div><strong>Use shared splits.</strong> All 7 models here use the same 30 train/validation/test splits — that's why their comparisons are fair.</div>
      </div>
    </div>
  </div>

</div>
</div><!-- /tab-reproducibility -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- RESEARCH CHALLENGES TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-challenges" class="tab-panel">
<div id="page">

  <!-- ── Page intro ── -->
  <div class="rq-intro-banner">
    <div class="rq-intro-left">
      <div class="rq-intro-title">Deliverables</div>
      <div class="rq-intro-sub">Everything you hand in for the final project: a written report with your task reflections, and your best model&rsquo;s results file. Submit your report to D2L, and upload your best model&rsquo;s my_model.csv here to update your model card.</div>
    </div>
    <div class="rq-intro-pills">
      <div class="rq-pill"><span class="rq-pill-n">4</span>Tasks</div>
      <div class="rq-pill"><span class="rq-pill-n">1</span>Report</div>
      <div class="rq-pill accent"><span class="rq-pill-n">1</span>Results file</div>
    </div>
  </div>

  <!-- ── Data splits ── -->
  <div class="rq-section-label">Data Splits <span class="rq-section-note">— 12,352 patients, split the same way for every model</span></div>
  <div class="card" style="padding:0;overflow:hidden;margin-bottom:10px">
    <table class="comp-table dl-splits">
      <thead>
        <tr><th>Split</th><th>Patients</th><th style="width:32%">Share of all patients</th><th>Class 0 (negative)</th><th>Class 1 (positive)</th></tr>
      </thead>
      <tbody>
        <tr><td>Train</td><td>8,646</td><td><div class="dl-share"><div class="dl-share-bar" style="width:70%"></div><span>70.0%</span></div></td><td>4,281 (49.5%)</td><td>4,365 (50.5%)</td></tr>
        <tr><td>Validation</td><td>1,853</td><td><div class="dl-share"><div class="dl-share-bar" style="width:15%"></div><span>15.0%</span></div></td><td>918 (49.5%)</td><td>935 (50.5%)</td></tr>
        <tr><td>Test</td><td>1,853</td><td><div class="dl-share"><div class="dl-share-bar" style="width:15%"></div><span>15.0%</span></div></td><td>918 (49.5%)</td><td>935 (50.5%)</td></tr>
        <tr class="dl-total"><td>Total</td><td>12,352</td><td>100%</td><td>6,117 (49.5%)</td><td>6,235 (50.5%)</td></tr>
      </tbody>
    </table>
  </div>
  <div class="ds-note" style="margin-bottom:28px">The split is stratified, so every part keeps the same class balance. Use the training set to fit your model, the validation set to tune your choices, and the test set only for your final scores.</div>

  <!-- ── Tasks ── -->
  <div class="rq-section-label">Tasks <span class="rq-section-note">— experiments to reflect on in your report</span></div>
  <div class="mission-callout">
    <div class="mission-icon">📌</div>
    <div class="mission-text">These tasks change your model on purpose: less training data, removed features, missing values. They are for <span class="mission-hl">reflection in your report only</span>. Do not upload their results; my_model.csv is for your best model.</div>
  </div>

  <div class="rq2-card">
    <div class="rq2-left"><div class="rq2-num">01</div></div>
    <div class="rq2-body">
      <div class="rq2-title">How Much Training Data Does Your Model Need?</div>
      <p class="rq2-context">Design your model, then train it three times: on 1,000, 4,000, and all 8,646 training patients. Keep the same validation and test patients for every run, so the only thing that changes is the amount of training data.</p>
      <ol class="dl-steps">
        <li>Before each run, write down what you expect to happen and why.</li>
        <li>Run your model.</li>
        <li>Compare the result with your expectation and explain any gap.</li>
      </ol>
      <div class="rq2-question">It is okay to be wrong. If your expectation missed, explain why you were wrong and what you learned.</div>
    </div>
  </div>

  <div class="rq2-card">
    <div class="rq2-left"><div class="rq2-num">02</div></div>
    <div class="rq2-body">
      <div class="rq2-title">Which Features Drive Your Model?</div>
      <p class="rq2-context">Find out which features your best-performing model relies on most, then test what happens without them.</p>
      <ol class="dl-steps">
        <li>Identify the top features contributing to your best-performing model, and explain why you think they matter most.</li>
        <li>Remove the top two features and retrain your model.</li>
        <li>Report the new results.</li>
      </ol>
      <div class="rq2-question">Explain your updated results and the reasoning behind them. Did performance change the way you expected, and what does that tell you about the remaining features?</div>
    </div>
  </div>

  <div class="rq2-card">
    <div class="rq2-left"><div class="rq2-num">03</div></div>
    <div class="rq2-body">
      <div class="rq2-title">How Robust Is Your Model to Missing Data?</div>
      <p class="rq2-context">Real health records are often incomplete. Simulate this by hiding values in your most important features.</p>
      <ol class="dl-steps">
        <li>Randomly set 50% of the values in each of your top 3 features to missing. Use a fixed random seed so you can reproduce it.</li>
        <li>Before running, predict how your model&rsquo;s performance will change and why.</li>
        <li>Fill the missing values using a statistic, such as the mean or median, computed on the training set.</li>
        <li>Run your code and report the results.</li>
      </ol>
      <div class="rq2-question">Compare the results with your expectation and explain any gap. It is okay to be wrong; reason about why and what you learned.</div>
    </div>
  </div>

  <div class="rq2-card">
    <div class="rq2-left"><div class="rq2-num">04</div></div>
    <div class="rq2-body">
      <div class="rq2-title">Design a Better Label <span class="dl-report-chip">For report only, not this explorer</span></div>
      <p class="rq2-context">Our current binary_label identifies patients whose diagnosis text contains any of 31 predefined keywords. However, this approach may introduce noisy labels (see the Dataset tab).</p>
      <ol class="dl-steps">
        <li>Propose an improved definition of binary_label.</li>
        <li>Justify each of your labeling rules.</li>
        <li>Report how many patients would receive a different label and how the class distribution changes.</li>
        <li>Using your new labels, retrain and evaluate one machine learning model for a single run. Report its performance and briefly discuss your findings.</li>
      </ol>
      <div class="rq2-question">For each rule, explain what noise it removes and what real cases it might miss.</div>
      <div class="ch-warn-box" style="margin:10px 0 0"><strong>⚠️ Important</strong>Be mindful of data leakage when selecting input features. Do not use information that directly defines or reveals your target label as a model input.</div>
    </div>
  </div>

  <!-- ── Results file ── -->
  <div class="rq-section-label" id="results-file" style="margin-top:28px;scroll-margin-top:16px">Results File <span class="rq-section-note">— upload my_model.csv here</span></div>
  <div class="card" style="margin-bottom:28px">
    <div class="mission-callout">
      <div class="mission-icon">🏆</div>
      <div class="mission-text">Upload <span class="mission-hl">your best model only</span>: run it 5 times, each with a different random seed, and record its performance in my_model.csv, one row per run. Results from the task experiments belong in your report, not in this file. We ask for 5 runs instead of the benchmark’s 30 because of limited computational resources.</div>
    </div>
    <div class="ds-text">Start from the template and keep the column names exactly as they are. Every column needs a real value in every row: blank, null, and 0 are not accepted, except gpu_hrs, which is 0 if you trained on a CPU. If anything is missing or invalid, the upload is rejected with a list of what to fix.</div>
    <table class="comp-table dl-columns">
      <thead><tr><th>Column</th><th>What to enter</th></tr></thead>
      <tbody>
        <tr><td><span class="feat-name">iteration</span></td><td>Run number, 1 to 5</td></tr>
        <tr><td><span class="feat-name">seed</span></td><td>The random seed used for that run; use a different, non-zero seed for each run</td></tr>
        <tr><td><span class="feat-name">Accuracy</span> <span class="feat-name">MacroAUC</span></td><td>Test-set accuracy and ROC AUC, between 0 and 1</td></tr>
        <tr><td><span class="feat-name">Precision_1</span> <span class="feat-name">Recall_1</span> <span class="feat-name">F1_1</span></td><td>Test-set precision, recall, and F1 for class 1 (positive), between 0 and 1</td></tr>
        <tr><td><span class="feat-name">elapsed_hr</span></td><td>Training time in hours</td></tr>
        <tr><td><span class="feat-name">gpu_hrs</span></td><td>GPU hours used; 0 if you trained on a CPU</td></tr>
        <tr><td><span class="feat-name">cpu_pct</span></td><td>CPU utilization (%)</td></tr>
      </tbody>
    </table>
    <div class="mm-upload">
      <button class="mm-btn" onclick="downloadMyModelTemplate()">&darr; Download template (my_model.csv)</button>
      <label class="mm-btn mm-btn-primary">&uarr; Upload my_model.csv<input type="file" accept=".csv,text/csv" hidden onchange="handleMyModelUpload(this)"></label>
      <div class="mm-upload-status"></div>
    </div>
    <div class="ds-note">Uploading updates your model card (My Model) on the Models tab, so you can compare it with the benchmark models. Upload again after each improvement.</div>
  </div>

  <!-- ── Report ── -->
  <div class="rq-section-label">Report</div>
  <div class="card-grid cols-2" style="margin-bottom:28px">
    <div class="card">
      <div class="ds-card-title">📝 What to include</div>
      <ul class="ch-reflect-list">
        <li>Everything you explored, including ideas that did not work</li>
        <li>Your reflections on each task: what you expected, what happened, and why</li>
        <li>Your best model for this task, design choices, feature engineering, and parameters</li>
        <li>A table of every setting needed to rerun your best model, like the example on right</li>
      </ul>
      <div class="ds-text" style="margin-top:8px">The goal: someone with only your report can replicate your results.</div>
    </div>
    <div class="card">
      <div class="ds-card-title">📋 Example settings table</div>
      <table class="comp-table dl-columns">
        <thead><tr><th>Setting</th><th>Your value</th></tr></thead>
        <tbody>
          <tr><td>Model</td><td>e.g., XGBoost classifier</td></tr>
          <tr><td>Features</td><td>e.g., the 13 baseline features plus your new ones, listed</td></tr>
          <tr><td>Feature engineering</td><td>How each new feature is computed</td></tr>
          <tr><td>Hyperparameters</td><td>Every value you set</td></tr>
          <tr><td>Random seeds</td><td>e.g., 43, 44, 45, 46, 47</td></tr>
          <tr><td>Training data</td><td>Number of training patients; same validation and test sets</td></tr>
          <tr><td>Software</td><td>Python and library versions</td></tr>
          <tr><td>Hardware</td><td>CPU or GPU used</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── Hand in ── -->
  <div class="rq-section-label">What to Hand In</div>
  <div class="card">
    <ul class="dl-checklist">
      <li>Your report, including your reflections on each task: submit it to D2L</li>
      <li>Your best model&rsquo;s performance over 5 runs (my_model.csv): upload it in the Results File section above</li>
    </ul>
  </div>

</div>
</div><!-- /tab-challenges -->


<script>
// ── Dataset stats injected from Python ──────────────────────────────────────
const DATASET_STATS = {{ stats }};

// ── My Model metrics — re-read from CSV on every page load ───────────────────
// An upload saved in this browser (Models or Deliverables tab) overrides the built-in file.
const MY_MODEL_UPLOAD_KEY = 'myModelUpload';
let MY_MODEL_UPLOAD = null;
try { MY_MODEL_UPLOAD = JSON.parse(localStorage.getItem(MY_MODEL_UPLOAD_KEY)); } catch (e) { MY_MODEL_UPLOAD = null; }
const MY_MODEL_STATS = (MY_MODEL_UPLOAD && MY_MODEL_UPLOAD.stats) || {{ my_model }};

// ── Per-iteration test metrics (all 30 runs, all 7 models) ───────────────────
// Source: ai-sandbox/analysis/results/metrics/*_metrics_iterN.csv (positive-class precision, recall, F1)
const ITER_DATA = {"iter1":{"random_forest":{"auc":0.7343,"accuracy":0.6762,"precision":0.6739,"recall":0.6941,"f1":0.6839},"xgboost":{"auc":0.7014,"accuracy":0.639,"precision":0.6421,"recall":0.6428,"f1":0.6424},"tfidf_logreg":{"auc":0.7841,"accuracy":0.7114,"precision":0.8473,"recall":0.6077,"f1":0.7078},"lstm":{"auc":0.798,"accuracy":0.7366,"precision":0.7884,"recall":0.6535,"f1":0.7146},"gru":{"auc":0.8012,"accuracy":0.7399,"precision":0.7793,"recall":0.6759,"f1":0.7239},"transformer":{"auc":0.799,"accuracy":0.7312,"precision":0.7849,"recall":0.6439,"f1":0.7074},"clinicalbert":{"auc":0.7987,"accuracy":0.719,"precision":0.8368,"recall":0.6354,"f1":0.7223}},"iter2":{"random_forest":{"auc":0.7318,"accuracy":0.673,"precision":0.6763,"recall":0.6749,"f1":0.6756},"xgboost":{"auc":0.7021,"accuracy":0.6454,"precision":0.6488,"recall":0.6481,"f1":0.6485},"tfidf_logreg":{"auc":0.792,"accuracy":0.7199,"precision":0.8262,"recall":0.6316,"f1":0.7159},"lstm":{"auc":0.7937,"accuracy":0.7221,"precision":0.807,"recall":0.5904,"f1":0.6819},"gru":{"auc":0.7967,"accuracy":0.7172,"precision":0.7834,"recall":0.6075,"f1":0.6843},"transformer":{"auc":0.7922,"accuracy":0.7107,"precision":0.8569,"recall":0.5123,"f1":0.6412},"clinicalbert":{"auc":0.8053,"accuracy":0.7171,"precision":0.8635,"recall":0.5865,"f1":0.6985}},"iter3":{"random_forest":{"auc":0.7157,"accuracy":0.6557,"precision":0.6616,"recall":0.6503,"f1":0.6559},"xgboost":{"auc":0.6857,"accuracy":0.6352,"precision":0.6422,"recall":0.6257,"f1":0.6338},"tfidf_logreg":{"auc":0.7849,"accuracy":0.7097,"precision":0.8235,"recall":0.6109,"f1":0.7015},"lstm":{"auc":0.7837,"accuracy":0.7107,"precision":0.7491,"recall":0.6417,"f1":0.6912},"gru":{"auc":0.7852,"accuracy":0.7091,"precision":0.7538,"recall":0.6289,"f1":0.6857},"transformer":{"auc":0.778,"accuracy":0.707,"precision":0.7269,"recall":0.6717,"f1":0.6982},"clinicalbert":{"auc":0.7838,"accuracy":0.6989,"precision":0.8209,"recall":0.5891,"f1":0.686}},"iter4":{"random_forest":{"auc":0.735,"accuracy":0.6746,"precision":0.6865,"recall":0.6535,"f1":0.6696},"xgboost":{"auc":0.7183,"accuracy":0.6503,"precision":0.6589,"recall":0.6364,"f1":0.6474},"tfidf_logreg":{"auc":0.7855,"accuracy":0.6941,"precision":0.8072,"recall":0.5961,"f1":0.6858},"lstm":{"auc":0.8009,"accuracy":0.7356,"precision":0.7916,"recall":0.646,"f1":0.7114},"gru":{"auc":0.8044,"accuracy":0.7275,"precision":0.7363,"recall":0.7166,"f1":0.7263},"transformer":{"auc":0.7936,"accuracy":0.7156,"precision":0.7636,"recall":0.6321,"f1":0.6916},"clinicalbert":{"auc":0.8019,"accuracy":0.72,"precision":0.8137,"recall":0.6484,"f1":0.7217}},"iter5":{"random_forest":{"auc":0.7408,"accuracy":0.6865,"precision":0.6993,"recall":0.6642,"f1":0.6813},"xgboost":{"auc":0.7147,"accuracy":0.6562,"precision":0.6652,"recall":0.6417,"f1":0.6532},"tfidf_logreg":{"auc":0.7968,"accuracy":0.7233,"precision":0.8516,"recall":0.6189,"f1":0.7168},"lstm":{"auc":0.7989,"accuracy":0.7232,"precision":0.7844,"recall":0.6225,"f1":0.6941},"gru":{"auc":0.8014,"accuracy":0.7221,"precision":0.7869,"recall":0.616,"f1":0.6911},"transformer":{"auc":0.794,"accuracy":0.7248,"precision":0.7845,"recall":0.6267,"f1":0.6968},"clinicalbert":{"auc":0.8019,"accuracy":0.715,"precision":0.8282,"recall":0.6262,"f1":0.7132}},"iter6":{"random_forest":{"auc":0.7349,"accuracy":0.6832,"precision":0.6879,"recall":0.6813,"f1":0.6846},"xgboost":{"auc":0.7083,"accuracy":0.6568,"precision":0.6602,"recall":0.6588,"f1":0.6595},"tfidf_logreg":{"auc":0.7824,"accuracy":0.718,"precision":0.8384,"recall":0.6171,"f1":0.7109},"lstm":{"auc":0.7965,"accuracy":0.7232,"precision":0.7769,"recall":0.6332,"f1":0.6977},"gru":{"auc":0.7964,"accuracy":0.7242,"precision":0.788,"recall":0.6203,"f1":0.6942},"transformer":{"auc":0.7914,"accuracy":0.7118,"precision":0.7621,"recall":0.6235,"f1":0.6859},"clinicalbert":{"auc":0.7947,"accuracy":0.7061,"precision":0.813,"recall":0.6196,"f1":0.7032}},"iter7":{"random_forest":{"auc":0.7323,"accuracy":0.673,"precision":0.6834,"recall":0.6556,"f1":0.6692},"xgboost":{"auc":0.713,"accuracy":0.6632,"precision":0.673,"recall":0.6471,"f1":0.6598},"tfidf_logreg":{"auc":0.7927,"accuracy":0.7264,"precision":0.8505,"recall":0.6321,"f1":0.7252},"lstm":{"auc":0.7993,"accuracy":0.7258,"precision":0.7961,"recall":0.6139,"f1":0.6932},"gru":{"auc":0.7959,"accuracy":0.7232,"precision":0.7851,"recall":0.6214,"f1":0.6937},"transformer":{"auc":0.7975,"accuracy":0.7232,"precision":0.7883,"recall":0.6171,"f1":0.6923},"clinicalbert":{"auc":0.7989,"accuracy":0.7172,"precision":0.8171,"recall":0.6506,"f1":0.7244}},"iter8":{"random_forest":{"auc":0.733,"accuracy":0.674,"precision":0.6805,"recall":0.6674,"f1":0.6739},"xgboost":{"auc":0.7022,"accuracy":0.6417,"precision":0.6511,"recall":0.6246,"f1":0.6376},"tfidf_logreg":{"auc":0.7636,"accuracy":0.683,"precision":0.8163,"recall":0.5697,"f1":0.671},"lstm":{"auc":0.7811,"accuracy":0.7037,"precision":0.7781,"recall":0.5775,"f1":0.663},"gru":{"auc":0.7816,"accuracy":0.7107,"precision":0.779,"recall":0.5957,"f1":0.6752},"transformer":{"auc":0.776,"accuracy":0.701,"precision":0.7218,"recall":0.6631,"f1":0.6912},"clinicalbert":{"auc":0.7719,"accuracy":0.6907,"precision":0.8277,"recall":0.5746,"f1":0.6783}},"iter9":{"random_forest":{"auc":0.7309,"accuracy":0.6778,"precision":0.6895,"recall":0.6578,"f1":0.6732},"xgboost":{"auc":0.7193,"accuracy":0.6649,"precision":0.6741,"recall":0.6503,"f1":0.6619},"tfidf_logreg":{"auc":0.7807,"accuracy":0.7022,"precision":0.8363,"recall":0.5762,"f1":0.6823},"lstm":{"auc":0.7848,"accuracy":0.7129,"precision":0.7554,"recall":0.6374,"f1":0.6914},"gru":{"auc":0.7828,"accuracy":0.7005,"precision":0.7617,"recall":0.5914,"f1":0.6659},"transformer":{"auc":0.7822,"accuracy":0.7097,"precision":0.7629,"recall":0.616,"f1":0.6817},"clinicalbert":{"auc":0.7859,"accuracy":0.6994,"precision":0.7927,"recall":0.6208,"f1":0.6963}},"iter10":{"random_forest":{"auc":0.7459,"accuracy":0.6767,"precision":0.685,"recall":0.6652,"f1":0.675},"xgboost":{"auc":0.709,"accuracy":0.6595,"precision":0.6681,"recall":0.646,"f1":0.6569},"tfidf_logreg":{"auc":0.7865,"accuracy":0.6966,"precision":0.8448,"recall":0.5673,"f1":0.6788},"lstm":{"auc":0.7907,"accuracy":0.7188,"precision":0.7924,"recall":0.6,"f1":0.6829},"gru":{"auc":0.7917,"accuracy":0.7221,"precision":0.8,"recall":0.5989,"f1":0.685},"transformer":{"auc":0.7877,"accuracy":0.714,"precision":0.7485,"recall":0.6524,"f1":0.6971},"clinicalbert":{"auc":0.7909,"accuracy":0.7082,"precision":0.8276,"recall":0.6109,"f1":0.7029}},"iter11":{"random_forest":{"auc":0.7275,"accuracy":0.6665,"precision":0.6646,"recall":0.6845,"f1":0.6744},"xgboost":{"auc":0.7057,"accuracy":0.6454,"precision":0.6413,"recall":0.6749,"f1":0.6576},"tfidf_logreg":{"auc":0.7735,"accuracy":0.7036,"precision":0.8132,"recall":0.6074,"f1":0.6954},"lstm":{"auc":0.7881,"accuracy":0.7151,"precision":0.7854,"recall":0.5989,"f1":0.6796},"gru":{"auc":0.79,"accuracy":0.7086,"precision":0.7702,"recall":0.6021,"f1":0.6759},"transformer":{"auc":0.7869,"accuracy":0.7194,"precision":0.765,"recall":0.6406,"f1":0.6973},"clinicalbert":{"auc":0.7954,"accuracy":0.7256,"precision":0.8081,"recall":0.6654,"f1":0.7299}},"iter12":{"random_forest":{"auc":0.7441,"accuracy":0.6854,"precision":0.6917,"recall":0.6791,"f1":0.6854},"xgboost":{"auc":0.7118,"accuracy":0.6557,"precision":0.6558,"recall":0.6684,"f1":0.6621},"tfidf_logreg":{"auc":0.7934,"accuracy":0.7085,"precision":0.8409,"recall":0.5938,"f1":0.6961},"lstm":{"auc":0.8093,"accuracy":0.728,"precision":0.767,"recall":0.662,"f1":0.7107},"gru":{"auc":0.8097,"accuracy":0.7312,"precision":0.7642,"recall":0.6759,"f1":0.7174},"transformer":{"auc":0.8006,"accuracy":0.7129,"precision":0.8433,"recall":0.5294,"f1":0.6505},"clinicalbert":{"auc":0.8135,"accuracy":0.7273,"precision":0.8336,"recall":0.6432,"f1":0.7261}},"iter13":{"random_forest":{"auc":0.732,"accuracy":0.6735,"precision":0.6842,"recall":0.6556,"f1":0.6696},"xgboost":{"auc":0.7153,"accuracy":0.6627,"precision":0.6715,"recall":0.6492,"f1":0.6601},"tfidf_logreg":{"auc":0.7886,"accuracy":0.7134,"precision":0.8379,"recall":0.6084,"f1":0.705},"lstm":{"auc":0.7927,"accuracy":0.7199,"precision":0.7938,"recall":0.6011,"f1":0.6841},"gru":{"auc":0.7962,"accuracy":0.728,"precision":0.7766,"recall":0.6471,"f1":0.706},"transformer":{"auc":0.7931,"accuracy":0.7307,"precision":0.7759,"recall":0.6556,"f1":0.7107},"clinicalbert":{"auc":0.7985,"accuracy":0.7245,"precision":0.8169,"recall":0.658,"f1":0.7289}},"iter14":{"random_forest":{"auc":0.7273,"accuracy":0.6848,"precision":0.701,"recall":0.6545,"f1":0.677},"xgboost":{"auc":0.6985,"accuracy":0.6525,"precision":0.6601,"recall":0.6417,"f1":0.6508},"tfidf_logreg":{"auc":0.7973,"accuracy":0.7216,"precision":0.8507,"recall":0.6221,"f1":0.7186},"lstm":{"auc":0.796,"accuracy":0.7215,"precision":0.7696,"recall":0.6396,"f1":0.6986},"gru":{"auc":0.7969,"accuracy":0.7248,"precision":0.7686,"recall":0.6503,"f1":0.7045},"transformer":{"auc":0.7905,"accuracy":0.7129,"precision":0.7522,"recall":0.6428,"f1":0.6932},"clinicalbert":{"auc":0.801,"accuracy":0.7104,"precision":0.8284,"recall":0.6221,"f1":0.7106}},"iter15":{"random_forest":{"auc":0.7293,"accuracy":0.674,"precision":0.6904,"recall":0.6417,"f1":0.6652},"xgboost":{"auc":0.7102,"accuracy":0.6589,"precision":0.6756,"recall":0.6235,"f1":0.6485},"tfidf_logreg":{"auc":0.7778,"accuracy":0.7065,"precision":0.8435,"recall":0.6034,"f1":0.7035},"lstm":{"auc":0.789,"accuracy":0.7167,"precision":0.7871,"recall":0.6011,"f1":0.6816},"gru":{"auc":0.7894,"accuracy":0.7172,"precision":0.7515,"recall":0.6567,"f1":0.7009},"transformer":{"auc":0.7836,"accuracy":0.708,"precision":0.7481,"recall":0.6353,"f1":0.6871},"clinicalbert":{"auc":0.7873,"accuracy":0.7065,"precision":0.8367,"recall":0.6107,"f1":0.706}},"iter16":{"random_forest":{"auc":0.7298,"accuracy":0.6762,"precision":0.6831,"recall":0.6684,"f1":0.6757},"xgboost":{"auc":0.7052,"accuracy":0.6541,"precision":0.6601,"recall":0.6481,"f1":0.6541},"tfidf_logreg":{"auc":0.802,"accuracy":0.7253,"precision":0.8726,"recall":0.6101,"f1":0.7181},"lstm":{"auc":0.7986,"accuracy":0.7237,"precision":0.8026,"recall":0.6,"f1":0.6867},"gru":{"auc":0.8033,"accuracy":0.7329,"precision":0.7743,"recall":0.6642,"f1":0.715},"transformer":{"auc":0.7983,"accuracy":0.7161,"precision":0.7994,"recall":0.584,"f1":0.6749},"clinicalbert":{"auc":0.795,"accuracy":0.7069,"precision":0.8286,"recall":0.6163,"f1":0.7069}},"iter17":{"random_forest":{"auc":0.7186,"accuracy":0.6654,"precision":0.6674,"recall":0.6717,"f1":0.6695},"xgboost":{"auc":0.6973,"accuracy":0.6476,"precision":0.65,"recall":0.6535,"f1":0.6517},"tfidf_logreg":{"auc":0.7711,"accuracy":0.6977,"precision":0.8214,"recall":0.5914,"f1":0.6877},"lstm":{"auc":0.7737,"accuracy":0.6989,"precision":0.7302,"recall":0.6396,"f1":0.6819},"gru":{"auc":0.7792,"accuracy":0.7032,"precision":0.7362,"recall":0.6417,"f1":0.6857},"transformer":{"auc":0.7682,"accuracy":0.6978,"precision":0.7295,"recall":0.6374,"f1":0.6804},"clinicalbert":{"auc":0.7781,"accuracy":0.7097,"precision":0.7991,"recall":0.6469,"f1":0.715}},"iter18":{"random_forest":{"auc":0.7128,"accuracy":0.6584,"precision":0.6631,"recall":0.6567,"f1":0.6599},"xgboost":{"auc":0.685,"accuracy":0.6363,"precision":0.6402,"recall":0.6374,"f1":0.6388},"tfidf_logreg":{"auc":0.7585,"accuracy":0.6943,"precision":0.8112,"recall":0.5807,"f1":0.6769},"lstm":{"auc":0.7764,"accuracy":0.7107,"precision":0.783,"recall":0.5904,"f1":0.6732},"gru":{"auc":0.7742,"accuracy":0.6956,"precision":0.7483,"recall":0.5979,"f1":0.6647},"transformer":{"auc":0.7697,"accuracy":0.6951,"precision":0.8043,"recall":0.523,"f1":0.6338},"clinicalbert":{"auc":0.7836,"accuracy":0.7164,"precision":0.807,"recall":0.6383,"f1":0.7128}},"iter19":{"random_forest":{"auc":0.7149,"accuracy":0.6595,"precision":0.6723,"recall":0.6342,"f1":0.6527},"xgboost":{"auc":0.7051,"accuracy":0.6492,"precision":0.6607,"recall":0.6267,"f1":0.6432},"tfidf_logreg":{"auc":0.7813,"accuracy":0.7009,"precision":0.8389,"recall":0.5727,"f1":0.6807},"lstm":{"auc":0.7802,"accuracy":0.707,"precision":0.7513,"recall":0.6267,"f1":0.6834},"gru":{"auc":0.7763,"accuracy":0.7016,"precision":0.7729,"recall":0.5786,"f1":0.6618},"transformer":{"auc":0.7754,"accuracy":0.6999,"precision":0.7628,"recall":0.5882,"f1":0.6643},"clinicalbert":{"auc":0.7939,"accuracy":0.7044,"precision":0.8107,"recall":0.6119,"f1":0.6974}},"iter20":{"random_forest":{"auc":0.7132,"accuracy":0.6686,"precision":0.6818,"recall":0.6439,"f1":0.6623},"xgboost":{"auc":0.6837,"accuracy":0.6449,"precision":0.6547,"recall":0.6267,"f1":0.6404},"tfidf_logreg":{"auc":0.7736,"accuracy":0.7079,"precision":0.821,"recall":0.6196,"f1":0.7062},"lstm":{"auc":0.7755,"accuracy":0.7032,"precision":0.7633,"recall":0.5968,"f1":0.6699},"gru":{"auc":0.7719,"accuracy":0.7059,"precision":0.7526,"recall":0.6214,"f1":0.6807},"transformer":{"auc":0.7707,"accuracy":0.6951,"precision":0.7705,"recall":0.5636,"f1":0.651},"clinicalbert":{"auc":0.7742,"accuracy":0.6868,"precision":0.7954,"recall":0.6022,"f1":0.6855}},"iter21":{"random_forest":{"auc":0.722,"accuracy":0.6713,"precision":0.6745,"recall":0.6738,"f1":0.6742},"xgboost":{"auc":0.6969,"accuracy":0.6476,"precision":0.6523,"recall":0.646,"f1":0.6491},"tfidf_logreg":{"auc":0.7835,"accuracy":0.6974,"precision":0.816,"recall":0.595,"f1":0.6882},"lstm":{"auc":0.7813,"accuracy":0.714,"precision":0.7696,"recall":0.6182,"f1":0.6856},"gru":{"auc":0.7819,"accuracy":0.708,"precision":0.7456,"recall":0.6396,"f1":0.6885},"transformer":{"auc":0.7821,"accuracy":0.714,"precision":0.7467,"recall":0.6556,"f1":0.6982},"clinicalbert":{"auc":0.7958,"accuracy":0.719,"precision":0.8211,"recall":0.6385,"f1":0.7184}},"iter22":{"random_forest":{"auc":0.72,"accuracy":0.6697,"precision":0.6804,"recall":0.6513,"f1":0.6656},"xgboost":{"auc":0.6985,"accuracy":0.6379,"precision":0.6444,"recall":0.6299,"f1":0.6371},"tfidf_logreg":{"auc":0.793,"accuracy":0.7171,"precision":0.8498,"recall":0.6148,"f1":0.7135},"lstm":{"auc":0.7976,"accuracy":0.7285,"precision":0.8,"recall":0.616,"f1":0.6961},"gru":{"auc":0.7986,"accuracy":0.728,"precision":0.7738,"recall":0.6513,"f1":0.7073},"transformer":{"auc":0.7908,"accuracy":0.7232,"precision":0.7997,"recall":0.6021,"f1":0.687},"clinicalbert":{"auc":0.808,"accuracy":0.7192,"precision":0.8391,"recall":0.6309,"f1":0.7202}},"iter23":{"random_forest":{"auc":0.7527,"accuracy":0.6913,"precision":0.7028,"recall":0.6727,"f1":0.6874},"xgboost":{"auc":0.7284,"accuracy":0.6746,"precision":0.6824,"recall":0.6642,"f1":0.6732},"tfidf_logreg":{"auc":0.8023,"accuracy":0.7188,"precision":0.8513,"recall":0.6081,"f1":0.7094},"lstm":{"auc":0.7956,"accuracy":0.7237,"precision":0.7933,"recall":0.6118,"f1":0.6908},"gru":{"auc":0.7948,"accuracy":0.7242,"precision":0.7953,"recall":0.6107,"f1":0.6909},"transformer":{"auc":0.7871,"accuracy":0.7053,"precision":0.8226,"recall":0.5305,"f1":0.645},"clinicalbert":{"auc":0.8003,"accuracy":0.7133,"precision":0.8298,"recall":0.619,"f1":0.7091}},"iter24":{"random_forest":{"auc":0.7148,"accuracy":0.6632,"precision":0.6726,"recall":0.6481,"f1":0.6601},"xgboost":{"auc":0.6964,"accuracy":0.6422,"precision":0.6478,"recall":0.6374,"f1":0.6426},"tfidf_logreg":{"auc":0.7855,"accuracy":0.7123,"precision":0.8207,"recall":0.6191,"f1":0.7058},"lstm":{"auc":0.7834,"accuracy":0.7091,"precision":0.7585,"recall":0.6214,"f1":0.6831},"gru":{"auc":0.7846,"accuracy":0.7118,"precision":0.746,"recall":0.6503,"f1":0.6949},"transformer":{"auc":0.78,"accuracy":0.7091,"precision":0.745,"recall":0.6439,"f1":0.6908},"clinicalbert":{"auc":0.7851,"accuracy":0.7095,"precision":0.8133,"recall":0.6216,"f1":0.7046}},"iter25":{"random_forest":{"auc":0.7362,"accuracy":0.6686,"precision":0.6797,"recall":0.6492,"f1":0.6641},"xgboost":{"auc":0.6962,"accuracy":0.639,"precision":0.6494,"recall":0.6182,"f1":0.6334},"tfidf_logreg":{"auc":0.7941,"accuracy":0.71,"precision":0.835,"recall":0.6102,"f1":0.7051},"lstm":{"auc":0.7945,"accuracy":0.7264,"precision":0.7846,"recall":0.631,"f1":0.6995},"gru":{"auc":0.7912,"accuracy":0.7264,"precision":0.8075,"recall":0.6011,"f1":0.6891},"transformer":{"auc":0.7915,"accuracy":0.7199,"precision":0.822,"recall":0.5679,"f1":0.6717},"clinicalbert":{"auc":0.7942,"accuracy":0.7059,"precision":0.8225,"recall":0.6151,"f1":0.7038}},"iter26":{"random_forest":{"auc":0.7195,"accuracy":0.6665,"precision":0.6767,"recall":0.6492,"f1":0.6627},"xgboost":{"auc":0.7023,"accuracy":0.6492,"precision":0.6636,"recall":0.6182,"f1":0.6401},"tfidf_logreg":{"auc":0.7659,"accuracy":0.686,"precision":0.83,"recall":0.5667,"f1":0.6735},"lstm":{"auc":0.7844,"accuracy":0.707,"precision":0.773,"recall":0.5936,"f1":0.6715},"gru":{"auc":0.7873,"accuracy":0.701,"precision":0.7464,"recall":0.6171,"f1":0.6756},"transformer":{"auc":0.7814,"accuracy":0.7064,"precision":0.805,"recall":0.5519,"f1":0.6548},"clinicalbert":{"auc":0.775,"accuracy":0.6909,"precision":0.8163,"recall":0.5926,"f1":0.6867}},"iter27":{"random_forest":{"auc":0.7427,"accuracy":0.6929,"precision":0.7011,"recall":0.6824,"f1":0.6916},"xgboost":{"auc":0.7032,"accuracy":0.6508,"precision":0.6593,"recall":0.6374,"f1":0.6482},"tfidf_logreg":{"auc":0.7717,"accuracy":0.689,"precision":0.8196,"recall":0.5927,"f1":0.6879},"lstm":{"auc":0.7913,"accuracy":0.7215,"precision":0.7602,"recall":0.6545,"f1":0.7034},"gru":{"auc":0.7908,"accuracy":0.7215,"precision":0.7724,"recall":0.6353,"f1":0.6972},"transformer":{"auc":0.7863,"accuracy":0.7167,"precision":0.8069,"recall":0.5765,"f1":0.6725},"clinicalbert":{"auc":0.7787,"accuracy":0.7003,"precision":0.8376,"recall":0.5976,"f1":0.6975}},"iter28":{"random_forest":{"auc":0.7246,"accuracy":0.6713,"precision":0.6819,"recall":0.6535,"f1":0.6674},"xgboost":{"auc":0.7128,"accuracy":0.6546,"precision":0.6701,"recall":0.6214,"f1":0.6448},"tfidf_logreg":{"auc":0.7748,"accuracy":0.7087,"precision":0.8331,"recall":0.6071,"f1":0.7023},"lstm":{"auc":0.7987,"accuracy":0.7264,"precision":0.7758,"recall":0.6439,"f1":0.7037},"gru":{"auc":0.8017,"accuracy":0.7215,"precision":0.7596,"recall":0.6556,"f1":0.7038},"transformer":{"auc":0.7971,"accuracy":0.7183,"precision":0.7706,"recall":0.6289,"f1":0.6926},"clinicalbert":{"auc":0.7938,"accuracy":0.7156,"precision":0.8241,"recall":0.6326,"f1":0.7158}},"iter29":{"random_forest":{"auc":0.7164,"accuracy":0.6557,"precision":0.6592,"recall":0.6578,"f1":0.6585},"xgboost":{"auc":0.7051,"accuracy":0.6487,"precision":0.6547,"recall":0.6428,"f1":0.6487},"tfidf_logreg":{"auc":0.7813,"accuracy":0.7063,"precision":0.8365,"recall":0.6002,"f1":0.6989},"lstm":{"auc":0.7922,"accuracy":0.7097,"precision":0.7715,"recall":0.6032,"f1":0.6771},"gru":{"auc":0.7941,"accuracy":0.7124,"precision":0.7652,"recall":0.6203,"f1":0.6852},"transformer":{"auc":0.7877,"accuracy":0.7097,"precision":0.7976,"recall":0.569,"f1":0.6642},"clinicalbert":{"auc":0.7894,"accuracy":0.7069,"precision":0.8143,"recall":0.6271,"f1":0.7086}},"iter30":{"random_forest":{"auc":0.7206,"accuracy":0.6649,"precision":0.6714,"recall":0.6578,"f1":0.6645},"xgboost":{"auc":0.6961,"accuracy":0.6427,"precision":0.6472,"recall":0.6417,"f1":0.6445},"tfidf_logreg":{"auc":0.7826,"accuracy":0.7017,"precision":0.8456,"recall":0.5821,"f1":0.6896},"lstm":{"auc":0.7826,"accuracy":0.7102,"precision":0.7439,"recall":0.6492,"f1":0.6933},"gru":{"auc":0.7775,"accuracy":0.7026,"precision":0.754,"recall":0.6096,"f1":0.6742},"transformer":{"auc":0.7766,"accuracy":0.7016,"precision":0.7449,"recall":0.6214,"f1":0.6776},"clinicalbert":{"auc":0.7728,"accuracy":0.6866,"precision":0.7906,"recall":0.6111,"f1":0.6894}}};

// ── Model data ──────────────────────────────────────────────────────────────
// Mean of one metric over the 30 iterations in ITER_DATA, so the cards always match it.
function iterMean(modelId, metric) {
  const vals = Object.values(ITER_DATA).map(it => it[modelId][metric]);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
function iterMetrics(modelId) {
  return Object.fromEntries(['auc', 'accuracy', 'f1', 'precision', 'recall'].map(k => [k, iterMean(modelId, k)]));
}

// Timing and GPU hours: 30-run means from ai-sandbox/analysis/logs/resource_usage.csv
// (NVIDIA A100 40 GB), including cpuPct (CPU utilization snapshot, %). Random Forest and XGBoost
// are timed together.
const MODELS = [
  {
    id: "random_forest",
    name: "Random Forest",
    icon: "🌲",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#16a34a",
    description: "An ensemble of 200 decision trees, each trained on a random sample of patients and features. Averaging many trees smooths out individual errors and reduces overfitting.",
    input: "13 structured features",
    ...iterMetrics("random_forest"),
    trainTimeSec: 4,
    gpuHrs: 0,
    cpuPct: 7.7,
    gpuRequired: false,
    params: "200 trees",
    interpretable: true,
    pros: [
      "Built-in feature importance shows what the model relies on",
      "Robust to outliers and needs no feature scaling",
      "Trains in seconds on any laptop (~4 sec per iteration, together with XGBoost)",
      "No GPU required"
    ],
    cons: [
      "Uses only the 13 structured features, with no medication text",
      "AUC 0.729, well below the text-based and neural models (about 0.78 to 0.79)",
      "Leans heavily on age and insurance coverage (see the Feature Engineering tab)",
      "Can take a lot of memory with many trees"
    ]
  },
  {
    id: "xgboost",
    name: "XGBoost",
    icon: "⚡",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#ea580c",
    description: "Gradient-boosted trees that train one after another, each new tree correcting the mistakes of the previous ones. The benchmark runs it with XGBoost's default settings.",
    input: "13 structured features",
    ...iterMetrics("xgboost"),
    trainTimeSec: 4,
    gpuHrs: 0,
    cpuPct: 7.7,
    gpuRequired: false,
    params: "100 boosted trees (defaults)",
    interpretable: true,
    pros: [
      "Often the strongest model family for tabular data",
      "Built-in regularization reduces overfitting",
      "Trains in seconds on CPU",
      "Feature importance available via gain or SHAP values"
    ],
    cons: [
      "Lowest F1 (0.649) and AUC (0.704) of the seven models in this benchmark",
      "Runs with default hyperparameters, so there is room to tune",
      "Uses only the 13 structured features, with no medication text",
      "Cannot use the order of events"
    ]
  },
  {
    id: "tfidf_logreg",
    name: "TF-IDF + LogReg",
    icon: "📝",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#0891b2",
    description: "Logistic regression on bag-of-words medication text. TF-IDF turns each patient's medication list into word and word-pair weights, and the model finds a linear boundary. Its regularization strength is picked on the validation set.",
    input: "Raw medication text",
    ...iterMetrics("tfidf_logreg"),
    trainTimeSec: 2,
    gpuHrs: 0,
    cpuPct: 7.9,
    gpuRequired: false,
    params: "Up to 20K word weights",
    interpretable: true,
    pros: [
      "Best CPU model: F1 0.699, second only to ClinicalBERT",
      "Highest precision of all seven models (0.834): when it predicts positive, it is usually right",
      "Fully interpretable: coefficients show which drug words push predictions up or down",
      "Trains in under 2 seconds"
    ],
    cons: [
      "Low recall (0.601): misses about 40% of positive patients",
      "Scored only on patients with medication records",
      "Bag-of-words ignores word order",
      "Medication text can partly reveal the label (see the Dataset tab)"
    ]
  },
  {
    id: "lstm",
    name: "LSTM",
    icon: "🔁",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#7c3aed",
    description: "Long Short-Term Memory networks read a sequence one step at a time, using gates to decide what to remember. Here each patient's 10-step sequence holds one real step, so the memory has little to work with.",
    input: "32 features as a 10-step sequence",
    ...iterMetrics("lstm"),
    trainTimeSec: 17,
    gpuHrs: 0.0048,
    cpuPct: 16.4,
    gpuRequired: true,
    params: "~58K parameters",
    interpretable: false,
    pros: [
      "AUC 0.790, close to the best models",
      "Uses both structured and medication text features",
      "Most stable of the GPU models (F1 std 0.012)",
      "Small model: about 58K parameters"
    ],
    cons: [
      "Only one real step per patient, so sequence memory adds little",
      "F1 0.689, below TF-IDF + LogReg, which runs on a CPU in seconds",
      "Requires a GPU for practical training",
      "Black box: no direct feature importance"
    ]
  },
  {
    id: "gru",
    name: "GRU",
    icon: "⚙️",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#9333ea",
    description: "Gated Recurrent Units are a lighter version of LSTM with fewer gates, so they train faster while keeping most of its memory ability. Like the LSTM, it sees one real step per patient here.",
    input: "32 features as a 10-step sequence",
    ...iterMetrics("gru"),
    trainTimeSec: 14,
    gpuHrs: 0.0038,
    cpuPct: 17.5,
    gpuRequired: true,
    params: "~19K parameters",
    interpretable: false,
    pros: [
      "Fastest GPU model (~14 sec per iteration) and the smallest (about 19K parameters)",
      "AUC 0.791, essentially tied with LSTM and ClinicalBERT",
      "Best recall of the GPU models (0.630)"
    ],
    cons: [
      "Requires a GPU",
      "F1 varies more than LSTM across runs (std 0.016)",
      "Black box with no direct interpretability",
      "Like LSTM, gains little from sequence modeling with a single real step"
    ]
  },
  {
    id: "transformer",
    name: "Transformer",
    icon: "🔮",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#4f46e5",
    description: "A from-scratch Transformer encoder with multi-head self-attention. Instead of reading steps one by one, it compares all steps at once. With one real step per patient, attention has little to compare.",
    input: "32 features as a 10-step sequence (masked)",
    ...iterMetrics("transformer"),
    trainTimeSec: 22,
    gpuHrs: 0.0062,
    cpuPct: 14.8,
    gpuRequired: true,
    params: "~564K parameters",
    interpretable: false,
    pros: [
      "Parallel training: all steps are processed at once",
      "AUC 0.786, in line with the other neural models",
      "Attention weights offer some post-hoc interpretability"
    ],
    cons: [
      "Least stable model: F1 ranges from 0.634 to 0.711 across runs (std 0.020)",
      "Largest custom model (about 564K parameters) for a lower F1 (0.679)",
      "Needs more data and real sequences to shine",
      "Requires a GPU"
    ]
  },
  {
    id: "clinicalbert",
    name: "ClinicalBERT",
    icon: "🧠",
    category: "bert",
    categoryLabel: "Pretrained BERT",
    color: "#1d4ed8",
    description: "Bio_ClinicalBERT, pre-trained on real clinical notes (MIMIC-III), turns each patient's medication text into a 768-number embedding once. A small Transformer head then learns from that embedding plus 32 structured and text features.",
    input: "Medication text embedding + 32 features",
    ...iterMetrics("clinicalbert"),
    trainTimeSec: 36,
    gpuHrs: 0.0100,
    cpuPct: 9.2,
    gpuRequired: true,
    params: "110M frozen (BERT) + 1.3M trained (head)",
    interpretable: false,
    pros: [
      "Highest F1 (0.707) and AUC (0.792) of the seven models",
      "High precision (0.821), second only to TF-IDF + LogReg",
      "Brings clinical language knowledge from pre-training on real notes",
      "Embeddings are computed once and reused, so each run only trains the small head"
    ],
    cons: [
      "BERT itself is not fine-tuned here; only the head on top is trained",
      "Scored only on the 1,445 test patients with medication records",
      "Needs a GPU, plus a one-time embedding step",
      "Hardest to interpret: 110M frozen weights plus a neural head"
    ]
  },
  {
    id: "my_model",
    name: "My Model",
    icon: "🧪",
    category: "my_model",
    categoryLabel: "My Submission",
    color: "#dc2626",
    description: "Your model, averaged over the 5 runs in your my_model.csv. Upload the file on the Deliverables tab, and upload it again after each improvement.",
    input: "Custom feature set",
    auc:       MY_MODEL_STATS.auc,
    accuracy:  MY_MODEL_STATS.accuracy,
    f1:        MY_MODEL_STATS.f1,
    precision: MY_MODEL_STATS.precision,
    recall:    MY_MODEL_STATS.recall,
    // Computational cost from the results file; null means "not reported"
    trainTimeSec: MY_MODEL_STATS.elapsed_hr == null ? null : Math.round(MY_MODEL_STATS.elapsed_hr * 3600 * 10) / 10,
    gpuHrs: MY_MODEL_STATS.gpu_hrs,
    cpuPct: MY_MODEL_STATS.cpu_pct,
    gpuRequired: MY_MODEL_STATS.gpu_hrs > 0,
    params: "—",
    interpretable: null,
    pros: [
      "This is your work — you control every feature and every parameter",
      `Precision ${MY_MODEL_STATS.precision.toFixed(3)} — check vs CPU baselines`,
      "No GPU required in the baseline starter"
    ],
    cons: [
      `F1 ${MY_MODEL_STATS.f1.toFixed(3)} — improve your features or model and re-run all 5 iterations`,
      `Recall ${MY_MODEL_STATS.recall.toFixed(3)} — consider cost-sensitive training to catch more patients`,
      `Accuracy ${MY_MODEL_STATS.accuracy.toFixed(3)}`
    ]
  }
];

// ── Tier thresholds (F1-based) ─────────────────────────────────────────────
const CPU_MODELS   = ['random_forest','xgboost','tfidf_logreg'];
const GPU_MODELS   = ['lstm','gru','transformer','clinicalbert'];
const bestOf       = ids => ids.map(id => MODELS.find(m => m.id === id)).reduce((a, b) => (b.f1 > a.f1 ? b : a));
const CPU_BEST     = bestOf(CPU_MODELS);
const GPU_BEST     = bestOf(GPU_MODELS);
const CPU_BEST_F1  = CPU_BEST.f1;
const GPU_BEST_F1  = GPU_BEST.f1;
const MY_MODEL_F1  = MODELS.find(m => m.id === 'my_model').f1;
const gpuUnlocked  = MY_MODEL_F1 > CPU_BEST_F1;
const GOOD_F1      = 0.75;   // improvement goal: above this counts as good performance

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  selectedIds: new Set(),
  activeMetric: "f1",
};

// ── Raw data injected from Python ────────────────────────────────────────────
const RAW_DATA = {{ raw }};

// ── Reproducibility tab ───────────────────────────────────────────────────────
const REPRO_MODELS = [
  { key: 'random_forest', label: 'Random Forest', color: '#16a34a', gpu: false },
  { key: 'xgboost',       label: 'XGBoost',       color: '#ea580c', gpu: false },
  { key: 'tfidf_logreg',  label: 'TF-IDF+LR',     color: '#0891b2', gpu: false },
  { key: 'lstm',          label: 'LSTM',           color: '#7c3aed', gpu: true  },
  { key: 'gru',           label: 'GRU',            color: '#9333ea', gpu: true  },
  { key: 'transformer',   label: 'Transformer',    color: '#4f46e5', gpu: true  },
  { key: 'clinicalbert',  label: 'ClinicalBERT',   color: '#1d4ed8', gpu: true  },
];

function buildReproRangeChart() {
  const container = document.getElementById('repro-range-chart');
  if (!container) return;
  const W = container.clientWidth || 700, H = 310;
  const padL = 120, padR = 30, padT = 30, padB = 40;
  const innerW = W - padL - padR;
  const GPU_GAP = 22; // extra vertical gap between CPU and GPU groups
  const cpuCount = REPRO_MODELS.filter(m => !m.gpu).length;
  const gpuCount = REPRO_MODELS.length - cpuCount;
  const rowH = (H - padT - padB - GPU_GAP) / REPRO_MODELS.length;
  const innerH = H - padT - padB;

  // compute per-model min/mean/max F1 across 30 iters
  const stats = REPRO_MODELS.map(m => {
    const vals = Object.values(ITER_DATA).map(it => it[m.key]?.f1).filter(v => v != null);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return { ...m, min: mn, max: mx, mean, vals };
  });

  const allVals = stats.flatMap(s => [s.min, s.max]);
  const domMin = Math.min(...allVals) - 0.005, domMax = Math.max(...allVals) + 0.005;
  const xScale = v => padL + ((v - domMin) / (domMax - domMin)) * innerW;

  // cy with GPU gap applied after cpuCount rows
  const rowCy = i => padT + rowH * i + rowH / 2 + (i >= cpuCount ? GPU_GAP : 0);

  // x-axis ticks
  const ticks = [];
  for (let t = Math.ceil(domMin * 100) / 100; t <= domMax + 0.001; t = Math.round((t + 0.01) * 100) / 100) {
    ticks.push(t);
  }
  const tickLineY2 = rowCy(REPRO_MODELS.length - 1) + rowH / 2;
  const tickLines = ticks.map(t =>
    `<line x1="${xScale(t)}" y1="${padT}" x2="${xScale(t)}" y2="${tickLineY2}" stroke="#334155" stroke-width="1" stroke-dasharray="3,3"/>
     <text x="${xScale(t)}" y="${tickLineY2 + 16}" text-anchor="middle" fill="#94a3b8" font-size="12">${t.toFixed(2)}</text>`
  ).join('');

  // divider y between CPU and GPU rows (in the gap)
  const dividerY = padT + rowH * cpuCount + GPU_GAP / 2;

  const rows = stats.map((s, i) => {
    const cy = rowCy(i);
    const locked = s.gpu && !gpuUnlocked;
    const color = locked ? '#475569' : s.color;
    const xMin = xScale(s.min), xMax = xScale(s.max), xMean = xScale(s.mean);
    const labelColor = locked ? '#475569' : s.color;
    const gpuTag = s.gpu ? ' ⚡' : '';

    if (locked) {
      return `
        <text x="${padL - 8}" y="${cy + 4}" text-anchor="end" fill="${labelColor}" font-size="12.5" font-style="italic" font-weight="600">${s.label}${gpuTag}</text>
        <line x1="${padL}" y1="${cy}" x2="${padL + innerW}" y2="${cy}" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>
        <text x="${padL + innerW/2}" y="${cy + 4}" text-anchor="middle" fill="#334155" font-size="12">🔒 Unlock GPU tier to reveal</text>
      `;
    }
    const dots = s.vals.map(a =>
      `<circle cx="${xScale(a)}" cy="${cy}" r="3" fill="${color}" fill-opacity="0.35"/>`
    ).join('');
    return `
      <text x="${padL - 8}" y="${cy + 4}" text-anchor="end" fill="${labelColor}" font-size="13" font-weight="700">${s.label}${gpuTag}</text>
      ${dots}
      <line x1="${xMin}" y1="${cy}" x2="${xMax}" y2="${cy}" stroke="${color}" stroke-width="3" stroke-linecap="round"/>
      <line x1="${xMin}" y1="${cy - 6}" x2="${xMin}" y2="${cy + 6}" stroke="${color}" stroke-width="2"/>
      <line x1="${xMax}" y1="${cy - 6}" x2="${xMax}" y2="${cy + 6}" stroke="${color}" stroke-width="2"/>
      <circle cx="${xMean}" cy="${cy}" r="5" fill="${color}" stroke="#0f172a" stroke-width="2"/>
    `;
  }).join('');

  // divider line + labels between CPU and GPU sections
  const gpuColor = gpuUnlocked ? '#7c3aed' : '#475569';
  const dividerLine = `
    <line x1="${padL}" y1="${dividerY}" x2="${padL + innerW}" y2="${dividerY}" stroke="#475569" stroke-width="1.5" stroke-dasharray="6,3"/>
    <rect x="${padL}" y="${dividerY - 10}" width="44" height="14" rx="3" fill="#1e293b"/>
    <text x="${padL + 22}" y="${dividerY + 0}" text-anchor="middle" fill="#64748b" font-size="9" font-weight="800">⚡ CPU</text>
    <rect x="${padL + innerW - 52}" y="${dividerY - 10}" width="52" height="14" rx="3" fill="${gpuUnlocked ? '#3b0764' : '#1e293b'}"/>
    <text x="${padL + innerW - 26}" y="${dividerY + 0}" text-anchor="middle" fill="${gpuColor}" font-size="9" font-weight="800">${gpuUnlocked ? '⚡ GPU' : '🔒 GPU'}</text>
  `;

  container.innerHTML = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}">
    <text x="${W/2}" y="16" text-anchor="middle" fill="#94a3b8" font-size="13">F1-Score Range across 30 Iterations — dots = individual runs, bar = min→max, filled circle = mean</text>
    ${tickLines}
    ${dividerLine}
    ${rows}
    <text x="${padL + innerW/2}" y="${H - 4}" text-anchor="middle" fill="#64748b" font-size="12">F1-Score (harmonic mean of precision &amp; recall)</text>
  </svg>`;
}

function updateIterationView(n) {
  document.getElementById('repro-iter-num').textContent = n;
  const data = ITER_DATA['iter' + n];
  if (!data) return;

  // winner by F1, only from unlocked models
  let bestF1 = -1, winner = '';
  REPRO_MODELS.filter(m => !m.gpu || gpuUnlocked).forEach(m => {
    if ((data[m.key]?.f1 || 0) > bestF1) { bestF1 = data[m.key].f1; winner = m.label; }
  });

  // insight
  const insight = document.getElementById('repro-insight');
  if (insight) {
    insight.innerHTML = `<strong>Iteration ${n}:</strong> Best F1 = <strong>${bestF1.toFixed(4)}</strong> by <strong>${winner}</strong>${!gpuUnlocked ? ' <em style="color:#94a3b8;font-size:.85em">(CPU models only — unlock GPU tier to see full ranking)</em>' : ''}.
    Each iteration uses a <em>different random split of patients</em> — notice how rankings shift between runs.`;
  }

  // table
  const tbody = document.getElementById('repro-iter-tbody');
  if (!tbody) return;

  const lockRow = gpuUnlocked
    ? `<tr style="background:#052e16">
    <td colspan="6" style="padding:8px 14px">
      <div class="tier-lock-banner" style="margin:0;padding:10px 16px;background:rgba(22,163,74,.15);border-color:#16a34a">
        <div class="tlb-icon" style="font-size:1.4rem">🔓</div>
        <div class="tlb-body">
          <div class="tlb-title" style="font-size:0.86rem;color:#4ade80">GPU Tier Unlocked!</div>
          <div class="tlb-sub" style="font-size:0.78rem">
            Your F1 <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> beat the CPU best
            (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>). GPU model results are now visible below.
          </div>
        </div>
        <div class="tlb-badge" style="background:#16a34a;color:#fff">✓ UNLOCKED</div>
      </div>
    </td>
  </tr>`
    : `<tr style="background:#0f172a">
    <td colspan="6" style="padding:10px 14px">
      <div class="tier-lock-banner" style="margin:0;padding:12px 16px">
        <div class="tlb-icon" style="font-size:1.4rem">🔒</div>
        <div class="tlb-body">
          <div class="tlb-title" style="font-size:0.86rem">GPU Models Locked</div>
          <div class="tlb-sub" style="font-size:0.78rem">
            Beat CPU best F1 (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>) with your model to unlock GPU rows.
            Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span>
          </div>
        </div>
        <div class="tlb-badge locked">⬆ BEAT CPU BEST</div>
      </div>
    </td>
  </tr>`;

  const rows = REPRO_MODELS.map((m, i) => {
    const locked = m.gpu && !gpuUnlocked;
    const d = data[m.key] || {};
    const dotColor = locked ? '#475569' : m.color;

    // insert lock banner row before first GPU model
    const banner = (i === REPRO_MODELS.findIndex(x => x.gpu)) ? lockRow : '';

    if (locked) {
      return banner + `<tr style="opacity:.4;background:#0f172a">
        <td><span class="repro-dot" style="background:#475569"></span><span style="color:#64748b;font-style:italic">${m.label}</span></td>
        <td style="color:#334155">🔒</td><td style="color:#334155">🔒</td>
        <td style="color:#334155">🔒</td><td style="color:#334155">🔒</td><td style="color:#334155">🔒</td>
      </tr>`;
    }
    const isWinner = d.f1 === bestF1;
    return banner + `<tr${isWinner ? ' class="repro-winner-row"' : ''}>
      <td><span class="repro-dot" style="background:${dotColor}"></span>${m.label}${isWinner ? ' <span class="repro-crown">&#9733;</span>' : ''}</td>
      <td style="${isWinner ? 'font-weight:900;color:#c2410c' : ''}">${d.f1?.toFixed(4) ?? '—'}</td>
      <td>${d.auc?.toFixed(4) ?? '—'}</td>
      <td>${d.accuracy?.toFixed(4) ?? '—'}</td>
      <td>${d.precision?.toFixed(4) ?? '—'}</td>
      <td>${d.recall?.toFixed(4) ?? '—'}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
}

function buildReproSummaryTable() {
  const tbody = document.getElementById('repro-summary-tbody');
  if (!tbody) return;
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std  = arr => { const mu = mean(arr); return Math.sqrt(arr.reduce((s, v) => s + (v - mu) ** 2, 0) / arr.length); };

  const summaryLockRow = gpuUnlocked
    ? `<tr style="background:#052e16">
    <td colspan="8" style="padding:8px 14px">
      <div class="tier-lock-banner" style="margin:0;padding:10px 16px;background:rgba(22,163,74,.15);border-color:#16a34a">
        <div class="tlb-icon" style="font-size:1.3rem">🔓</div>
        <div class="tlb-body">
          <div class="tlb-title" style="font-size:0.87rem;color:#4ade80">GPU Stats Unlocked!</div>
          <div class="tlb-sub" style="font-size:0.78rem">
            Your F1 <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> beat the CPU best
            (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>). Full 30-run GPU statistics are now shown below.
          </div>
        </div>
        <div class="tlb-badge" style="background:#16a34a;color:#fff;font-size:0.69rem">✓ UNLOCKED</div>
      </div>
    </td>
  </tr>`
    : `<tr style="background:#0f172a">
    <td colspan="8" style="padding:8px 14px">
      <div class="tier-lock-banner" style="margin:0;padding:10px 16px">
        <div class="tlb-icon" style="font-size:1.3rem">🔒</div>
        <div class="tlb-body">
          <div class="tlb-title" style="font-size:0.87rem">GPU model stats hidden</div>
          <div class="tlb-sub" style="font-size:0.78rem">
            Beat CPU best F1 (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>) to reveal all 30-run GPU statistics.
            Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span>
          </div>
        </div>
        <div class="tlb-badge locked" style="font-size:0.69rem">⬆ BEAT CPU BEST</div>
      </div>
    </td>
  </tr>`;

  const rows = REPRO_MODELS.map((m, i) => {
    const locked = m.gpu && !gpuUnlocked;
    const dotColor = locked ? '#475569' : m.color;

    const banner = (i === REPRO_MODELS.findIndex(x => x.gpu)) ? summaryLockRow : '';

    if (locked) {
      return banner + `<tr style="opacity:.4;background:#0f172a">
        <td><span class="repro-dot" style="background:#475569"></span><span style="color:#64748b;font-style:italic">${m.label}</span></td>
        <td style="color:#334155">🔒</td><td style="color:#334155">🔒</td>
        <td style="color:#334155">🔒</td><td style="color:#334155">🔒</td>
        <td style="color:#334155">🔒</td><td style="color:#334155">🔒</td><td style="color:#334155">🔒</td>
      </tr>`;
    }

    const aucs = Object.values(ITER_DATA).map(it => it[m.key]?.auc).filter(v => v != null);
    const f1s  = Object.values(ITER_DATA).map(it => it[m.key]?.f1).filter(v => v != null);
    const aucMean = mean(aucs), aucStd = std(aucs);
    const f1Mean  = mean(f1s),  f1Std  = std(f1s);
    const f1Min   = Math.min(...f1s),  f1Max  = Math.max(...f1s);
    const f1Range = f1Max - f1Min;
    return banner + `<tr>
      <td><span class="repro-dot" style="background:${dotColor}"></span>${m.label}</td>
      <td style="font-weight:800;color:#c2410c">${f1Mean.toFixed(4)}</td>
      <td>${f1Std.toFixed(4)}</td>
      <td>${f1Min.toFixed(4)}</td>
      <td>${f1Max.toFixed(4)}</td>
      <td>${f1Range.toFixed(4)}</td>
      <td>${aucMean.toFixed(4)}</td>
      <td>${aucStd.toFixed(4)}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
}

let reproInited = false;
function initReproTab() {
  if (reproInited) return;
  reproInited = true;
  buildReproRangeChart();
  buildReproSummaryTable();
  updateIterationView(1);
  const slider = document.getElementById('repro-slider');
  if (slider) slider.addEventListener('input', e => updateIterationView(+e.target.value));
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(btn, tab) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('tab-' + tab);
  if (panel) panel.classList.add('active');
  btn.classList.add('active');
  try {
    if (tab === 'compare') renderCompare();
    if (tab === 'reproducibility') initReproTab();
  } catch(e) { console.error('Tab content render failed:', e); }
}

// ── Dataset stats ──────────────────────────────────────────────────────────────
function initStats() {
  const s = DATASET_STATS;
  const negP = +(100 - s.pos_rate).toFixed(1);
  document.getElementById('pos-bar').style.width = s.pos_rate + '%';
  document.getElementById('neg-bar').style.width = negP + '%';
  document.getElementById('bar-pos-pct').textContent = `${s.n_positive.toLocaleString()} positive (${s.pos_rate}%)`;
  document.getElementById('bar-neg-pct').textContent = `${s.n_negative.toLocaleString()} negative (${negP}%)`;
  document.getElementById('label-keywords').innerHTML =
    s.keywords.map(k => `<span class="feature-tag">${escHtml(k)}</span>`).join('');
}

function escHtml(v) {
  return String(v).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
}

// ── Raw data sub-tabs ──────────────────────────────────────────────────────────
const RAW_META = {
  patients:          { icon: '🧍', desc: 'One row per patient: demographics, birth and death dates, address, and lifetime healthcare costs.' },
  encounters:        { icon: '🏥', desc: 'One row per visit: when it happened, the type of visit, who provided care, who paid, and why.' },
  conditions:        { icon: '🩺', desc: 'Diagnoses with start and stop dates. The label is built from this file.' },
  medications:       { icon: '💊', desc: 'Prescriptions: drug, dates, cost, and the condition being treated.' },
  observations:      { icon: '📋', desc: 'Vital signs, lab results, and survey answers recorded during visits.' },
  procedures:        { icon: '🔬', desc: 'Procedures performed, with cost and the reason for each.' },
  immunizations:     { icon: '💉', desc: 'Vaccines given, with dates and cost.' },
  allergies:         { icon: '🤧', desc: 'Recorded allergies, with start and stop dates.' },
  careplans:         { icon: '📝', desc: 'Care plans, such as diabetes self-management, and the condition each one addresses.' },
  devices:           { icon: '🔋', desc: 'Medical devices used or implanted, such as defibrillators.' },
  imaging_studies:   { icon: '📷', desc: 'Imaging studies: body part, type of imaging, and date.' },
  supplies:          { icon: '📦', desc: 'Medical supplies used during visits, such as masks and gloves.' },
  providers:         { icon: '👤', desc: 'Clinicians who see patients, with specialty, location, and number of visits.' },
  organizations:     { icon: '🏢', desc: 'Hospitals and clinics where visits take place.' },
  payers:            { icon: '💳', desc: 'Insurers, with totals for what they covered and did not cover.' },
  payer_transitions: { icon: '🔄', desc: 'Which insurer covered each patient, year by year.' },
};

// What each column means. Columns shared by many files fall back to COLUMN_DOCS_COMMON.
const COLUMN_DOCS_COMMON = {
  PATIENT:   'Patient this record belongs to. Links to patients.csv.',
  ENCOUNTER: 'Visit where this was recorded. Links to encounters.csv.',
  PAYER:     'Insurer that paid. Links to payers.csv.',
  ADDRESS:   'Street address.',
  CITY:      'City.',
  STATE:     'State.',
  ZIP:       'ZIP code.',
  LAT:       'Latitude.',
  LON:       'Longitude.',
  PHONE:     'Phone number.',
};
const COLUMN_DOCS = {
  patients: {
    Id: 'Unique patient ID. Every other file links back to it through its PATIENT column.',
    BIRTHDATE: 'Date of birth.',
    DEATHDATE: 'Date of death. Blank if the patient is alive.',
    SSN: 'Synthetic Social Security number.',
    DRIVERS: 'Synthetic driver\'s license number. Blank if the patient has none, such as a child.',
    PASSPORT: 'Synthetic passport number. Blank if the patient has none.',
    PREFIX: 'Name prefix, such as Mr. or Mrs.',
    FIRST: 'First name. Synthea adds digits so names are clearly fake.',
    LAST: 'Last name, with digits added.',
    SUFFIX: 'Name suffix, such as MD or PhD.',
    MAIDEN: 'Maiden name, for patients who changed their name at marriage.',
    MARITAL: 'Marital status: M = married, S = single.',
    RACE: 'Race.',
    ETHNICITY: 'Ethnicity: hispanic or nonhispanic.',
    GENDER: 'Gender: F or M.',
    BIRTHPLACE: 'Town, state, and country of birth.',
    STATE: 'State of residence. Every patient in this export lives in Massachusetts.',
    COUNTY: 'County of residence.',
    LAT: 'Latitude of the home address.',
    LON: 'Longitude of the home address.',
    HEALTHCARE_EXPENSES: 'Lifetime healthcare costs paid by the patient.',
    HEALTHCARE_COVERAGE: 'Lifetime healthcare costs paid by insurers.',
  },
  encounters: {
    Id: 'Unique visit ID. Other clinical files link to it through their ENCOUNTER column.',
    START: 'Date and time the visit started.',
    STOP: 'Date and time the visit ended.',
    ORGANIZATION: 'Hospital or clinic where the visit took place. Links to organizations.csv.',
    PROVIDER: 'Clinician who saw the patient. Links to providers.csv.',
    PAYER: 'Insurer billed for the visit. Links to payers.csv.',
    ENCOUNTERCLASS: 'Type of visit: wellness, ambulatory, outpatient, inpatient, emergency, or urgent care.',
    CODE: 'SNOMED-CT code for the type of visit.',
    DESCRIPTION: 'Type of visit, in words.',
    BASE_ENCOUNTER_COST: 'Base cost of the visit, before medications and procedures.',
    TOTAL_CLAIM_COST: 'Total cost claimed for the visit.',
    PAYER_COVERAGE: 'Amount the insurer covered.',
    REASONCODE: 'SNOMED-CT code for the diagnosis that prompted the visit. Blank for routine visits.',
    REASONDESCRIPTION: 'Reason for the visit, in words. Can repeat diagnoses used to build the label.',
  },
  conditions: {
    START: 'Date the condition was diagnosed.',
    STOP: 'Date the condition resolved. Blank means it was still active at the end of the record.',
    ENCOUNTER: 'Visit where the condition was diagnosed. Links to encounters.csv.',
    CODE: 'SNOMED-CT diagnosis code.',
    DESCRIPTION: 'Diagnosis, in words. The label is built by searching this column for keywords.',
  },
  medications: {
    START: 'Date the prescription started.',
    STOP: 'Date the prescription ended. Blank if still active.',
    PAYER: 'Insurer that paid for the medication. Links to payers.csv.',
    ENCOUNTER: 'Visit where the medication was prescribed. Links to encounters.csv.',
    CODE: 'RxNorm drug code.',
    DESCRIPTION: 'Drug name, strength, and form. This is the medication text that TF-IDF and ClinicalBERT read.',
    BASE_COST: 'Cost of one dispense.',
    PAYER_COVERAGE: 'Amount the insurer covered.',
    DISPENSES: 'Number of times the prescription was filled.',
    TOTALCOST: 'Total cost across all dispenses.',
    REASONCODE: 'SNOMED-CT code for the condition being treated.',
    REASONDESCRIPTION: 'Condition being treated, in words. Often the same diagnosis that set the label.',
  },
  observations: {
    DATE: 'Date the measurement was taken.',
    ENCOUNTER: 'Visit where the measurement was recorded. Links to encounters.csv. Blank for some records.',
    CODE: 'LOINC code for what was measured.',
    DESCRIPTION: 'What was measured, such as body weight, blood pressure, or a lab test.',
    VALUE: 'The result. Mostly numbers, with some text answers such as "Never smoker".',
    UNITS: 'Units for VALUE, such as mg/dL or kg.',
    TYPE: 'Whether VALUE is numeric or text.',
  },
  procedures: {
    DATE: 'Date the procedure was performed.',
    CODE: 'SNOMED-CT procedure code.',
    DESCRIPTION: 'Procedure, in words.',
    BASE_COST: 'Cost of the procedure.',
    REASONCODE: 'SNOMED-CT code for the condition the procedure was for.',
    REASONDESCRIPTION: 'Condition the procedure was for, in words.',
  },
  immunizations: {
    DATE: 'Date the vaccine was given.',
    CODE: 'CVX vaccine code.',
    DESCRIPTION: 'Vaccine name.',
    BASE_COST: 'Cost of the vaccine.',
  },
  allergies: {
    START: 'Date the allergy was recorded.',
    STOP: 'Date the allergy resolved. Blank if still active.',
    CODE: 'SNOMED-CT allergy code.',
    DESCRIPTION: 'Allergy, in words.',
  },
  careplans: {
    Id: 'Unique care plan ID.',
    START: 'Date the care plan started.',
    STOP: 'Date the care plan ended. Blank if still active.',
    CODE: 'SNOMED-CT code for the care plan.',
    DESCRIPTION: 'Care plan, in words.',
    REASONCODE: 'SNOMED-CT code for the condition the plan addresses.',
    REASONDESCRIPTION: 'Condition the plan addresses, in words.',
  },
  devices: {
    START: 'Date the device was first used or implanted.',
    STOP: 'Date the device was removed. Blank if still in use.',
    CODE: 'SNOMED-CT device code.',
    DESCRIPTION: 'Device, in words.',
    UDI: 'Unique Device Identifier: a barcode-style ID for the specific device.',
  },
  imaging_studies: {
    Id: 'Unique imaging study ID.',
    DATE: 'Date of the study.',
    BODYSITE_CODE: 'SNOMED-CT code for the body part imaged.',
    BODYSITE_DESCRIPTION: 'Body part imaged, in words.',
    MODALITY_CODE: 'DICOM code for the type of imaging, such as DX for digital X-ray.',
    MODALITY_DESCRIPTION: 'Type of imaging, in words.',
    SOP_CODE: 'DICOM code for how the image is stored.',
    SOP_DESCRIPTION: 'How the image is stored, in words.',
  },
  supplies: {
    DATE: 'Date the supplies were used.',
    CODE: 'SNOMED-CT supply code.',
    DESCRIPTION: 'Supply, in words, such as a respirator mask or gloves.',
    QUANTITY: 'Number of units supplied.',
  },
  providers: {
    Id: 'Unique provider ID. Encounters link to it through their PROVIDER column.',
    ORGANIZATION: 'Organization the provider works for. Links to organizations.csv.',
    NAME: 'Provider name (synthetic).',
    GENDER: 'Gender: F or M.',
    SPECIALITY: 'Medical specialty, such as general practice.',
    UTILIZATION: 'Number of visits the provider handled.',
  },
  organizations: {
    Id: 'Unique organization ID. Encounters and providers link to it.',
    NAME: 'Name of the hospital or clinic.',
    REVENUE: 'Total revenue from visits.',
    UTILIZATION: 'Number of visits at this organization.',
  },
  payers: {
    Id: 'Unique insurer ID. Encounters, medications, and payer_transitions link to it.',
    NAME: 'Insurer name, such as Medicare, Medicaid, or a private plan.',
    STATE_HEADQUARTERED: 'State where the insurer is headquartered.',
    AMOUNT_COVERED: 'Total dollars the insurer paid.',
    AMOUNT_UNCOVERED: 'Total dollars the insurer did not pay.',
    REVENUE: 'Total premiums collected.',
    COVERED_ENCOUNTERS: 'Number of visits the insurer paid for.',
    UNCOVERED_ENCOUNTERS: 'Number of visits the insurer did not pay for.',
    COVERED_MEDICATIONS: 'Number of prescriptions the insurer paid for.',
    UNCOVERED_MEDICATIONS: 'Number of prescriptions the insurer did not pay for.',
    COVERED_PROCEDURES: 'Number of procedures the insurer paid for.',
    UNCOVERED_PROCEDURES: 'Number of procedures the insurer did not pay for.',
    COVERED_IMMUNIZATIONS: 'Number of vaccines the insurer paid for.',
    UNCOVERED_IMMUNIZATIONS: 'Number of vaccines the insurer did not pay for.',
    UNIQUE_CUSTOMERS: 'Number of different patients the insurer covered.',
    QOLS_AVG: 'Average quality-of-life score (0 to 1) of covered patients.',
    MEMBER_MONTHS: 'Total months of coverage across all members.',
  },
  payer_transitions: {
    START_YEAR: 'First year of this coverage period.',
    END_YEAR: 'Last year of this coverage period.',
    PAYER: 'Insurer during this period. Links to payers.csv.',
    OWNERSHIP: 'Who holds the policy: Self, Guardian, or Spouse.',
  },
};

const KIND_LABELS = { id: 'ID', date: 'Date', numeric: 'Number', category: 'Category', empty: 'Empty' };

function fmtNum(x) {
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function fmtPct(n, total) {
  if (!n || !total) return '0%';
  const p = n / total * 100;
  return p < 0.1 ? '<0.1%' : p.toFixed(1) + '%';
}

function profileSummary(p, key, nRows) {
  const filled = nRows - p.missing;
  const topValues = (p.top || []).map(([v, c]) => {
    const shown = v.length > 40 ? v.slice(0, 40) + '…' : v;
    return `${escHtml(shown)} (${fmtPct(c, filled)})`;
  }).join(', ');
  switch (p.kind) {
    case 'id':
      return p.ref === key ? 'Unique for every row' : `Links to ${p.ref}.csv`;
    case 'date':
      return `${p.min} to ${p.max}`;
    case 'numeric': {
      let text = `min ${fmtNum(p.min)} · median ${fmtNum(p.median)} · max ${fmtNum(p.max)}`;
      if (p.text_rows) text += `<br>Text answers: ${topValues}`;
      return text;
    }
    case 'category':
      return `Most common: ${topValues}`;
    default:
      return 'Always blank';
  }
}

// ── EDA charts (one series each: one color, no legend, the title names it) ────
function compactNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  return String(n);
}

// Vertical columns from one baseline. Each full-height slot is the hover target.
function columnChart(bars, { capFor = () => '', ticks = [] } = {}) {
  const max = Math.max(...bars.map(b => b.value), 1);
  const cols = bars.map(b => `
    <div class="eda-col" data-tip-value="${escHtml(b.tipValue)}" data-tip-label="${escHtml(b.label)}">
      <span class="eda-cap">${capFor(b)}</span>
      <div class="eda-bar" style="height:${b.value ? `max(1px, ${b.value / max * 100}%)` : 0}"></div>
    </div>`).join('');
  const tickLabels = ticks.map(t =>
    `<span style="left:${(t.index + 0.5) / bars.length * 100}%">${escHtml(t.label)}</span>`).join('');
  return `<div class="eda-plot">${cols}</div><div class="eda-xaxis">${tickLabels}</div>`;
}

function yearChart(byYear, total) {
  const counts = new Map(byYear.counts);
  const first = byYear.counts[0][0], last = byYear.counts[byYear.counts.length - 1][0];
  const bars = [];
  for (let y = first; y <= last; y++) {
    const n = counts.get(y) || 0;
    bars.push({ label: String(y), value: n, tipValue: `${n.toLocaleString()} ${byYear.column === 'BIRTHDATE' ? 'patients' : 'rows'} (${fmtPct(n, total)})` });
  }
  const peak = bars.reduce((a, b) => (b.value > a.value ? b : a));
  const ticks = bars.map((b, index) => ({ index, label: b.label }))
                    .filter(t => +t.label % 10 === 0 || bars.length < 10);
  const tableRows = bars.filter(b => b.value).map(b =>
    `<tr><td>${b.label}</td><td>${b.value.toLocaleString()}</td></tr>`).join('');
  const isBirth = byYear.column === 'BIRTHDATE';
  const noun = isBirth ? 'patients' : 'rows';
  return `
    <div class="eda-card wide">
      <div class="eda-card-title">${isBirth ? 'Patients by birth year' : `Rows per year, by ${escHtml(byYear.column)}`}</div>
      <div class="eda-card-sub">Peak: ${peak.label}, with ${peak.value.toLocaleString()} ${noun} (${fmtPct(peak.value, total)} of the file). Hover a bar for its count.</div>
      ${columnChart(bars, { capFor: b => (b === peak ? compactNum(b.value) : ''), ticks })}
      <details class="eda-table"><summary>View as table</summary>
        <div><table><thead><tr><th>Year</th><th>${isBirth ? 'Patients' : 'Rows'}</th></tr></thead><tbody>${tableRows}</tbody></table></div>
      </details>
    </div>`;
}

function perPatientChart(perPatient) {
  // drop empty bins past the last non-empty one
  let lastBin = perPatient.length - 1;
  while (lastBin > 0 && perPatient[lastBin][1] === 0) lastBin--;
  const nPatients = DATASET_STATS.n_total;
  const bars = perPatient.slice(0, lastBin + 1).map(([label, n]) => ({
    label: `${label} rows`, value: n, tipValue: `${n.toLocaleString()} patients (${fmtPct(n, nPatients)})`,
  }));
  const ticks = perPatient.slice(0, lastBin + 1).map(([label], index) => ({ index, label }));
  const none = perPatient[0][1];
  return `
    <div class="eda-card">
      <div class="eda-card-title">Rows per patient</div>
      <div class="eda-card-sub">How many rows each of the ${nPatients.toLocaleString()} patients has in this file. ${none.toLocaleString()} patients (${fmtPct(none, nPatients)}) have none.</div>
      ${columnChart(bars, { capFor: b => (b.value ? compactNum(b.value) : ''), ticks })}
    </div>`;
}

// Horizontal bars, longest first, value at the tip.
function barList(title, sub, items) {
  const max = Math.max(...items.map(i => i.value)) || 1;
  const rows = items.map(i => `
    <div class="eda-hrow" data-tip-value="${escHtml(i.tipValue)}" data-tip-label="${escHtml(i.label)}">
      <span class="eda-hlabel">${escHtml(i.label)}</span>
      <div class="eda-hplot">
        <div class="eda-hbar" style="width:calc((100% - 52px) * ${i.value / max})${i.value ? '' : ';min-width:0'}"></div>
        <span class="eda-hval">${escHtml(i.valueText)}</span>
      </div>
    </div>`).join('');
  return `
    <div class="eda-card">
      <div class="eda-card-title">${title}</div>
      <div class="eda-card-sub">${sub}</div>
      ${rows}
    </div>`;
}

function edaCharts(d) {
  const e = d.eda || {};
  const cards = [];
  if (e.by_year && e.by_year.counts.length) cards.push(yearChart(e.by_year, d.total));
  if (e.per_patient) cards.push(perPatientChart(e.per_patient));
  (e.top || []).filter(t => t.values.length).forEach(t => {
    const items = t.values.map(([v, n]) => ({
      label: v, value: n, valueText: fmtPct(n, t.total),
      tipValue: `${n.toLocaleString()} rows (${fmtPct(n, t.total)})`,
    }));
    const shown = t.values.length < 10 ? `All ${t.values.length}` : 'Top 10';
    cards.push(barList(`Most common ${escHtml(t.column)} values`,
      `${shown} values, as a share of the ${t.total.toLocaleString()} rows where ${escHtml(t.column)} is filled in.`, items));
  });
  if (e.value_by) {
    const items = e.value_by.values.map(([name, v]) => ({
      label: name, value: v, valueText: compactNum(v), tipValue: `${v.toLocaleString()} ${e.value_by.value}`,
    }));
    cards.push(barList(`${escHtml(e.value_by.value)} by insurer`,
      'Number of different patients each insurer covered.', items));
  }
  if (!cards.length) return '';
  const halfCards = cards.filter(c => !c.includes('eda-card wide'));
  if (halfCards.length === 1) cards[cards.indexOf(halfCards[0])] = halfCards[0].replace('class="eda-card"', 'class="eda-card wide"');
  return `
    <div class="eda-title">Exploratory data analysis</div>
    <div class="eda-grid">${cards.join('')}</div>`;
}

// One shared tooltip for every EDA mark. Values lead, labels follow.
const edaTip = document.createElement('div');
edaTip.className = 'eda-tip';
document.body.appendChild(edaTip);
document.addEventListener('pointermove', e => {
  const mark = e.target.closest('[data-tip-value]');
  if (!mark) { edaTip.style.display = 'none'; return; }
  const value = document.createElement('strong');
  value.textContent = mark.dataset.tipValue;
  const label = document.createElement('div');
  label.textContent = mark.dataset.tipLabel;
  edaTip.replaceChildren(value, label);
  edaTip.style.display = 'block';
  const flip = e.clientX > window.innerWidth - 260;
  edaTip.style.left = (flip ? e.clientX - edaTip.offsetWidth - 12 : e.clientX + 12) + 'px';
  edaTip.style.top = (e.clientY + 14) + 'px';
});

function buildRawTabs() {
  const nav    = document.getElementById('raw-tab-nav');
  const panels = document.getElementById('raw-panels');
  const keys   = Object.keys(RAW_DATA);

  nav.innerHTML = keys.map((key, i) => {
    const total = RAW_DATA[key].total.toLocaleString();
    return `<button class="raw-tab-btn${i === 0 ? ' active' : ''}" id="rtb-${key}" onclick="switchRawTab('${key}')">${RAW_META[key].icon} ${key}.csv <span class="rtb-count">${total} rows</span></button>`;
  }).join('');

  panels.innerHTML = keys.map((key, i) => {
    const d = RAW_DATA[key];
    const meta = RAW_META[key];
    const isPatientId = c => c === 'PATIENT' || (key === 'patients' && c === 'Id');

    const header = d.cols.map(c => `<th>${escHtml(c)}</th>`).join('');
    const body = d.rows.map((row, ri) => {
      const tds = d.cols.map(c => {
        const full = row[c] || '';
        if (isPatientId(c)) return `<td class="pid">${escHtml(full) || '—'}</td>`;
        // shorten other UUIDs (visit, provider, payer IDs) for readability
        const shown = /^[0-9a-f]{8}-[0-9a-f]{4}-/i.test(full) ? full.slice(0, 8) + '…' : full;
        return `<td title="${escHtml(full)}">${escHtml(shown) || '—'}</td>`;
      }).join('');
      return `<tr><td>${ri + 1}</td>${tds}</tr>`;
    }).join('');

    // File-level profile chips: patients covered and years spanned
    const patientCol = d.profile.find(p => p.kind === 'id' && p.ref === 'patients');
    const dateCol = d.profile.find(p => p.kind === 'date');
    const chips = [`${d.cols.length} columns`];
    if (patientCol) chips.push(`${patientCol.unique.toLocaleString()} of ${DATASET_STATS.n_total.toLocaleString()} patients`);
    if (dateCol) {
      const [y0, y1] = [dateCol.min.slice(0, 4), dateCol.max.slice(0, 4)];
      chips.push(y0 === y1 ? y0 : `${y0}–${y1}`);
    }

    const guideRows = d.profile.map((p, ci) => {
      const docs = (COLUMN_DOCS[key] || {})[p.name] || COLUMN_DOCS_COMMON[p.name] || '';
      return `<tr>
        <td>${ci + 1}</td>
        <td><span class="feat-name">${escHtml(p.name)}</span></td>
        <td>${escHtml(docs)}</td>
        <td><span class="kind-chip kind-${p.kind}">${KIND_LABELS[p.kind]}</span></td>
        <td>${fmtPct(p.missing, d.total)}</td>
        <td>${p.unique.toLocaleString()}</td>
        <td>${profileSummary(p, key, d.total)}</td>
      </tr>`;
    }).join('');

    return `
    <div class="raw-panel${i === 0 ? ' active' : ''}" id="rp-${key}">
      <div class="raw-file-meta">
        <span class="rfm-name">${meta.icon} ${key}.csv</span>
        ${chips.map(c => `<span class="raw-col-count">${c}</span>`).join('')}
        <span class="rfm-count">Showing first ${d.rows.length} of ${d.total.toLocaleString()} rows</span>
        <span class="rfm-desc">${meta.desc}</span>
      </div>
      <div class="raw-table-wrap">
        <table>
          <thead><tr><th>#</th>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
      <div class="eda-title">Column guide: what each column means and what it contains</div>
      <div class="feat-table-wrap">
        <table class="feat-table col-guide">
          <thead><tr><th style="width:32px">#</th><th>Column</th><th style="min-width:240px">What it means</th><th>Type</th><th>Missing</th><th>Distinct values</th><th style="min-width:240px">Summary</th></tr></thead>
          <tbody>${guideRows}</tbody>
        </table>
      </div>
      ${edaCharts(d)}
    </div>`;
  }).join('');
}

function switchRawTab(key) {
  document.querySelectorAll('.raw-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.raw-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('rtb-' + key).classList.add('active');
  document.getElementById('rp-'  + key).classList.add('active');
}

// ── Model gallery ─────────────────────────────────────────────────────────────
function modelCard(m, locked) {
  const catClass = {cpu:'mc-cat-cpu', gpu:'mc-cat-gpu', bert:'mc-cat-bert', my_model:'mc-cat-mymodel'}[m.category];
  const noResults = m.id === 'my_model' && !MY_MODEL_STATS.f1;   // template still at 0.0
  const f1Val = locked ? '🔒' : noResults ? '—' : (m.f1 != null ? m.f1.toFixed(3) : '—');
  const f1Color = locked ? '#94a3b8' : m.color;
  return `
  <div class="model-card${locked ? ' locked-card' : ''}" id="mc-${m.id}" onclick="${locked ? '' : "toggleModel('"+m.id+"')"}">
    <div class="mc-check" id="mc-check-${m.id}">✓</div>
    <div class="mc-icon">${m.icon}</div>
    <div><span class="mc-category ${catClass}">${m.categoryLabel}</span></div>
    <div class="mc-name">${m.name}</div>
    <div class="mc-auc" style="color:${f1Color}">${f1Val}</div>
    <div class="mc-auc-lbl">${locked ? 'Unlock to reveal' : noResults ? 'No results yet' : m.id === 'my_model' ? 'F1-Score (avg 5 runs)' : 'F1-Score (avg 30 runs)'}</div>
    <div class="mc-desc">${m.description}</div>
    <div class="mc-input">Input: ${m.input}</div>
  </div>`;
}

function renderModelsGrid() {
  const grid = document.getElementById('models-grid');
  const myModel = MODELS.find(m => m.id === 'my_model');
  const cpuModels = MODELS.filter(m => CPU_MODELS.includes(m.id));
  const gpuModels = MODELS.filter(m => GPU_MODELS.includes(m.id));

  // Section 1 — My Model vs CPU Baselines
  let html = `<div class="model-section-hdr">⚡ Tier 1 — Your Model vs CPU Baselines</div>`;
  html += `<div class="models-grid" style="margin-bottom:8px">`;
  html += modelCard(myModel, false);
  cpuModels.forEach(m => { html += modelCard(m, false); });
  html += `</div>`;

  // CPU unlock status bar
  const cpuDelta = (MY_MODEL_F1 - CPU_BEST_F1).toFixed(3);
  const cpuBeat = MY_MODEL_F1 > CPU_BEST_F1;
  html += `<div class="tier-lock-banner">
    <div class="tlb-icon">${cpuBeat ? '🔓' : '🔒'}</div>
    <div class="tlb-body">
      <div class="tlb-title">${cpuBeat ? 'GPU Tier Unlocked!' : 'GPU Tier Locked'}</div>
      <div class="tlb-sub">
        Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> &nbsp;·&nbsp;
        Need to beat: <span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span> (${CPU_BEST.name}, best CPU model)
        ${cpuBeat ? `&nbsp;·&nbsp; You lead by <span class="tlb-need">+${cpuDelta}</span>` : `&nbsp;·&nbsp; Gap: <span class="tlb-got">${cpuDelta}</span>`}
        <br/>Improve your model and re-run to unlock GPU models (if locked).
      </div>
    </div>
    <div class="tlb-badge ${cpuBeat ? '' : 'locked'}">${cpuBeat ? '✓ UNLOCKED' : '⬆ BEAT CPU BEST'}</div>
  </div>`;

  // Section 2 — GPU Models (locked or unlocked)
  html += `<div class="model-section-hdr" style="margin-top:4px">🧠 Tier 2 — GPU Sequence Models</div>`;
  html += `<div class="models-grid" style="margin-bottom:8px">`;
  gpuModels.forEach(m => { html += modelCard(m, !gpuUnlocked); });
  html += `</div>`;

  // Final goal: beat the GPU tier, then aim for the improvement goal
  const gpuBeat = MY_MODEL_F1 > GPU_BEST_F1;
  const gpuDelta = (MY_MODEL_F1 - GPU_BEST_F1).toFixed(3);
  const goodBeat = MY_MODEL_F1 > GOOD_F1;
  const goalNeed = (GOOD_F1 - MY_MODEL_F1).toFixed(3);
  const goalChip = goodBeat
    ? `<div class="goal-chip reached">🌟 Improvement goal reached: your F1 is above <span class="goal-num">${GOOD_F1.toFixed(2)}</span>, which counts as good performance for this problem.</div>`
    : `<div class="goal-chip">🎯 Improvement goal: an F1 above <span class="goal-num">${GOOD_F1.toFixed(2)}</span> counts as good performance for this problem. You need <span class="goal-num">+${goalNeed}</span> more.</div>`;
  const goalLine = goodBeat
    ? `Excellent work. ${goalChip}`
    : gpuBeat
      ? `Well done beating every baseline. ${goalChip}`
      : `Improve your model, re-run all 5 runs, and upload your updated my_model.csv on the Deliverables tab. ${goalChip}`;
  html += `<div class="tier-lock-banner">
    <div class="tlb-icon">${goodBeat ? '🌟' : gpuBeat ? '🏁' : '🎯'}</div>
    <div class="tlb-body">
      <div class="tlb-title">${goodBeat ? 'Task Complete, with Good Performance!' : gpuBeat ? 'Task Complete! You beat the GPU tier' : 'Final Goal: Beat the GPU Tier'}</div>
      <div class="tlb-sub">
        ${gpuUnlocked
          ? `Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> &nbsp;·&nbsp;
             Need to beat: <span class="tlb-need">${GPU_BEST_F1.toFixed(3)}</span> (${GPU_BEST.name}, best GPU model)
             ${gpuBeat ? `&nbsp;·&nbsp; You lead by <span class="tlb-need">+${gpuDelta}</span>` : `&nbsp;·&nbsp; Gap: <span class="tlb-got">${gpuDelta}</span>`}`
          : 'Unlock the GPU tier first by beating the best CPU model.'}
        <br/>${goalLine}
      </div>
    </div>
    <div class="tlb-badge ${gpuBeat ? '' : 'locked'}">${goodBeat ? '★ ABOVE 0.75' : gpuBeat ? '✓ COMPLETE' : '⬆ BEAT GPU BEST'}</div>
  </div>`;

  grid.innerHTML = html;
}

function toggleModel(id) {
  const m = MODELS.find(x => x.id === id);
  if (!m) return;
  // Block GPU models if tier is locked
  if (GPU_MODELS.includes(id) && !gpuUnlocked) return;

  if (state.selectedIds.has(id)) {
    state.selectedIds.delete(id);
    document.getElementById('mc-' + id).classList.remove('selected');
  } else {
    state.selectedIds.add(id);
    document.getElementById('mc-' + id).classList.add('selected');
  }
  updateCompareBar();
}

function updateCompareBar() {
  const bar = document.getElementById('compare-bar');
  const pills = document.getElementById('selected-pills');
  const badge = document.getElementById('cart-count');
  const n = state.selectedIds.size;

  if (n === 0) {
    bar.style.display = 'none';
    badge.style.display = 'none';
    return;
  }
  bar.style.display = 'flex';
  badge.style.display = 'inline-flex';
  badge.textContent = n;

  pills.innerHTML = [...state.selectedIds].map(id => {
    const m = MODELS.find(x => x.id === id);
    return `<span class="selected-pill" style="border-left:3px solid ${m.color}">
      ${m.icon} ${m.name}
      <span class="sp-remove" onclick="event.stopPropagation();toggleModel('${id}')">×</span>
    </span>`;
  }).join('');
}

// ── My Model results file: template download, upload, status ─────────────────
// Column names must match the template exactly; each row is one run. Every cell needs a
// real value: blank, null, and 0 are rejected, except gpu_hrs, which is 0 for CPU training.
const MY_MODEL_METRIC_COLS = { Accuracy: 'accuracy', MacroAUC: 'auc', Precision_1: 'precision', Recall_1: 'recall', F1_1: 'f1' };
const MY_MODEL_COST_COLS = ['elapsed_hr', 'gpu_hrs', 'cpu_pct'];
const MY_MODEL_COLUMNS = ['iteration', 'seed', ...Object.keys(MY_MODEL_METRIC_COLS), ...MY_MODEL_COST_COLS];
const MY_MODEL_ZERO_OK = new Set(['gpu_hrs']);
const MY_MODEL_RUNS = 5;
const MY_MODEL_TEMPLATE = MY_MODEL_COLUMNS.join(',') + '\n' +
  [1, 2, 3, 4, 5].map(i => `${i},null,0.0,0.0,0.0,0.0,0.0,null,null,null`).join('\n') + '\n';
const isMissingValue = v => ['', 'null', 'none', 'na', 'nan'].includes(String(v ?? '').trim().toLowerCase());

function downloadMyModelTemplate() {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([MY_MODEL_TEMPLATE], { type: 'text/csv' }));
  link.download = 'my_model.csv';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// Splits one CSV line, honoring double-quoted cells such as "13 baseline, 5 new features".
function splitCsvLine(line) {
  const cells = [];
  let cur = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { cells.push(cur); cur = ''; }
    else cur += ch;
  }
  cells.push(cur);
  return cells.map(c => c.trim());
}

// Checks every rule and returns all problems at once, so the file can be fixed in one pass.
function validateMyModelCsv(text) {
  const lines = text.replace(/^﻿/, '').split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return { errors: ['The file has no result rows.'] };
  const header = splitCsvLine(lines[0]);
  const missingCols = MY_MODEL_COLUMNS.filter(c => !header.includes(c));
  if (missingCols.length) {
    return { errors: [`Missing column(s): ${missingCols.join(', ')}. Column names must match the template exactly.`] };
  }
  const rows = lines.slice(1).map(l => {
    const cells = splitCsvLine(l);
    return Object.fromEntries(header.map((h, i) => [h, cells[i] ?? '']));
  });

  const errors = [];
  if (rows.length !== MY_MODEL_RUNS) errors.push(`Expected ${MY_MODEL_RUNS} runs (rows), found ${rows.length}.`);
  rows.forEach((r, idx) => {
    const where = `Row ${idx + 1}`;
    const num = c => Number(r[c]);
    const empty = MY_MODEL_COLUMNS.filter(c => isMissingValue(r[c]));
    if (empty.length) errors.push(`${where}: missing ${empty.join(', ')} (null and blank are not accepted).`);
    const filled = MY_MODEL_COLUMNS.filter(c => !isMissingValue(r[c]));
    const notNumbers = filled.filter(c => !Number.isFinite(num(c)));
    if (notNumbers.length) errors.push(`${where}: ${notNumbers.join(', ')} must be ${notNumbers.length > 1 ? 'numbers' : 'a number'}.`);
    const zeros = filled.filter(c => !MY_MODEL_ZERO_OK.has(c) && Number.isFinite(num(c)) && num(c) === 0);
    if (zeros.length) errors.push(`${where}: ${zeros.join(', ')} ${zeros.length > 1 ? 'are' : 'is'} 0, which is not a valid value. Enter your results.`);
    const ok = c => !isMissingValue(r[c]) && Number.isFinite(num(c)) && (num(c) !== 0 || MY_MODEL_ZERO_OK.has(c));
    if (ok('iteration') && !(Number.isInteger(num('iteration')) && num('iteration') >= 1 && num('iteration') <= MY_MODEL_RUNS)) {
      errors.push(`${where}: iteration must be a whole number from 1 to ${MY_MODEL_RUNS}.`);
    }
    if (ok('seed') && !Number.isInteger(num('seed'))) errors.push(`${where}: seed must be a whole number.`);
    Object.keys(MY_MODEL_METRIC_COLS).forEach(c => {
      if (ok(c) && (num(c) < 0 || num(c) > 1)) errors.push(`${where}: ${c} must be between 0 and 1 (for example 0.72, not 72).`);
    });
    if (ok('elapsed_hr') && num('elapsed_hr') < 0) errors.push(`${where}: elapsed_hr must be greater than 0 (training time in hours).`);
    if (ok('gpu_hrs') && num('gpu_hrs') < 0) errors.push(`${where}: gpu_hrs must be 0 or more (use 0 if you trained on a CPU).`);
    if (ok('cpu_pct') && (num('cpu_pct') < 0 || num('cpu_pct') > 100)) errors.push(`${where}: cpu_pct must be between 0 and 100.`);
  });
  const seeds = rows.map(r => r.seed).filter(s => !isMissingValue(s));
  if (seeds.length === rows.length && new Set(seeds).size !== seeds.length) errors.push('Each run must use a different seed.');
  const iters = rows.map(r => r.iteration).filter(s => !isMissingValue(s));
  if (iters.length === rows.length && new Set(iters).size !== iters.length) errors.push('Each iteration number must appear only once.');
  return { errors, rows };
}

// Same averaging as load_my_model_stats() in Python, applied only to a file that passed validation.
function parseMyModelCsv(text) {
  const { errors, rows } = validateMyModelCsv(text);
  if (errors.length) {
    const err = new Error(errors.join(' '));
    err.problems = errors;
    throw err;
  }
  const mean = c => rows.reduce((s, r) => s + Number(r[c]), 0) / rows.length;
  const stats = { n_iterations: rows.length };
  Object.entries(MY_MODEL_METRIC_COLS).forEach(([c, k]) => { stats[k] = Math.round(mean(c) * 1e4) / 1e4; });
  MY_MODEL_COST_COLS.forEach(c => { stats[c] = mean(c); });
  return stats;
}

function showUploadErrors(fileName, problems) {
  const shown = problems.slice(0, 8);
  document.querySelectorAll('.mm-upload-status').forEach(el => {
    el.className = 'mm-upload-status mm-error';
    const title = document.createElement('div');
    title.className = 'mm-error-title';
    title.textContent = `${fileName} was not loaded. Fix these values and upload the file again:`;
    const list = document.createElement('ul');
    shown.forEach(p => { const li = document.createElement('li'); li.textContent = p; list.appendChild(li); });
    if (problems.length > shown.length) {
      const li = document.createElement('li');
      li.textContent = `…and ${problems.length - shown.length} more.`;
      list.appendChild(li);
    }
    el.replaceChildren(title, list);
  });
}

function handleMyModelUpload(input) {
  const file = input.files && input.files[0];
  input.value = '';
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const stats = parseMyModelCsv(String(reader.result));
      localStorage.setItem(MY_MODEL_UPLOAD_KEY, JSON.stringify({ stats, fileName: file.name, loadedAt: new Date().toISOString() }));
      location.hash = 'models';
      location.reload();
    } catch (e) {
      const blocked = e.name === 'SecurityError' || e.name === 'QuotaExceededError';
      showUploadErrors(file.name, e.problems || [blocked ? 'Your browser blocked local storage, so the upload could not be saved.' : e.message]);
    }
  };
  reader.readAsText(file);
}

function clearMyModelUpload() {
  try { localStorage.removeItem(MY_MODEL_UPLOAD_KEY); } catch (e) { /* storage unavailable */ }
  location.hash = 'models';
  location.reload();
}

function renderUploadStatus() {
  document.querySelectorAll('.mm-upload-status').forEach(el => {
    el.className = 'mm-upload-status';
    if (MY_MODEL_UPLOAD && MY_MODEL_UPLOAD.stats) {
      const s = MY_MODEL_UPLOAD.stats;
      const when = new Date(MY_MODEL_UPLOAD.loadedAt).toLocaleDateString();
      el.textContent = `Showing ${MY_MODEL_UPLOAD.fileName} (${s.n_iterations} runs, F1 ${s.f1.toFixed(3)}, loaded ${when}). Saved in this browser only. `;
      const clear = document.createElement('button');
      clear.className = 'mm-link';
      clear.textContent = 'Clear upload';
      clear.onclick = clearMyModelUpload;
      el.appendChild(clear);
    } else {
      el.textContent = 'No file uploaded yet. My Model shows the built-in results file.';
    }
  });
}

function goToDeliverables() {
  document.querySelector('[data-tab="challenges"]').click();
  const target = document.getElementById('results-file');
  if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function goToModels() {
  document.querySelector('[data-tab="models"]').click();
}

function goToCompare() {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-compare').classList.add('active');
  const compareBtn = document.querySelector('[data-tab="compare"]');
  if (compareBtn) compareBtn.classList.add('active');
  window.scrollTo(0, 0);
  renderCompare();
}

// ── Comparison rendering ──────────────────────────────────────────────────────
const METRIC_LABELS = {
  auc: 'AUC-ROC',
  accuracy: 'Accuracy',
  f1: 'F1-Score',
  precision: 'Precision',
  recall: 'Recall',
};

function setMetric(btn, m) {
  state.activeMetric = m;
  document.querySelectorAll('.metric-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderChart();
}

function renderCompare() {
  const selected = MODELS.filter(m => state.selectedIds.has(m.id));
  if (selected.length < 2) {
    document.getElementById('compare-empty').style.display = 'block';
    document.getElementById('compare-content').style.display = 'none';
    return;
  }
  document.getElementById('compare-empty').style.display = 'none';
  document.getElementById('compare-content').style.display = 'block';

  renderChart();
  renderPerfTable(selected);
  renderComputeTable(selected);
  renderProsCons(selected);
}

function renderChart() {
  const selected = MODELS.filter(m => state.selectedIds.has(m.id));
  const metric = state.activeMetric;
  const wrap = document.getElementById('chart-svg-wrap');

  const W = 560, H = 180, PAD_L = 50, PAD_R = 20, PAD_T = 20, PAD_B = 40;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;
  const n = selected.length;
  const barW = Math.min(60, (chartW / n) * 0.6);
  const gap = chartW / n;

  const vals = selected.map(m => m[metric]);
  const hasData = vals.some(v => v !== null);
  if (!hasData) {
    wrap.innerHTML = '<div style="text-align:center;padding:40px;color:#86674a;font-size:0.86rem">No data logged for this metric for selected models.</div>';
    return;
  }

  const maxVal = Math.max(...vals.filter(v => v !== null), 0.95);
  const minVal = Math.min(...vals.filter(v => v !== null), 0.6);
  const scale = v => chartH - (v - minVal) / (maxVal - minVal + 0.05) * chartH;

  let svgBars = selected.map((m, i) => {
    const v = m[metric];
    const cx = PAD_L + gap * i + gap / 2;
    if (v === null) {
      return `<text x="${cx}" y="${PAD_T + chartH / 2}" text-anchor="middle" font-size="11" fill="#86674a">—</text>`;
    }
    const y = PAD_T + scale(v);
    const barH = (PAD_T + chartH) - y;
    return `
      <rect x="${cx - barW / 2}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${m.color}" opacity=".85"/>
      <text x="${cx}" y="${y - 5}" text-anchor="middle" font-size="11" fill="#2e1f10" font-weight="700">${v.toFixed(3)}</text>
    `;
  }).join('');

  let svgLabels = selected.map((m, i) => {
    const cx = PAD_L + gap * i + gap / 2;
    return `<text x="${cx}" y="${H - 8}" text-anchor="middle" font-size="11" fill="${m.color}" font-weight="700">${m.name}</text>`;
  }).join('');

  // Y-axis ticks
  const ticks = [minVal, (minVal + maxVal) / 2, maxVal].map(v => {
    const y = PAD_T + scale(v);
    return `<text x="${PAD_L - 5}" y="${y + 3}" text-anchor="end" font-size="9" fill="#86674a">${v.toFixed(2)}</text>
            <line x1="${PAD_L}" y1="${y}" x2="${W - PAD_R}" y2="${y}" stroke="#f0e8d8" stroke-width="1"/>`;
  }).join('');

  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
    ${ticks}
    <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${PAD_T + chartH}" stroke="#e8ddd0" stroke-width="1.5"/>
    <line x1="${PAD_L}" y1="${PAD_T + chartH}" x2="${W - PAD_R}" y2="${PAD_T + chartH}" stroke="#e8ddd0" stroke-width="1.5"/>
    ${svgBars}
    ${svgLabels}
    <text x="${W / 2}" y="12" text-anchor="middle" font-size="11" fill="#7a5c3c" font-weight="700">${METRIC_LABELS[metric]}</text>
  </svg>`;
}

function renderPerfTable(selected) {
  const thead = document.getElementById('perf-thead-row');
  const tbody = document.getElementById('perf-tbody');
  const cols = ['f1', 'auc', 'accuracy', 'precision', 'recall'];
  const colLabels = ['F1-Score ★', 'AUC-ROC', 'Accuracy', 'Precision', 'Recall'];

  thead.innerHTML = '<th>Metric</th>' + selected.map(m =>
    `<th style="color:${m.color}">${m.icon} ${m.name}</th>`).join('');

  tbody.innerHTML = cols.map((c, i) => {
    const isF1 = c === 'f1';
    const cells = selected.map(m => {
      const v = m[c];
      if (v === null) return '<td><span class="na-chip">—</span></td>';
      const pct = Math.round(v * 100);
      return `<td${isF1 ? ' style="background:#fef9f0"' : ''}><strong${isF1 ? ' style="color:#c2410c"' : ''}>${v.toFixed(3)}</strong> <span style="color:#86674a;font-size:0.75rem">(${pct}%)</span></td>`;
    }).join('');
    const rowStyle = isF1 ? ' style="background:#fef9f0"' : '';
    const labelStyle = isF1 ? ' style="font-weight:900;color:#c2410c"' : 'style="font-weight:700;color:#3d2a18"';
    return `<tr${rowStyle}><td ${labelStyle}>${colLabels[i]}${isF1 ? ' ★' : ''}</td>${cells}</tr>`;
  }).join('');
}

// ── Cloud cost estimate ───────────────────────────────────────────────────────
// Jetstream2 (where the benchmark ran) has no dollar pricing: it charges ACCESS credits.
// So we price each run at Google Cloud on-demand list prices, us-central1, checked Oct 12, 2026:
//   GPU runs: a2-highgpu-1g, 1x NVIDIA A100 40 GB (same GPU as the benchmark)  $3.6734/hr
//   CPU runs: n2-standard-8, 8 vCPUs, 32 GB                                   $0.3885/hr
const CLOUD_PRICE = {
  gpu: { machine: 'a2-highgpu-1g', perHour: 3.6734 },
  cpu: { machine: 'n2-standard-8', perHour: 0.3885 },
};

function estimateRunCost(m) {
  if (m.gpuHrs > 0) return { usd: m.gpuHrs * CLOUD_PRICE.gpu.perHour, machine: CLOUD_PRICE.gpu.machine };
  if (m.trainTimeSec != null) return { usd: m.trainTimeSec / 3600 * CLOUD_PRICE.cpu.perHour, machine: CLOUD_PRICE.cpu.machine };
  return null;
}

function fmtUsd(v) {
  return '$' + (v < 0.01 ? v.toFixed(4) : v < 1 ? v.toFixed(3) : v.toFixed(2));
}

function renderComputeTable(selected) {
  const thead = document.getElementById('comp-thead-row');
  const tbody = document.getElementById('comp-tbody');

  thead.innerHTML = '<th>Attribute</th>' + selected.map(m =>
    `<th style="color:${m.color}">${m.icon} ${m.name}</th>`).join('');

  const rows = [
    {label: 'Requires GPU', fn: m => m.gpuHrs == null ? '<span class="na-chip">—</span>' : m.gpuRequired ? '<span class="yes-chip">✓ Yes</span>' : '<span class="no-chip">✗ No</span>'},
    {label: 'Training Time / Iter', fn: m => m.trainTimeSec == null ? '<span class="na-chip">—</span>' : m.trainTimeSec < 60 ? m.trainTimeSec + ' sec' : Math.round(m.trainTimeSec / 60) + ' min ' + (m.trainTimeSec % 60) + ' sec'},
    {label: 'GPU Hours / Iter', fn: m => m.gpuHrs == null ? '<span class="na-chip">—</span>' : m.gpuHrs === 0 ? '0 (CPU only)' : m.gpuHrs.toFixed(3) + ' hr'},
    {label: 'CPU Utilization', fn: m => m.cpuPct == null ? '<span class="na-chip">—</span>' : m.cpuPct.toFixed(1) + '%'},
    {label: 'Parameters', fn: m => m.params},
    {label: 'Interpretable?', fn: m => m.interpretable == null ? '<span class="na-chip">—</span>' : m.interpretable ? '<span class="yes-chip">✓ Yes</span>' : '<span class="no-chip">✗ No</span>'},
    {label: 'Cost in Dollars / Run', fn: m => {
      const c = estimateRunCost(m);
      return c ? `<strong>${fmtUsd(c.usd)}</strong><br/><span style="color:#86674a;font-size:0.76rem">on ${c.machine}</span>` : '<span class="na-chip">—</span>';
    }},
    {label: 'Cost in Dollars / 1,000 Runs', fn: m => {
      const c = estimateRunCost(m);
      return c ? fmtUsd(c.usd * 1000) : '<span class="na-chip">—</span>';
    }},
    {label: 'Input Type', fn: m => m.input},
  ];

  tbody.innerHTML = rows.map(r =>
    `<tr><td style="font-weight:700;color:#3d2a18">${r.label}</td>${selected.map(m => '<td>' + r.fn(m) + '</td>').join('')}</tr>`
  ).join('');
}

function renderProsCons(selected) {
  const grid = document.getElementById('pros-cons-grid');
  const colW = Math.max(1, Math.floor(12 / selected.length));
  grid.style.gridTemplateColumns = selected.map(() => '1fr').join(' ');

  grid.innerHTML = selected.map(m => {
    const bodyContent = m.id === 'my_model'
      ? `<div style="color:#86674a;font-size:0.87rem;font-style:italic;padding:16px 0">
           Strengths and limitations depend on how you implement your model.<br/>
           Modify <code>final_project_starter.py</code> and re-run to improve your score.
         </div>`
      : `<div class="pc-section-label">Strengths</div>
         ${m.pros.map(p => `<div class="pro-item"><span class="pro-dot">+</span>${p}</div>`).join('')}
         <div class="pc-section-label" style="margin-top:10px">Limitations</div>
         ${m.cons.map(c => `<div class="con-item"><span class="con-dot">−</span>${c}</div>`).join('')}`;
    return `
    <div>
      <div class="pc-model-title" style="background:${m.color}18;border-left:4px solid ${m.color}">
        ${m.icon} ${m.name}
      </div>
      <div style="padding:12px 14px;background:#fff;border:1px solid #e8ddd0;border-top:none;border-radius:0 0 8px 8px">
        ${bodyContent}
      </div>
    </div>`;
  }).join('');
}

function renderSummary(selected) {
  const div = document.getElementById('summary-text');
  const ids = new Set(selected.map(m => m.id));
  const hasBert = ids.has('clinicalbert');
  const hasGpu = selected.some(m => m.gpuRequired);
  const hasCpu = selected.some(m => !m.gpuRequired);
  const bestAUC = selected.reduce((a, b) => a.auc >= b.auc ? a : b);
  const bestF1  = selected.filter(m => m.f1 !== null).reduce((a, b) => (a && a.f1 >= b.f1) ? a : b, null);
  const fastest = selected.reduce((a, b) => a.trainTimeSec <= b.trainTimeSec ? a : b);

  let text = `<strong>${bestAUC.name}</strong> achieves the highest AUC (${bestAUC.auc.toFixed(3)}) among the selected models.`;
  if (bestF1 && bestF1.id !== bestAUC.id) {
    text += ` <strong>${bestF1.name}</strong> leads on F1-score (${bestF1.f1.toFixed(3)}), meaning it balances precision and recall best.`;
  }
  text += ` <strong>${fastest.name}</strong> is the fastest to train at ${fastest.trainTimeSec} seconds per iteration.`;
  if (hasGpu && hasCpu) {
    text += ` The GPU models achieve higher or equal AUC compared to CPU baselines, but require specialized hardware and more infrastructure.`;
  }
  if (hasBert) {
    text += ` ClinicalBERT delivers the best F1 and precision thanks to its pre-trained clinical language knowledge &mdash; ideal when you need high confidence predictions, and have a high-memory GPU and a one-time embedding precomputation budget.`;
  }
  text += ` <strong>For quick deployment on any hardware</strong>, Random Forest is hard to beat &mdash; it comes within 0.002 AUC of the deep models at a fraction of the cost.`;

  div.innerHTML = text;
}

// ── Feature table data ────────────────────────────────────────────────────────
// use: "all" = RF/XGB + sequence models (LSTM, GRU, Transformer) + ClinicalBERT, "text" = sequence models + ClinicalBERT
const FEATURES = [
  {group:"Demographics (patients.csv)"},
  {n:"race",      desc:"Race, as a number",                      src:"patients", cols:"RACE",      use:"all", calc:"Label-encoded in alphabetical order: asian 0, black 1, native 2, other 3, white 4"},
  {n:"ethnicity", desc:"Ethnicity, as a number",                 src:"patients", cols:"ETHNICITY", use:"all", calc:"Label-encoded: hispanic 0, nonhispanic 1"},
  {n:"age",       desc:"Age in whole years",                     src:"patients", cols:"BIRTHDATE", use:"all", calc:"(date the pipeline ran − BIRTHDATE) in days ÷ 365, rounded down. DEATHDATE is not used, so deceased patients keep aging"},
  {n:"is_female", desc:"Gender indicator",                       src:"patients", cols:"GENDER",    use:"all", calc:"1 if GENDER is F, otherwise 0"},
  {group:"Healthcare costs (patients.csv)"},
  {n:"healthcare_expenses", desc:"Lifetime healthcare costs paid by the patient ($)", src:"patients", cols:"HEALTHCARE_EXPENSES", use:"all", calc:"Copied directly"},
  {n:"healthcare_coverage", desc:"Lifetime healthcare costs paid by insurers ($)",    src:"patients", cols:"HEALTHCARE_COVERAGE", use:"all", calc:"Copied directly"},
  {group:"Vital signs (observations.csv)"},
  {n:"body_height",              desc:"Height (cm)",                      src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Body Height' reading in the patient's life, so a child's height is averaged across growth. Missing values filled with the median"},
  {n:"body_weight",              desc:"Weight (kg)",                      src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Body Weight' reading. Missing values filled with the median"},
  {n:"body_mass_index",          desc:"Body mass index (kg/m²)",          src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Body Mass Index' reading. Missing values filled with the median"},
  {n:"systolic_blood_pressure",  desc:"Systolic blood pressure (mm Hg)",  src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Systolic Blood Pressure' reading. Missing values filled with the median"},
  {n:"diastolic_blood_pressure", desc:"Diastolic blood pressure (mm Hg)", src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Diastolic Blood Pressure' reading. Missing values filled with the median"},
  {n:"heart_rate",               desc:"Heart rate (beats per minute)",    src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Heart rate' reading. Missing values filled with the median"},
  {n:"respiratory_rate",         desc:"Breathing rate (breaths per minute)", src:"observations", cols:"DESCRIPTION, VALUE", use:"all", calc:"Average of every 'Respiratory rate' reading. Missing values filled with the median"},
  {group:"Drug-name TF-IDF weights (medications.csv)"},
  {n:"tfidf_acetaminophen", desc:"How strongly 'acetaminophen' stands out in the medication note. Non-zero for 24% of patients", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"TF-IDF with a fixed 12-word vocabulary on the patient's joined medication names; each patient's 12 weights are scaled to length 1"},
  {n:"tfidf_naproxen",      desc:"Same, for 'naproxen'. Non-zero for 10% of patients",     src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_ibuprofen",     desc:"Same, for 'ibuprofen'. Non-zero for 2.7% of patients",   src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_oxycodone",     desc:"Same, for 'oxycodone'. Non-zero for 1.5% of patients",   src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_hydrocodone",   desc:"Same, for 'hydrocodone'. Non-zero for 1.4% of patients", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_meperidine",    desc:"Same, for 'meperidine'. Non-zero for 0.9% of patients",  src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_pregabalin",    desc:"Same, for 'pregabalin'. Non-zero for 8 patients",        src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_morphine",      desc:"Same, for 'morphine'. Non-zero for 3 patients",          src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_fentanyl",      desc:"Same, for 'fentanyl'. Non-zero for 1 patient",           src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_opioid",        desc:"Same, for 'opioid'. Always 0: the word never appears",   src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_gabapentin",    desc:"Same, for 'gabapentin'. Always 0: the word never appears", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {n:"tfidf_tramadol",      desc:"Same, for 'tramadol'. Always 0: the word never appears", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same TF-IDF as above"},
  {group:"Medication note summaries (medications.csv)"},
  {n:"topic_1", desc:"Share of the medication note that belongs to medication theme 1", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"LDA with 5 topics on word counts (words in at least 10 notes, English stop words removed). Topics are found automatically and are not named; the 5 shares add up to 1"},
  {n:"topic_2", desc:"Same, for medication theme 2", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same LDA model as above"},
  {n:"topic_3", desc:"Same, for medication theme 3", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same LDA model as above"},
  {n:"topic_4", desc:"Same, for medication theme 4", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same LDA model as above"},
  {n:"topic_5", desc:"Same, for medication theme 5", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"Same LDA model as above"},
  {n:"sentiment",    desc:"Positive or negative tone of the medication note, from −1 to 1", src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"TextBlob polarity. Drug names carry almost no tone, so it is 0 for 96% of patients with medications"},
  {n:"note_cluster", desc:"Which of 5 medication-pattern groups the patient falls in",       src:"medications", cols:"PATIENT, DESCRIPTION", use:"text", calc:"K-means with 5 clusters on the standardized TF-IDF weights and topics. Fed to models as a number; one cluster holds a single patient"},
];

const SRC_CLASS = {
  patients:"src-patients", conditions:"src-conditions", encounters:"src-encounters",
  medications:"src-medications", procedures:"src-procedures", observations:"src-observations"
};

const FEATURE_USE_TAGS = {
  all:  '<span class="model-use-tag mtu-cpu">RF / XGB</span> <span class="model-use-tag mtu-gpu">Sequence models</span> <span class="model-use-tag mtu-bert">ClinicalBERT</span>',
  text: '<span class="model-use-tag mtu-gpu">Sequence models</span> <span class="model-use-tag mtu-bert">ClinicalBERT</span>',
};

// Random Forest permutation importance on the 13 baseline features: mean and std of the
// test-AUC drop over the 30 seeded splits (200 trees, balanced class weights, 5 repeats).
const FEATURE_IMPORTANCE = [
  ["age", 0.0626, 0.0077], ["healthcare_coverage", 0.0445, 0.0061],
  ["healthcare_expenses", 0.0099, 0.0034], ["body_weight", 0.0093, 0.0036],
  ["body_height", 0.0044, 0.0026], ["heart_rate", 0.0032, 0.0022],
  ["respiratory_rate", 0.0011, 0.0020], ["systolic_blood_pressure", 0.0007, 0.0017],
  ["diastolic_blood_pressure", 0.0001, 0.0014], ["ethnicity", 0.0000, 0.0007],
  ["body_mass_index", -0.0001, 0.0023], ["race", -0.0005, 0.0007], ["is_female", -0.0005, 0.0011],
];

function buildFeatTable() {
  const tbody = document.getElementById('feat-tbody');
  if (!tbody) return;
  let rowNum = 0;
  tbody.innerHTML = FEATURES.map(f => {
    if (f.group) {
      return `<tr class="feat-group-row"><td colspan="7">${f.group}</td></tr>`;
    }
    rowNum++;
    const srcLabel = f.src + '.csv';
    const colPills = f.cols.split(',').map(c =>
      `<code style="font-size:0.72rem;background:#f0e8d8;padding:1px 5px;border-radius:3px;white-space:nowrap;color:#5c3d22">${c.trim()}</code>`
    ).join(' ');
    return `<tr>
      <td>${rowNum}</td>
      <td><span class="feat-name">${f.n}</span></td>
      <td style="font-size:0.81rem;color:#4b3320">${f.desc}</td>
      <td><span class="src-badge ${SRC_CLASS[f.src]}">${srcLabel}</span></td>
      <td style="line-height:1.8">${colPills}</td>
      <td style="font-size:0.78rem;color:#6b5440">${f.calc}</td>
      <td style="line-height:1.9">${FEATURE_USE_TAGS[f.use]}</td>
    </tr>`;
  }).join('');

  const importance = document.getElementById('feat-importance');
  if (importance) {
    const items = FEATURE_IMPORTANCE.map(([name, mean, std]) => ({
      label: name,
      value: Math.max(0, mean),
      valueText: Math.max(0, mean).toFixed(3),
      tipValue: `AUC drop ${mean.toFixed(4)} ± ${std.toFixed(4)}`,
    }));
    importance.innerHTML = barList('Random Forest feature importance, 13 baseline features',
      'Mean drop in test AUC when the feature is shuffled. Hover a bar for the spread across the 30 runs.', items);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
[initStats, buildRawTabs, buildFeatTable, renderModelsGrid, renderUploadStatus].forEach(fn => {
  try { fn(); } catch(e) { console.error(fn.name + ' failed:', e); }
});

// ── Tab nav event listener (data-tab approach, no inline onclick needed) ──────
document.getElementById('tab-nav').addEventListener('click', function(e) {
  const btn = e.target.closest('[data-tab]');
  if (!btn) return;
  const tab = btn.dataset.tab;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('tab-' + tab);
  if (panel) panel.classList.add('active');
  btn.classList.add('active');
  window.scrollTo(0, 0);
  try {
    if (tab === 'compare') renderCompare();
    if (tab === 'reproducibility') initReproTab();
  } catch(e) { console.error('Tab render failed:', e); }
});

// Open the tab named in the URL hash, e.g. #models after uploading results
const startTabBtn = location.hash && document.querySelector(`#tab-nav [data-tab="${location.hash.slice(1)}"]`);
if (startTabBtn) startTabBtn.click();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Final Project Explorer  —  http://localhost:3002")
    log.info("=" * 50)
    app.run(port=3002, debug=False, use_reloader=False)
