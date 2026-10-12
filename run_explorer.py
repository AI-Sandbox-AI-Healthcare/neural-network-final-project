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

n_total = len(all_rows)
n_pos   = sum(1 for r in all_rows if r.get("binary_label") == "1")
n_neg   = n_total - n_pos
ages    = [float(r["age"]) for r in all_rows if r.get("age")]
DATASET_STATS = {
    "n_total":   n_total,
    "n_positive": n_pos,
    "n_negative": n_neg,
    "pos_rate":  round(n_pos / n_total * 100, 1),
    "age_min":   int(min(ages)),
    "age_max":   int(max(ages)),
    "n_features": len(all_rows[0]) - 2 if all_rows else 0,   # minus id, binary_label
}
log.info("Loaded %d patients (showing first %d)", n_total, len(PATIENTS_SAMPLE))

# ---------------------------------------------------------------------------
# Load top 20 rows from each raw Synthea CSV
# ---------------------------------------------------------------------------
RAW_FILES = [
    ("patients",     "patients.csv"),
    ("conditions",   "conditions.csv"),
    ("encounters",   "encounters.csv"),
    ("medications",  "medications.csv"),
    ("procedures",   "procedures.csv"),
    ("observations", "observations.csv"),
]

RAW_DATA = {}   # {key: {"cols": [...], "rows": [...], "total": N}}
RAW_COUNTS = {}

for key, fname in RAW_FILES:
    fpath = RAW_DIR / fname
    try:
        with open(fpath, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            total = 0
            for row in reader:
                total += 1
                if len(rows) < 20:
                    rows.append(dict(row))
        RAW_DATA[key] = {"cols": list(rows[0].keys()), "rows": rows, "total": total}
        log.info("Raw %-14s  %6d rows  %d cols", fname, total, len(rows[0]))
    except Exception as exc:
        log.error("Could not load %s: %s", fname, exc)
        RAW_DATA[key] = {"cols": [], "rows": [], "total": 0}

# ---------------------------------------------------------------------------
# My Model — loaded fresh on every request so edits to the CSV are reflected
# on page refresh without restarting the server.
# ---------------------------------------------------------------------------
_MY_MODEL_METRIC_MAP = {
    "MacroAUC":   "auc",
    "Accuracy":   "accuracy",
    "F1_1":       "f1",
    "Precision_1":"precision",
    "Recall_1":   "recall",
}
_MY_MODEL_CSV = ROOT / "my_model_over_30_iterations.csv"

def load_my_model_stats():
    result = {"auc": 0.0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    try:
        with open(_MY_MODEL_CSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = _MY_MODEL_METRIC_MAP.get(row.get("Metric", ""))
                if key:
                    result[key] = round(float(row["Value"]), 4)
        log.info("My Model  F1=%.4f  AUC=%.4f  (from %s)", result["f1"], result["auc"], _MY_MODEL_CSV.name)
    except Exception as exc:
        log.warning("Could not load my_model CSV (%s): %s — using zeros", _MY_MODEL_CSV, exc)
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
#app-header .logo{font-size:1.1rem;font-weight:800;letter-spacing:.5px}
#app-header .subtitle{font-size:.75rem;color:#c4a882;font-weight:500}
#app-header .badge{background:#7c4f2f;color:#f5f0e8;font-size:.65rem;padding:2px 8px;border-radius:20px;margin-left:auto;font-weight:700;letter-spacing:.5px}

/* ── Tab nav ── */
#tab-nav{background:#3d2a18;display:flex;gap:0;border-bottom:2px solid #5c3d22}
.tab-btn{padding:12px 22px;font-size:.8rem;font-weight:700;color:#c4a882;cursor:pointer;border:none;background:none;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .15s;letter-spacing:.3px;text-transform:uppercase}
.tab-btn:hover{color:#f5f0e8;background:rgba(255,255,255,.05)}
.tab-btn.active{color:#f5f0e8;border-bottom-color:#e8a250}
.tab-btn .tab-badge{display:inline-flex;align-items:center;justify-content:center;background:#e8a250;color:#2e1f10;border-radius:50%;width:16px;height:16px;font-size:.6rem;font-weight:900;margin-left:5px}

/* ── Page content ── */
#page{max-width:1180px;margin:0 auto;padding:28px 20px}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* ── Section titles ── */
.section-title{font-size:1.2rem;font-weight:800;color:#2e1f10;margin-bottom:6px}
.section-sub{font-size:.82rem;color:#7a5c3c;margin-bottom:22px;line-height:1.5}

/* ── Cards ── */
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);border:1px solid #e8ddd0}
.card-grid{display:grid;gap:16px}
.card-grid.cols-3{grid-template-columns:repeat(3,1fr)}
.card-grid.cols-4{grid-template-columns:repeat(4,1fr)}
.card-grid.cols-5{grid-template-columns:repeat(5,1fr)}
.card-grid.cols-2{grid-template-columns:repeat(2,1fr)}

/* ── Stat cards ── */
.stat-card{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 6px rgba(0,0,0,.07);border:1px solid #e8ddd0;text-align:center}
.stat-card .val{font-size:2rem;font-weight:900;color:#2e1f10;line-height:1}
.stat-card .lbl{font-size:.7rem;color:#7a5c3c;font-weight:600;margin-top:5px;text-transform:uppercase;letter-spacing:.5px}

/* ── Problem tab ── */
.problem-hero{background:linear-gradient(135deg,#2e1f10 0%,#5c3d22 100%);color:#f5f0e8;border-radius:14px;padding:36px;margin-bottom:24px}
.problem-hero h1{font-size:1.6rem;font-weight:900;margin-bottom:10px}
.problem-hero p{font-size:.88rem;color:#d4b896;line-height:1.7;max-width:700px}
.pipeline-steps{display:flex;gap:0;margin:20px 0}
.pipeline-step{flex:1;background:#fff;border:1px solid #e8ddd0;border-radius:8px;padding:14px 16px;position:relative;text-align:center}
.pipeline-step+.pipeline-step::before{content:"→";position:absolute;left:-14px;top:50%;transform:translateY(-50%);color:#b8996c;font-size:1.1rem;font-weight:900;z-index:1}
.pipeline-step .step-icon{font-size:1.5rem;margin-bottom:6px}
.pipeline-step .step-title{font-size:.72rem;font-weight:800;color:#2e1f10;text-transform:uppercase;letter-spacing:.4px}
.pipeline-step .step-desc{font-size:.7rem;color:#7a5c3c;margin-top:3px;line-height:1.4}
.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:700;margin:2px}
.pill.cpu{background:#dcfce7;color:#15803d}
.pill.gpu{background:#ede9fe;color:#5b21b6}
.pill.bert{background:#dbeafe;color:#1d4ed8}

/* ── Dataset tab ── */
.table-wrap{overflow-x:auto;border-radius:10px;border:1px solid #e8ddd0;background:#fff;margin-top:16px}
table{width:100%;border-collapse:collapse;font-size:.72rem}
thead tr{background:#3d2a18;color:#f5f0e8}
thead th{padding:10px 12px;font-weight:700;white-space:nowrap;text-align:left;font-size:.68rem;letter-spacing:.3px}
tbody tr:nth-child(even){background:#faf7f2}
tbody tr:hover{background:#f0e8d8}
tbody td{padding:8px 12px;border-bottom:1px solid #f0e8d8;white-space:nowrap;color:#2e1f10}
.label-badge-1{display:inline-block;background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-weight:700;font-size:.65rem}
.label-badge-0{display:inline-block;background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-weight:700;font-size:.65rem}
.table-pager{display:flex;align-items:center;gap:10px;padding:12px 16px;background:#faf7f2;border-top:1px solid #e8ddd0;font-size:.72rem;color:#7a5c3c}
.pager-btn{background:#3d2a18;color:#f5f0e8;border:none;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:.72rem;font-weight:700}
.pager-btn:disabled{opacity:.4;cursor:not-allowed}

/* ── Raw data sub-tabs ── */
.raw-tab-btn{padding:9px 18px;font-size:.75rem;font-weight:700;color:#7a5c3c;cursor:pointer;border:none;background:#faf7f2;border-bottom:3px solid transparent;border-top:1px solid #e8ddd0;border-left:1px solid #e8ddd0;border-right:1px solid #e8ddd0;border-radius:8px 8px 0 0;margin-bottom:-2px;transition:all .15s;white-space:nowrap}
.raw-tab-btn:hover{color:#2e1f10;background:#f0e8d8}
.raw-tab-btn.active{color:#2e1f10;background:#fff;border-bottom-color:#e8a250;border-top-color:#b8996c}
.raw-tab-btn .rtb-count{font-size:.6rem;color:#9a7a5c;font-weight:500;margin-left:4px}
.raw-panel{display:none}
.raw-panel.active{display:block}
.raw-file-meta{background:#fdf8f2;border:1px solid #e8ddd0;border-bottom:none;border-radius:0;padding:10px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.raw-file-meta .rfm-name{font-size:.8rem;font-weight:800;color:#2e1f10}
.raw-file-meta .rfm-desc{font-size:.72rem;color:#6b5440}
.raw-file-meta .rfm-count{font-size:.7rem;color:#9a7a5c;margin-left:auto;white-space:nowrap}
.raw-col-count{font-size:.65rem;background:#e8ddd0;color:#5c3d22;border-radius:4px;padding:2px 6px;font-weight:700;flex-shrink:0}
.raw-table-wrap{overflow-x:auto;border-radius:0 0 10px 10px;border:1px solid #e8ddd0;background:#fff}
.raw-table-wrap table{width:100%;border-collapse:collapse;font-size:.7rem}
.raw-table-wrap thead tr{background:#3d2a18;color:#f5f0e8}
.raw-table-wrap thead th{padding:8px 12px;font-weight:700;white-space:nowrap;text-align:left;font-size:.65rem;letter-spacing:.2px}
.raw-table-wrap tbody tr:nth-child(even){background:#faf7f2}
.raw-table-wrap tbody tr:hover{background:#f0e8d8}
.raw-table-wrap tbody td{padding:7px 12px;border-bottom:1px solid #f5f0e8;white-space:nowrap;color:#2e1f10;max-width:220px;overflow:hidden;text-overflow:ellipsis}
.raw-table-wrap tbody td:first-child{color:#9a7a5c;font-size:.62rem}

/* ── Feature table ── */
.src-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.65rem;font-weight:700;letter-spacing:.2px}
.src-patients{background:#fef3c7;color:#92400e}
.src-conditions{background:#ede9fe;color:#5b21b6}
.src-encounters{background:#ccfbf1;color:#0f766e}
.src-medications{background:#ffedd5;color:#c2410c}
.src-procedures{background:#dbeafe;color:#1e40af}
.src-observations{background:#dcfce7;color:#166534}
.feat-table-wrap{max-height:460px;overflow-y:auto;border-radius:10px;border:1px solid #e8ddd0;background:#fff}
.feat-table{width:100%;border-collapse:collapse;font-size:.72rem}
.feat-table thead{position:sticky;top:0;z-index:2}
.feat-table thead tr{background:#3d2a18;color:#f5f0e8}
.feat-table thead th{padding:9px 14px;font-weight:700;text-align:left;font-size:.67rem;letter-spacing:.3px;white-space:nowrap}
.feat-table tbody tr:nth-child(even){background:#faf7f2}
.feat-table tbody tr:hover{background:#f0e8d8}
.feat-table tbody td{padding:8px 14px;border-bottom:1px solid #f5f0e8;vertical-align:top;color:#2e1f10;line-height:1.45}
.feat-table tbody td:first-child{color:#9a7a5c;font-size:.65rem;text-align:center}
.feat-name{font-family:monospace;font-size:.72rem;font-weight:700;color:#1e1208;background:#f0e8d8;padding:2px 6px;border-radius:4px;white-space:nowrap}
.feat-group-row td{background:#f0e8d8!important;font-size:.67rem;font-weight:800;color:#5c3d22;text-transform:uppercase;letter-spacing:.5px;padding:5px 14px!important}
.feature-group{background:#fff;border-radius:10px;padding:16px 18px;border:1px solid #e8ddd0;margin-bottom:12px}
.feature-group .fg-title{font-size:.8rem;font-weight:800;color:#2e1f10;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.feature-group .fg-tags{display:flex;flex-wrap:wrap;gap:5px}
.feature-tag{background:#f0e8d8;color:#5c3d22;border-radius:6px;padding:3px 9px;font-size:.65rem;font-weight:600}

/* ── Feature engineering tab ── */
.pipeline-flow{display:flex;flex-direction:column;gap:12px;margin:16px 0}
.flow-row{display:flex;align-items:stretch;gap:12px}
.flow-node{background:#fff;border:2px solid #e8ddd0;border-radius:10px;padding:14px 18px;flex:1;position:relative}
.flow-node.source{border-color:#b8996c;background:#fdf8f2}
.flow-node.process{border-color:#7c3aed;background:#faf5ff}
.flow-node.output{border-color:#16a34a;background:#f0fdf4}
.flow-node .fn-label{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.flow-node.source .fn-label{color:#b8996c}
.flow-node.process .fn-label{color:#7c3aed}
.flow-node.output .fn-label{color:#16a34a}
.flow-node .fn-title{font-size:.85rem;font-weight:800;color:#1e1208;margin-bottom:4px}
.flow-node .fn-desc{font-size:.72rem;color:#6b5440;line-height:1.5}
.flow-arrow{display:flex;align-items:center;justify-content:center;font-size:1.4rem;color:#b8996c;font-weight:900;flex-shrink:0;width:28px}
.model-uses{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.model-use-tag{font-size:.62rem;font-weight:700;padding:2px 7px;border-radius:12px}
.mtu-cpu{background:#dcfce7;color:#15803d}
.mtu-gpu{background:#ede9fe;color:#5b21b6}
.mtu-bert{background:#dbeafe;color:#1d4ed8}

/* ── Model cards ── */
.models-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px;margin-top:16px}
.model-card{background:#fff;border-radius:14px;padding:20px;border:2px solid #e8ddd0;cursor:pointer;transition:all .2s;position:relative;user-select:none}
.model-card:hover{border-color:#b8996c;box-shadow:0 4px 16px rgba(0,0,0,.1)}
.model-card.selected{border-color:#e8a250;box-shadow:0 0 0 3px rgba(232,162,80,.25)}
.model-card .mc-check{position:absolute;top:14px;right:14px;width:22px;height:22px;border-radius:50%;border:2px solid #e8ddd0;display:flex;align-items:center;justify-content:center;font-size:.75rem;transition:all .2s;background:#fff}
.model-card.selected .mc-check{background:#e8a250;border-color:#e8a250;color:#fff}
.model-card .mc-icon{font-size:2rem;margin-bottom:10px}
.model-card .mc-category{font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;padding:2px 8px;border-radius:12px;display:inline-block}
.mc-cat-cpu{background:#dcfce7;color:#15803d}
.mc-cat-gpu{background:#ede9fe;color:#5b21b6}
.mc-cat-bert{background:#dbeafe;color:#1d4ed8}
.mc-cat-mymodel{background:#fee2e2;color:#991b1b}
.model-section-hdr{font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.8px;color:#7a5c3c;margin:22px 0 10px;display:flex;align-items:center;gap:10px}
.model-section-hdr::after{content:'';flex:1;height:1px;background:#e8ddd0}
.model-card.locked-card{opacity:.45;pointer-events:none;filter:grayscale(.6)}
.tier-lock-banner{background:linear-gradient(135deg,#1e1208 0%,#3d2a18 100%);border-radius:12px;padding:18px 22px;margin:4px 0 18px;display:flex;align-items:center;gap:16px;border:1px solid #6b4c2e}
.tlb-icon{font-size:2rem;flex-shrink:0}
.tlb-body{flex:1}
.tlb-title{font-size:.9rem;font-weight:800;color:#f5f0e8;margin-bottom:4px}
.tlb-sub{font-size:.75rem;color:#b8996c;line-height:1.5}
.tlb-need{font-weight:800;color:#e8a250}
.tlb-got{font-weight:800;color:#dc2626}
.tlb-badge{background:#e8a250;color:#2e1f10;font-weight:900;font-size:.65rem;padding:4px 12px;border-radius:20px;flex-shrink:0}
.tlb-badge.locked{background:#4b3320;color:#b8996c}
/* leaderboard */
.lb-section{background:#fff;border-radius:14px;border:1px solid #e8ddd0;overflow:hidden;margin-top:6px}
.lb-header{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);padding:18px 22px;display:flex;align-items:center;gap:12px}
.lb-header-icon{font-size:1.8rem}
.lb-header-title{font-size:1rem;font-weight:800;color:#f1f5f9}
.lb-header-sub{font-size:.72rem;color:#94a3b8;margin-top:2px}
.lb-table{width:100%;border-collapse:collapse}
.lb-table th{background:#f8fafc;font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#64748b;padding:10px 14px;border-bottom:2px solid #e2e8f0;text-align:left}
.lb-table td{padding:11px 14px;font-size:.8rem;border-bottom:1px solid #f1f5f9;color:#334155}
.lb-table tr:last-child td{border-bottom:none}
.lb-rank{font-weight:900;font-size:.9rem;color:#94a3b8;min-width:28px}
.lb-rank.top1{color:#f59e0b}
.lb-rank.top2{color:#94a3b8}
.lb-rank.top3{color:#cd7c47}
.lb-f1{font-weight:900;font-size:.95rem}
.lb-row-baseline{background:#fef9f0}
.lb-row-mine{background:#fef2f2}
.lb-tag{font-size:.6rem;font-weight:800;text-transform:uppercase;padding:2px 8px;border-radius:10px;margin-left:6px;vertical-align:middle}
.lb-tag-baseline{background:#fde68a;color:#92400e}
.lb-tag-mine{background:#fee2e2;color:#991b1b}
.lb-locked-msg{padding:30px;text-align:center;color:#9a7a5c;font-size:.82rem;line-height:1.7}
.lb-locked-icon{font-size:2.5rem;margin-bottom:10px}
.model-card .mc-name{font-size:1rem;font-weight:800;color:#1e1208;margin-bottom:6px}
.model-card .mc-auc{font-size:1.6rem;font-weight:900;margin-bottom:4px}
.model-card .mc-auc-lbl{font-size:.62rem;font-weight:600;color:#7a5c3c;text-transform:uppercase;letter-spacing:.4px}
.model-card .mc-desc{font-size:.72rem;color:#6b5440;line-height:1.5;margin-top:8px;border-top:1px solid #f0e8d8;padding-top:8px}
.model-card .mc-input{font-size:.65rem;color:#9a7a5c;margin-top:6px;font-style:italic}
.compare-bar{background:#3d2a18;color:#f5f0e8;padding:12px 20px;display:flex;align-items:center;gap:14px;border-radius:10px;margin-bottom:20px;flex-wrap:wrap}
.compare-bar .cb-title{font-size:.8rem;font-weight:700}
.selected-pills{display:flex;gap:6px;flex-wrap:wrap;flex:1}
.selected-pill{display:flex;align-items:center;gap:4px;background:rgba(255,255,255,.15);padding:4px 10px;border-radius:20px;font-size:.7rem;font-weight:600}
.selected-pill .sp-remove{cursor:pointer;opacity:.7;font-size:.8rem;margin-left:2px}
.selected-pill .sp-remove:hover{opacity:1}
.compare-go-btn{background:#e8a250;color:#2e1f10;border:none;padding:8px 18px;border-radius:8px;font-weight:800;font-size:.8rem;cursor:pointer;flex-shrink:0}
.compare-go-btn:hover{background:#f0b864}
.compare-go-btn:disabled{opacity:.4;cursor:not-allowed}

/* ── Compare tab ── */
#compare-empty{text-align:center;padding:60px 20px;color:#9a7a5c}
#compare-empty .ce-icon{font-size:3rem;margin-bottom:14px}
#compare-empty h3{font-size:1rem;font-weight:700;margin-bottom:6px}
#compare-empty p{font-size:.8rem;line-height:1.5}
#compare-content{display:none}
.compare-section{background:#fff;border-radius:12px;padding:22px;border:1px solid #e8ddd0;margin-bottom:20px}
.compare-section-title{font-size:.85rem;font-weight:800;color:#2e1f10;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #f0e8d8;display:flex;align-items:center;gap:8px}
.metric-tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.metric-tab{padding:5px 14px;border-radius:20px;font-size:.72rem;font-weight:700;cursor:pointer;border:2px solid #e8ddd0;background:#fff;color:#6b5440;transition:all .15s}
.metric-tab.active{background:#3d2a18;color:#f5f0e8;border-color:#3d2a18}
#chart-svg-wrap{width:100%;min-height:200px}
.comp-table{width:100%;border-collapse:collapse;font-size:.76rem}
.comp-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;text-align:left;color:#3d2a18;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.comp-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10;vertical-align:top}
.comp-table tr:last-child td{border-bottom:none}
.comp-table tr:hover td{background:#faf7f2}
.yes-chip{display:inline-flex;align-items:center;gap:4px;background:#dcfce7;color:#15803d;border-radius:6px;padding:2px 8px;font-size:.68rem;font-weight:700}
.no-chip{display:inline-flex;align-items:center;gap:4px;background:#fee2e2;color:#b91c1c;border-radius:6px;padding:2px 8px;font-size:.68rem;font-weight:700}
.na-chip{display:inline-flex;align-items:center;background:#f0e8d8;color:#7a5c3c;border-radius:6px;padding:2px 8px;font-size:.68rem;font-weight:700}
.pros-cons-grid{display:grid;gap:16px}
.pc-model-title{font-size:.8rem;font-weight:800;color:#1e1208;padding:10px 14px;border-radius:8px 8px 0 0;margin-bottom:0}
.pro-item,.con-item{display:flex;align-items:flex-start;gap:8px;font-size:.75rem;line-height:1.5;padding:6px 0;border-bottom:1px solid #f5f0e8;color:#2e1f10}
.pro-item:last-child,.con-item:last-child{border-bottom:none}
.pro-dot{color:#16a34a;font-weight:900;flex-shrink:0;font-size:.9rem}
.con-dot{color:#dc2626;font-weight:900;flex-shrink:0;font-size:.9rem}
.pc-section-label{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;padding:6px 0 4px;color:#7a5c3c}
.model-color-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;display:inline-block;margin-right:4px}

/* ── Reproducibility tab ── */
.repro-hero{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:28px}
.repro-hero-left{background:#2e1f10;border-radius:14px;padding:26px 28px;display:flex;flex-direction:column;justify-content:center}
.repro-hero-eyebrow{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:#7a5c3c;margin-bottom:6px}
.repro-hero-title{font-size:2rem;font-weight:900;color:#e8a250;margin-bottom:12px;line-height:1}
.repro-hero-body{font-size:.78rem;color:#c4a882;line-height:1.75;margin:0}
.repro-hero-right{display:flex;flex-direction:column;gap:10px}
.repro-concept-card{background:#fff;border:1px solid #e8ddd0;border-radius:10px;padding:14px 16px;display:flex;align-items:flex-start;gap:12px}
.repro-concept-icon{font-size:1.3rem;flex-shrink:0;line-height:1;margin-top:1px}
.repro-concept-label{font-size:.76rem;font-weight:800;color:#2e1f10;margin-bottom:3px}
.repro-concept-desc{font-size:.7rem;color:#5c3d22;line-height:1.5}
/* Range chart */
#repro-range-chart{background:#fff;border:1px solid #e8ddd0;border-radius:12px;padding:20px 24px}
/* Slider */
.repro-slider-wrap{background:#fff;border:1px solid #e8ddd0;border-radius:12px;padding:20px 24px;margin-bottom:14px}
.repro-slider-top{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.repro-slider-label{font-size:.72rem;font-weight:700;color:#9a7a5c;text-transform:uppercase;letter-spacing:.4px}
.repro-iter-display{font-size:2rem;font-weight:900;color:#2e1f10;background:#f5f0e8;border-radius:8px;padding:4px 18px;min-width:70px;text-align:center;line-height:1.2}
.repro-slider{width:100%;accent-color:#e8a250;height:6px;cursor:pointer}
.repro-slider-ticks{display:flex;justify-content:space-between;font-size:.65rem;color:#9a7a5c;margin-top:6px;padding:0 2px}
/* Insight box */
.repro-insight-box{background:#faf7f2;border-left:4px solid #e8a250;border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:18px;font-size:.78rem;color:#2e1f10;line-height:1.65;min-height:44px}
.repro-insight-winner{color:#b45309;font-weight:800}
.repro-insight-delta{font-weight:700}
.repro-insight-delta.up{color:#15803d}
.repro-insight-delta.down{color:#b91c1c}
/* Iteration table */
.repro-iter-table-wrap{overflow-x:auto;border-radius:12px;border:1px solid #e8ddd0}
.repro-iter-table{width:100%;border-collapse:collapse;font-size:.75rem}
.repro-iter-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;color:#3d2a18;text-align:left;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.repro-iter-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10;white-space:nowrap}
.repro-iter-table tr:last-child td{border-bottom:none}
.repro-iter-table tr.iter-winner td{background:#fffbeb}
.repro-iter-table tr.iter-winner td:first-child{border-left:3px solid #e8a250}
.repro-winner-badge{background:#e8a250;color:#1e1208;font-weight:900;font-size:.6rem;border-radius:10px;padding:2px 7px;margin-left:6px;text-transform:uppercase}
.repro-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle;flex-shrink:0}
.repro-winner-row{background:rgba(234,179,8,.08)!important}
.repro-winner-row td{font-weight:700}
.repro-crown{color:#f59e0b;margin-left:4px;font-size:.85rem}
.repro-delta{font-size:.7rem;font-weight:700}
.repro-delta.up{color:#15803d}
.repro-delta.down{color:#dc2626}
.repro-delta.flat{color:#9a7a5c}
/* Summary table */
.repro-summary-wrap{overflow-x:auto;border-radius:12px;border:1px solid #e8ddd0;margin-bottom:24px}
.repro-summary-table{width:100%;border-collapse:collapse;font-size:.75rem}
.repro-summary-table th{background:#f0e8d8;padding:10px 14px;font-weight:800;color:#3d2a18;text-align:left;border-bottom:2px solid #e8ddd0;white-space:nowrap}
.repro-summary-table td{padding:10px 14px;border-bottom:1px solid #f5f0e8;color:#2e1f10}
.repro-summary-table tr:last-child td{border-bottom:none}
.repro-summary-table tr:hover td{background:#faf7f2}
.repro-range-bar{display:inline-flex;align-items:center;gap:6px;font-size:.7rem;font-weight:700}
.repro-range-pill{background:#f0e8d8;border-radius:4px;padding:1px 7px;color:#5c3d22}
/* Takeaway */
.repro-takeaway{background:#f5f0e8;border-radius:12px;padding:22px 24px}
.repro-takeaway-title{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#7a5c3c;margin-bottom:16px}
.repro-takeaway-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.repro-takeaway-item{display:flex;gap:12px;align-items:flex-start;font-size:.78rem;color:#2e1f10;line-height:1.65}
.repro-takeaway-num{background:#e8a250;color:#1e1208;font-weight:900;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:.68rem;flex-shrink:0;margin-top:2px}
/* ── Tab intro banner ── */
.rq-intro-banner{display:flex;align-items:center;justify-content:space-between;gap:20px;background:#2e1f10;border-radius:14px;padding:22px 26px;margin-bottom:28px}
.rq-intro-title{font-size:1rem;font-weight:900;color:#f5f0e8;margin-bottom:5px}
.rq-intro-sub{font-size:.75rem;color:#c4a882;line-height:1.6;max-width:520px}
.rq-intro-pills{display:flex;gap:10px;flex-shrink:0}
.rq-pill{display:flex;align-items:center;gap:8px;background:rgba(255,255,255,.08);border-radius:20px;padding:7px 14px;font-size:.72rem;font-weight:700;color:#c4a882;white-space:nowrap}
.rq-pill.accent{background:#e8a250;color:#1e1208}
.rq-pill-n{font-size:1rem;font-weight:900}
/* ── Section label ── */
.rq-section-label{font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:#9a7a5c;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #e8ddd0}
.rq-section-note{font-weight:500;text-transform:none;letter-spacing:0;color:#b8996c;font-size:.68rem}
/* ── New RQ cards ── */
.rq2-card{display:flex;gap:0;background:#fff;border-radius:12px;border:1px solid #e8ddd0;margin-bottom:14px;overflow:hidden;transition:box-shadow .15s}
.rq2-card:hover{box-shadow:0 4px 16px rgba(46,31,16,.08)}
.rq2-left{width:72px;flex-shrink:0;background:#faf7f2;display:flex;flex-direction:column;align-items:center;padding:20px 10px;gap:10px;border-right:1px solid #e8ddd0}
.rq2-num{font-size:1.6rem;font-weight:900;color:#c2410c;line-height:1}
.rq2-diff{padding:3px 7px;border-radius:10px;font-size:.58rem;font-weight:800;text-transform:uppercase;letter-spacing:.3px;text-align:center}
.rq2-diff.hard{background:#fee2e2;color:#b91c1c}
.rq2-diff.medium{background:#fef9c3;color:#92400e}
.rq2-body{flex:1;padding:18px 20px}
.rq2-title{font-size:.88rem;font-weight:800;color:#1e1208;margin-bottom:8px}
.rq2-context{font-size:.77rem;color:#5c3d22;line-height:1.65;margin:0 0 12px}
.rq2-context strong{color:#2e1f10}
.rq2-question{font-size:.78rem;color:#2e1f10;line-height:1.75;background:#f5f0e8;border-left:3px solid #e8a250;padding:11px 15px;border-radius:0 8px 8px 0;margin-bottom:10px}
.rq2-question em{color:#7a5c3c;font-style:italic}
.rq2-where{font-size:.68rem;color:#9a7a5c;line-height:1.5}
.rq2-where-icon{color:#e8a250;font-weight:900;margin-right:4px}
/* ── Final Project 2.0 ── */
.fp2-hero{background:linear-gradient(135deg,#1e1208 0%,#2e1f10 60%,#3d2a18 100%);border-radius:14px;padding:2px;margin-bottom:4px}
.fp2-hero-inner{background:linear-gradient(135deg,#2e1f10 0%,#3d2a18 100%);border-radius:13px;padding:26px 28px}
.fp2-hero-top{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:22px}
.fp2-hero-title{font-size:1.2rem;font-weight:900;color:#e8a250;margin-bottom:8px}
.fp2-hero-sub{font-size:.78rem;color:#c4a882;line-height:1.7;max-width:480px}
.fp2-hero-sub code{background:rgba(255,255,255,.12);padding:2px 6px;border-radius:4px;color:#f5f0e8;font-size:.74rem}
.fp2-badge-col{display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex-shrink:0}
.fp2-file-chip{background:#e8a250;color:#1e1208;font-weight:800;font-family:monospace;font-size:.74rem;padding:9px 16px;border-radius:8px;white-space:nowrap}
.fp2-path{font-size:.65rem;color:#7a5c3c;text-align:right}
.fp2-stats-row{display:flex;align-items:center;gap:0;background:rgba(0,0,0,.2);border-radius:10px;overflow:hidden}
.fp2-stat{flex:1;text-align:center;padding:14px 10px}
.fp2-stat-val{font-size:1.15rem;font-weight:900;color:#f5f0e8}
.fp2-stat-lbl{font-size:.6rem;color:#7a5c3c;margin-top:3px;text-transform:uppercase;letter-spacing:.3px}
.fp2-stat-div{width:1px;background:rgba(255,255,255,.06);align-self:stretch}
/* ── Challenge cards ── */
.challenge-card{background:#fff;border-radius:14px;padding:26px 28px;border:1px solid #e8ddd0;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
.ch-header{display:flex;align-items:flex-start;gap:16px;margin-bottom:18px;padding-bottom:16px;border-bottom:2px solid #f0e8d8}
.ch-num{font-size:2rem;font-weight:900;color:#e8ddd0;line-height:1;flex-shrink:0;font-variant-numeric:tabular-nums}
.ch-title{font-size:.95rem;font-weight:800;color:#2e1f10;margin-bottom:3px}
.ch-file{font-size:.72rem;color:#7a5c3c;font-family:monospace;background:#f5f0e8;padding:2px 8px;border-radius:4px;display:inline-block}
.ch-badge{margin-left:auto;flex-shrink:0;padding:4px 12px;border-radius:20px;font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px}
.cpu-badge{background:#dbeafe;color:#1d4ed8}
.hard-badge{background:#fef9c3;color:#92400e;border:1px solid #fde68a}
.ch-section-label{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#7a5c3c;margin:16px 0 6px}
.ch-text{font-size:.8rem;color:#2e1f10;line-height:1.7;margin:0 0 4px}
.ch-task-list{margin:0 0 18px;padding-left:20px;font-size:.8rem;color:#2e1f10;line-height:1.9}
.ch-task-list li{margin-bottom:2px}
.ch-task-list code{font-size:.77rem;background:#f0e8d8;padding:1px 5px;border-radius:3px;color:#5c3d22}
/* Feature ideas grid */
.ch-ideas-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.ch-idea{background:#faf7f2;border:1px solid #e8ddd0;border-radius:8px;padding:10px 12px}
.ch-idea.ch-idea-stretch{border-color:#f6d860;background:#fffbeb}
.ch-idea-name{font-size:.75rem;font-weight:800;color:#2e1f10;margin-bottom:3px;font-family:monospace}
.ch-idea-desc{font-size:.7rem;color:#5c3d22;line-height:1.5;margin-bottom:4px}
.ch-idea-src{font-size:.62rem;color:#9a7a5c;font-style:italic}
/* Stacking diagram */
.ch-stacking-diagram{display:flex;align-items:center;gap:12px;background:#faf7f2;border-radius:10px;padding:16px 20px;margin-bottom:16px;flex-wrap:wrap}
.ch-stack-col{display:flex;flex-direction:column;gap:8px}
.ch-stack-model,.ch-stack-meta,.ch-stack-out{padding:10px 14px;border-radius:8px;font-size:.72rem;font-weight:700;text-align:center;line-height:1.3}
.ch-stack-model span,.ch-stack-meta span{font-weight:400;font-size:.65rem;color:inherit;opacity:.8;display:block}
.ch-stack-model.rf{background:#dbeafe;color:#1e40af}
.ch-stack-model.tfidf{background:#dcfce7;color:#15803d}
.ch-stack-model.xgb{background:#fce7f3;color:#9d174d}
.ch-stack-meta{background:#3d2a18;color:#f5f0e8;min-width:110px}
.ch-stack-meta span{color:#c4a882}
.ch-stack-out{background:#e8a250;color:#1e1208}
.ch-stack-arrow{font-size:.7rem;color:#9a7a5c;font-weight:700;text-align:center;white-space:nowrap}
/* Warning box */
.ch-warn-box{background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:12px 16px;font-size:.75rem;color:#92400e;line-height:1.6;margin-bottom:18px}
.ch-warn-box strong{display:block;margin-bottom:3px}
/* Cost matrix table */
.ch-cost-table{overflow-x:auto;margin:0 0 18px}
.ch-cost-table table{border-collapse:collapse;width:100%;font-size:.76rem}
.ch-cost-table th,.ch-cost-table td{padding:10px 14px;border:1px solid #e8ddd0;text-align:center}
.ch-cost-table th{background:#f0e8d8;font-weight:800;color:#3d2a18}
.ch-cost-table td:first-child{background:#faf7f2;font-weight:700;color:#3d2a18;text-align:left}
.cost-tp,.cost-tn{background:#f0fdf4;color:#15803d}
.cost-fp{background:#fef2f2;color:#b91c1c}
.cost-tp span,.cost-tn span,.cost-fp span,.cost-fn span{display:block;font-size:.65rem;font-weight:400;margin-top:3px}
.cost-fn{background:#fef2f2;color:#b91c1c;font-weight:700}
/* Reflect box */
.ch-reflect-box{background:#f5f0e8;border-radius:8px;padding:14px 16px;margin-bottom:18px}
.ch-reflect-title{font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.4px;color:#7a5c3c;margin-bottom:8px}
.ch-reflect-list{margin:0;padding-left:18px;font-size:.78rem;color:#2e1f10;line-height:1.8}
/* Baseline → target row */
.ch-metrics-row{display:flex;align-items:center;gap:12px;background:#faf7f2;border-radius:10px;padding:14px 18px;flex-wrap:wrap}
.ch-metric-box{text-align:center;padding:8px 16px}
.ch-metric-box.ch-target{background:#fff;border-radius:8px;border:2px solid #e8a250}
.ch-metric-val{font-size:1.1rem;font-weight:900;color:#2e1f10}
.ch-target .ch-metric-val{color:#b45309}
.ch-metric-lbl{font-size:.62rem;color:#7a5c3c;margin-top:2px}
.ch-arrow{font-size:1.2rem;color:#c4a882;font-weight:700}

/* ── Responsive ── */
@media(max-width:700px){
  .card-grid.cols-3,.card-grid.cols-4,.card-grid.cols-5{grid-template-columns:1fr 1fr}
  .pipeline-steps{flex-direction:column}
  .pipeline-step+.pipeline-step::before{content:"↓";left:50%;top:-14px;transform:translateX(-50%)}
}
@media(max-width:480px){
  .card-grid.cols-3,.card-grid.cols-4,.card-grid.cols-5,.card-grid.cols-2{grid-template-columns:1fr}
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
  <button class="tab-btn" data-tab="challenges">Research Questions + Project</button>
</div>

<!-- ══════════════════════════════════════════════════════ -->
<!-- PROBLEM TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-problem" class="tab-panel active">
<div id="page">

  <div class="problem-hero">
    <h1>Benchmarking Machine Learning Models on Synthetic Clinical Records</h1>
    <p>This project benchmarks seven machine learning models on a binary patient classification task using 12,352 synthetic patients generated by Synthea. We compare models ranging from logistic regression to a fine-tuned ClinicalBERT transformer. Each model uses patient information, including demographics, vitals, medications, and encounter notes. We evaluate performance and computational cost across 30 iterations with different random seeds.</p>
  </div>

  <!-- Benchmark at a glance -->
  <div class="card-grid cols-5" style="margin-bottom:24px">
    <div class="stat-card"><div class="val">12,352</div><div class="lbl">Synthetic Patients</div></div>
    <div class="stat-card"><div class="val" style="color:#16a34a">50.5%</div><div class="lbl">Labeled Positive</div></div>
    <div class="stat-card"><div class="val">7</div><div class="lbl">Models Compared</div></div>
    <div class="stat-card"><div class="val">30</div><div class="lbl">Random Seeds</div></div>
    <div class="stat-card"><div class="val">~4 min</div><div class="lbl">Per Iteration (A100)</div></div>
  </div>

  <!-- Context cards -->
  <div class="card-grid cols-3" style="margin-bottom:24px">
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🏥</div>
      <div class="section-title" style="font-size:.9rem;margin-bottom:6px">Clinical Context</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6;margin-bottom:8px">Chronic pain affects roughly 20% of US adults and is a major healthcare challenge. Electronic health records (EHRs) contain valuable information that can support research into pain-related conditions.</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6">In this project, we use synthetic clinical records and keyword-based labels to create a simplified classification task. The labels represent a broad set of pain-related diagnoses rather than clinically confirmed chronic pain.</div>
    </div>
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🎯</div>
      <div class="section-title" style="font-size:.9rem;margin-bottom:6px">Research Question</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6">Which machine learning architecture, from simple tabular baselines to deep learning models, offers the best trade-off between <strong>classification performance</strong> and <strong>computational cost</strong> on a synthetic clinical classification task?</div>
    </div>
    <div class="card">
      <div style="font-size:1.6rem;margin-bottom:8px">🔬</div>
      <div class="section-title" style="font-size:.9rem;margin-bottom:6px">Scientific Rigor</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6;margin-bottom:8px">All seven models share the same stratified training and validation split within each iteration. We repeat the experiments across 30 random seeds to compare average performance, variability, and computational cost under consistent conditions.</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6">The goal is to provide a reproducible environment for comparing machine learning models, rather than a clinically validated pain prediction system. The project also demonstrates how label quality and data leakage can influence model evaluation.</div>
    </div>
  </div>

  <!-- Label definition -->
  <div class="section-title">How the Label Is Built</div>
  <div class="section-sub">A patient is labeled <strong>1</strong> if any diagnosis in <code>conditions.csv</code> contains one of 31 pain-related keywords, and <strong>0</strong> otherwise. The result is close to balanced: 6,235 positive (50.5%) and 6,117 negative (49.5%).</div>
  <div class="card-grid cols-2" style="margin-bottom:24px">
    <div class="card">
      <div style="font-weight:800;font-size:.85rem;margin-bottom:10px">🏷️ Keyword List</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">
        <span class="feature-tag">chronic</span><span class="feature-tag">pain</span><span class="feature-tag">arthritis</span><span class="feature-tag">osteoarthritis</span><span class="feature-tag">rheumatoid</span><span class="feature-tag">fibromyalgia</span><span class="feature-tag">migraine</span><span class="feature-tag">neuropathy</span><span class="feature-tag">neuralgia</span><span class="feature-tag">sciatica</span><span class="feature-tag">back pain</span><span class="feature-tag">neck pain</span><span class="feature-tag">spinal</span><span class="feature-tag">fracture</span><span class="feature-tag">injury</span><span class="feature-tag">burn</span><span class="feature-tag">wound</span><span class="feature-tag">trauma</span><span class="feature-tag">sprain</span><span class="feature-tag">strain</span><span class="feature-tag">tendon</span><span class="feature-tag">ligament</span><span class="feature-tag">joint</span><span class="feature-tag">osteoporosis</span><span class="feature-tag">gout</span><span class="feature-tag">lupus</span><span class="feature-tag">paralysis</span><span class="feature-tag">amputation</span><span class="feature-tag">surgery</span><span class="feature-tag">postoperative</span><span class="feature-tag">whiplash</span>
      </div>
    </div>
    <div class="card">
      <div style="font-weight:800;font-size:.85rem;margin-bottom:8px">⚠️ What This Means for Evaluation</div>
      <div style="font-size:.76rem;color:#6b5440;line-height:1.6;margin-bottom:8px"><strong>Label quality:</strong> acute events such as a sprain, burn, or fracture count the same as fibromyalgia or chronic back pain. Models learn to detect &ldquo;any pain-related diagnosis,&rdquo; not chronic pain.</div>
      <div style="font-size:.76rem;color:#6b5440;line-height:1.6"><strong>Leakage:</strong> condition text is excluded from every model because it defines the label. Medication text is still included, and Synthea prescribes pain medications as part of the same conditions, so text-based features can act as a partial proxy for the label.</div>
    </div>
  </div>

  <!-- Pipeline overview -->
  <div class="section-title">End-to-End Pipeline</div>
  <div class="section-sub">Every model sees the same patients and the same split in each iteration &mdash; only the input representation and architecture differ.</div>
  <div class="pipeline-steps">
    <div class="pipeline-step">
      <div class="step-icon">📋</div>
      <div class="step-title">Raw Data</div>
      <div class="step-desc">Synthea COVID-19 10K export: patients, conditions, observations, and medications CSVs</div>
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
      <div style="font-size:1.2rem;margin-bottom:8px">🖥️ <span class="pill cpu">CPU Baseline</span></div>
      <div style="font-weight:800;font-size:.9rem;margin-bottom:8px">Tabular &amp; Text</div>
      <div style="font-size:.76rem;color:#6b5440;line-height:1.6;margin-bottom:10px">Random Forest and XGBoost use 13 structured features (demographics, costs, vitals). TF-IDF Logistic Regression reads each patient&rsquo;s raw medication text. All three train in a few seconds per iteration on CPU.</div>
      <div style="font-size:.7rem;color:#16a34a;font-weight:700">F1 range: 0.649 &ndash; 0.699</div>
    </div>
    <div class="card" style="border-left:4px solid #7c3aed">
      <div style="font-size:1.2rem;margin-bottom:8px">⚡ <span class="pill gpu">GPU Sequence</span></div>
      <div style="font-weight:800;font-size:.9rem;margin-bottom:8px">Sequence Models</div>
      <div style="font-size:.76rem;color:#6b5440;line-height:1.6;margin-bottom:10px">LSTM, GRU, and a custom Transformer take structured and medication-text features as 10-step sequences. The feature table has one row per patient, so each sequence is left-padded and holds a single real time step. 14&ndash;22 s per iteration on an A100.</div>
      <div style="font-size:.7rem;color:#7c3aed;font-weight:700">F1 range: 0.679 &ndash; 0.692</div>
    </div>
    <div class="card" style="border-left:4px solid #1d4ed8">
      <div style="font-size:1.2rem;margin-bottom:8px">🧠 <span class="pill bert">Pretrained BERT</span></div>
      <div style="font-weight:800;font-size:.9rem;margin-bottom:8px">Clinical Language Model</div>
      <div style="font-size:.76rem;color:#6b5440;line-height:1.6;margin-bottom:10px">Bio_ClinicalBERT, pre-trained on real clinical notes (MIMIC-III), with its last 2 layers fine-tuned on tokenized medication text and fused with structured features. Scored on the 1,445 test patients who have medication records. ~36 s per iteration.</div>
      <div style="font-size:.7rem;color:#1d4ed8;font-weight:700">F1: 0.707 (highest mean F1)</div>
    </div>
  </div>

</div>
</div><!-- /tab-problem -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- DATASET TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-dataset" class="tab-panel">
<div id="page">

  <div class="section-title">Raw Data Explorer</div>
  <div class="section-sub">Six Synthea CSV files feed the pipeline. Click a file tab to browse the first 20 rows of each. All IDs are synthetic UUIDs &mdash; no real patient data.</div>

  <!-- Stats row -->
  <div class="card-grid cols-3" style="margin-bottom:20px">
    <div class="stat-card"><div class="val" id="st-total">—</div><div class="lbl">Total Patients</div></div>
    <div class="stat-card"><div class="val" id="st-pos" style="color:#16a34a">—</div><div class="lbl">Pain Positive</div></div>
    <div class="stat-card"><div class="val" id="st-neg" style="color:#dc2626">—</div><div class="lbl">Pain Negative</div></div>
  </div>

  <!-- Class balance bar -->
  <div class="card" style="margin-bottom:22px;padding:14px 20px">
    <div style="font-weight:800;font-size:.8rem;margin-bottom:8px">Label Distribution</div>
    <div style="display:flex;gap:12px;align-items:center">
      <div style="flex:1;height:22px;border-radius:5px;overflow:hidden;display:flex">
        <div id="pos-bar" style="background:#16a34a;height:100%;transition:width .6s"></div>
        <div id="neg-bar" style="background:#dc2626;height:100%;transition:width .6s"></div>
      </div>
      <div style="font-size:.72rem;color:#6b5440;white-space:nowrap">
        <span style="color:#16a34a;font-weight:800" id="bar-pos-pct">50%</span> pain &nbsp;|&nbsp;
        <span style="color:#dc2626;font-weight:800" id="bar-neg-pct">50%</span> no pain
      </div>
    </div>
  </div>

  <!-- Raw file sub-tabs -->
  <div id="raw-tab-nav" style="display:flex;gap:0;border-bottom:2px solid #e8ddd0;margin-bottom:0;flex-wrap:wrap"></div>

  <!-- Raw table panels (one per file) -->
  <div id="raw-panels"></div>

</div>
</div><!-- /tab-dataset -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- FEATURES TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-features" class="tab-panel">
<div id="page">

  <div class="section-title">Feature Engineering Pipeline</div>
  <div class="section-sub">Raw Synthea CSVs go through five processing stages before reaching the models. Each stage adds a new type of signal. Click any stage to learn more.</div>

  <div class="pipeline-flow">

    <!-- Stage 1 -->
    <div class="flow-row">
      <div class="flow-node source">
        <div class="fn-label">Stage 0 — Raw Input</div>
        <div class="fn-title">6 Synthea CSV Files</div>
        <div class="fn-desc">patients.csv &bull; conditions.csv &bull; encounters.csv &bull; medications.csv &bull; procedures.csv &bull; observations.csv</div>
      </div>
    </div>

    <div class="flow-row"><div class="flow-arrow">↓</div></div>

    <!-- Stage 2 -->
    <div class="flow-row">
      <div class="flow-node process">
        <div class="fn-label">Stage 1 — Structured Extraction</div>
        <div class="fn-title">Patient Feature Vector</div>
        <div class="fn-desc">Join all six tables on patient ID. Aggregate per-patient: count unique medications, total encounters, average condition duration, latest vitals from observations, pain severity score. Label patients as <strong>pain=1</strong> if any active pain condition exists.</div>
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

    <!-- Stage 3 -->
    <div class="flow-row" style="gap:12px">
      <div class="flow-node process" style="flex:2">
        <div class="fn-label">Stage 2 — Text Feature Extraction</div>
        <div class="fn-title">TF-IDF Drug Frequencies + LDA Topics</div>
        <div class="fn-desc">Concatenate all medication names and descriptions per patient into a single &ldquo;medication note.&rdquo; Run TF-IDF to extract 12 key drug term frequencies (opioids, anti-inflammatories, etc.). Fit a 5-topic LDA model to discover latent medication usage patterns across the population.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-cpu">TF-IDF LogReg</span>
          <span class="model-use-tag mtu-cpu">Random Forest</span>
          <span class="model-use-tag mtu-cpu">XGBoost</span>
        </div>
      </div>
      <div class="flow-node process" style="flex:1">
        <div class="fn-label">Stage 2b — Dimensionality Reduction</div>
        <div class="fn-title">PCA &amp; UMAP</div>
        <div class="fn-desc">Compress the full TF-IDF feature space into 2 PCA components and 2 UMAP components. These low-dimensional embeddings capture the global structure of medication patterns and are appended to the structured feature vector.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-cpu">RF</span>
          <span class="model-use-tag mtu-cpu">XGB</span>
        </div>
      </div>
    </div>

    <div class="flow-row"><div class="flow-arrow">↓</div></div>

    <!-- Stage 4 -->
    <div class="flow-row" style="gap:12px">
      <div class="flow-node process" style="flex:2">
        <div class="fn-label">Stage 3 — Visit Sequences</div>
        <div class="fn-title">Time-Ordered Encounter Arrays</div>
        <div class="fn-desc">For each patient, sort encounters chronologically. For each encounter, build a feature vector (vitals, medications dispensed, procedures done). Pad or truncate to a fixed length. This 2-D array (visits &times; features) is the input to sequence models — each time-step is one clinical visit.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-gpu">LSTM</span>
          <span class="model-use-tag mtu-gpu">GRU</span>
          <span class="model-use-tag mtu-gpu">Transformer</span>
        </div>
      </div>
      <div class="flow-node process" style="flex:1">
        <div class="fn-label">Stage 3b — BERT Tokenization</div>
        <div class="fn-title">ClinicalBERT Embeddings</div>
        <div class="fn-desc">Tokenize the medication note sequence using the ClinicalBERT WordPiece tokenizer (max 512 tokens). Pre-compute the [CLS] embedding (768-dim) for every patient &mdash; this step runs once and is cached, saving GPU time during training.</div>
        <div class="model-uses" style="margin-top:10px">
          <span class="model-use-tag mtu-bert">ClinicalBERT</span>
        </div>
      </div>
    </div>

    <div class="flow-row"><div class="flow-arrow">↓</div></div>

    <!-- Stage 5 -->
    <div class="flow-row">
      <div class="flow-node output">
        <div class="fn-label">Stage 4 — Output</div>
        <div class="fn-title">Model-Specific Feature Sets Ready for Training</div>
        <div class="fn-desc">Each model type receives the representation that best fits its architecture: flat 44-dim vectors for tree models, 2-D sequence arrays for RNNs/Transformers, or 768-dim BERT embeddings fused with structured features for ClinicalBERT.</div>
      </div>
    </div>

  </div><!-- /pipeline-flow -->

  <!-- Why these features matter -->
  <div class="section-title" style="margin-top:28px">Why Medication Text?</div>
  <div class="card-grid cols-2" style="margin-bottom:20px">
    <div class="card">
      <div style="font-weight:800;font-size:.85rem;margin-bottom:8px">🔑 Key Signal</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6">Pain medications (opioids, anti-inflammatories, gabapentin) are often the earliest documented evidence of chronic pain in the EHR &mdash; sometimes appearing months before a formal diagnosis code. TF-IDF medication frequencies capture this signal even when structured pain severity scores are missing or unreported.</div>
    </div>
    <div class="card">
      <div style="font-weight:800;font-size:.85rem;margin-bottom:8px">⚠️ Leakage Risk</div>
      <div style="font-size:.78rem;color:#6b5440;line-height:1.6">We deliberately <em>excluded</em> the raw condition text (diagnosis codes and descriptions) because the pain label was created from that text directly &mdash; using it as a feature would give models a &ldquo;cheat sheet&rdquo; and produce artificially perfect AUC scores. Medication text avoids this because it is independently generated from prescribing behavior.</div>
    </div>
  </div>

  <!-- All 45 features table -->
  <div class="section-title" style="margin-top:28px">All Features Used in Training</div>
  <div class="section-sub">45 features across 5 groups. Each row shows the exact column name, what it represents, which raw file it comes from, and how it is computed from the raw data.</div>

  <!-- Source legend -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
    <span class="src-badge src-patients">patients.csv</span>
    <span class="src-badge src-conditions">conditions.csv</span>
    <span class="src-badge src-encounters">encounters.csv</span>
    <span class="src-badge src-medications">medications.csv</span>
    <span class="src-badge src-procedures">procedures.csv</span>
    <span class="src-badge src-observations">observations.csv</span>
  </div>

  <div class="feat-table-wrap">
    <table class="feat-table" id="feat-table">
      <thead>
        <tr>
          <th style="width:32px">#</th>
          <th style="min-width:200px">Feature Name</th>
          <th style="min-width:200px">Description</th>
          <th style="min-width:130px">Source File</th>
          <th style="min-width:160px">Raw Column(s) Used</th>
          <th style="min-width:260px">How Calculated</th>
        </tr>
      </thead>
      <tbody id="feat-tbody"></tbody>
    </table>
  </div>

  <!-- Feature importance visual (static SVG) -->
  <div class="card" style="margin-top:20px">
    <div style="font-weight:800;font-size:.85rem;margin-bottom:14px">Top Features by Importance (Random Forest, avg over 30 iterations)</div>
    <svg viewBox="0 0 560 200" width="100%" style="display:block">
      <defs>
        <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#e8a250"/>
          <stop offset="100%" stop-color="#b8996c"/>
        </linearGradient>
      </defs>
      <!-- bars -->
      <g transform="translate(160,10)">
        <!-- Feature names -->
        <text x="-8" y="17" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">pain_severity</text>
        <text x="-8" y="39" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">num_pain_meds</text>
        <text x="-8" y="61" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">tfidf_opioid</text>
        <text x="-8" y="83" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">tfidf_gabapentin</text>
        <text x="-8" y="105" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">num_conditions</text>
        <text x="-8" y="127" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">avg_condition_dur</text>
        <text x="-8" y="149" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">healthcare_expenses</text>
        <text x="-8" y="171" text-anchor="end" font-size="11" fill="#5c3d22" font-weight="600">tfidf_morphine</text>
        <!-- Bars -->
        <rect y="6" width="316" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="28" width="274" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="50" width="237" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="72" width="198" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="94" width="168" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="116" width="142" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="138" width="122" height="14" rx="3" fill="url(#barGrad)"/>
        <rect y="160" width="104" height="14" rx="3" fill="url(#barGrad)"/>
        <!-- Values -->
        <text x="320" y="17" font-size="10" fill="#3d2a18" font-weight="700">0.186</text>
        <text x="278" y="39" font-size="10" fill="#3d2a18" font-weight="700">0.161</text>
        <text x="241" y="61" font-size="10" fill="#3d2a18" font-weight="700">0.140</text>
        <text x="202" y="83" font-size="10" fill="#3d2a18" font-weight="700">0.117</text>
        <text x="172" y="105" font-size="10" fill="#3d2a18" font-weight="700">0.099</text>
        <text x="146" y="127" font-size="10" fill="#3d2a18" font-weight="700">0.084</text>
        <text x="126" y="149" font-size="10" fill="#3d2a18" font-weight="700">0.072</text>
        <text x="108" y="171" font-size="10" fill="#3d2a18" font-weight="700">0.061</text>
      </g>
      <text x="280" y="195" text-anchor="middle" font-size="10" fill="#9a7a5c">Mean Decrease in Impurity (normalized)</text>
    </svg>
    <div style="font-size:.7rem;color:#9a7a5c;margin-top:8px">Pain severity score and pain medication count dominate &mdash; confirming that clinical observations of pain and prescription behavior are the strongest predictors. TF-IDF drug features add complementary signal not captured in the structured counts.</div>
  </div>

</div>
</div><!-- /tab-features -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- MODELS TAB -->
<!-- ══════════════════════════════════════════════════════ -->
<div id="tab-models" class="tab-panel">
<div id="page">

  <div class="section-title">Model Gallery</div>
  <div class="section-sub">Compare your model against CPU baselines. Beat the CPU tier to unlock GPU models. Beat the GPU tier to enter the leaderboard. <strong>F1-Score is the primary metric.</strong></div>

  <!-- Selected models bar -->
  <div class="compare-bar" id="compare-bar" style="display:none">
    <span class="cb-title">Comparing:</span>
    <div class="selected-pills" id="selected-pills"></div>
    <button class="compare-go-btn" onclick="goToCompare()">Compare Selected &rarr;</button>
  </div>

  <div id="models-grid"></div>

  <!-- Leaderboard -->
  <div class="model-section-hdr" style="margin-top:28px">🏆 Class Leaderboard</div>
  <div class="lb-section" id="lb-section">
    <div class="lb-header">
      <div class="lb-header-icon">🏆</div>
      <div>
        <div class="lb-header-title">Beat the Baseline — Leaderboard</div>
        <div class="lb-header-sub">Ranked by F1-Score · Unlock by beating GPU models (F1 &gt; <strong id="lb-threshold-val"></strong>)</div>
      </div>
    </div>
    <div id="lb-body"></div>
  </div>

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

    <!-- Performance section -->
    <div class="compare-section">
      <div class="compare-section-title">📈 Performance (avg over 30 iterations)</div>
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
      <div style="font-size:.7rem;color:#9a7a5c;margin-top:10px">
        All values are means across 30 independent random-seed runs on the same stratified 80/20 train/val split.
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
      <div style="font-size:.7rem;color:#9a7a5c;margin-top:10px">
        GPU hours measured on NVIDIA A100 (80 GB) via NAIRR Classroom Pilot allocation. Training times are per iteration; ClinicalBERT includes only fine-tuning (BERT embedding precomputation runs once offline).
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
        A model that reports F1 0.770 from a single run may have simply gotten lucky
        with its random train/validation split. A model that reports F1 0.741 ± 0.007
        across 30 independent runs is telling you something you can trust.
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
  <p style="color:#5c3d22;font-size:.78rem;line-height:1.6;margin-bottom:16px">
    Each bar shows the <strong>full spread</strong> (min → max) of F1 values across 30 runs.
    The filled circle marks the <strong>mean</strong>. Notice that all models overlap — no model is
    statistically dominant.
  </p>
  <div id="repro-range-chart"></div>

  <!-- Divider -->
  <div style="height:28px"></div>

  <!-- Interactive slider section -->
  <div class="section-header">Explore Each Iteration</div>
  <p style="color:#5c3d22;font-size:.78rem;line-height:1.6;margin-bottom:20px">
    Drag the slider to see how every model performs on a specific iteration.
    Watch the winner change. This is why a single run cannot be trusted.
  </p>

  <div class="repro-slider-wrap">
    <div class="repro-slider-top">
      <span class="repro-slider-label">Iteration</span>
      <div class="repro-iter-display" id="repro-iter-num">1</div>
      <span class="repro-slider-label" style="color:#9a7a5c">of 30</span>
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
  <p style="color:#5c3d22;font-size:.78rem;line-height:1.6;margin-bottom:16px">
    Mean and standard deviation across all 30 runs. A high std means the model's apparent
    performance depends heavily on which patients ended up in the validation set.
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
        <div><strong>Never report a single run.</strong> A lucky split can inflate F1 by 0.01–0.02 over the true value.</div>
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
        <div><strong>Use shared splits.</strong> All 7 models here use the same 30 train/val splits — that's why their comparisons are fair.</div>
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
      <div class="rq-intro-title">Research Questions &amp; Final Project</div>
      <div class="rq-intro-sub">Explore the UI across all tabs before answering. There are no single right answers — only well-supported and poorly-supported ones.</div>
    </div>
    <div class="rq-intro-pills">
      <div class="rq-pill"><span class="rq-pill-n">5</span>Research Questions</div>
      <div class="rq-pill accent"><span class="rq-pill-n">1</span>Final Project</div>
    </div>
  </div>

  <!-- ── Research Questions ── -->
  <div class="rq-section-label">Research Questions <span class="rq-section-note">— answer in your written report, 150–250 words each</span></div>

  <!-- Q1 -->
  <div class="rq2-card">
    <div class="rq2-left">
      <div class="rq2-num">01</div>
    </div>
    <div class="rq2-body">
      <div class="rq2-title">The AUC Convergence Problem</div>
      <p class="rq2-context">
        Seven models — from a 5-line logistic regression to 110M-parameter BERT — all land at
        AUC <strong>0.826–0.832</strong>, a spread of 0.006. The per-model standard deviation
        across 30 iterations is ±0.007, meaning these models are statistically indistinguishable.
      </p>
      <div class="rq2-question">
        Propose a specific mechanism that explains this performance ceiling. Is it a
        <em>data limitation</em>, a <em>label quality problem</em> (chronic pain is
        subjectively diagnosed), or a <em>fundamental ambiguity</em> in the EHR signal?
        How would you empirically test your hypothesis — without collecting new data?
      </div>
      <div class="rq2-where">
        <span class="rq2-where-icon">→</span>
        Compare tab (AUC std dev) &nbsp;·&nbsp; Dataset tab (patient count, positive rate) &nbsp;·&nbsp; Feature Engineering tab (top predictors)
      </div>
    </div>
  </div>

  <!-- Q2 -->
  <div class="rq2-card">
    <div class="rq2-left">
      <div class="rq2-num">02</div>
    </div>
    <div class="rq2-body">
      <div class="rq2-title">The Efficiency Paradox</div>
      <p class="rq2-context">
        Random Forest: <strong>$0 · 5 sec · AUC 0.830 · Precision 0.761</strong>.<br/>
        ClinicalBERT: <strong>$0.05 · 49 sec · AUC 0.832 · Precision 0.829</strong>.<br/>
        The AUC gap is noise. The precision gap is real.
      </p>
      <div class="rq2-question">
        Design two clinical deployment scenarios — one where ClinicalBERT's precision
        justifies its 10× cost, and one where Random Forest is the only <em>ethical</em> choice.
        What specific factors in each scenario drive the decision?
        Your answer must cite both metrics and the computational comparison from this UI.
      </div>
      <div class="rq2-where">
        <span class="rq2-where-icon">→</span>
        Compare tab (Precision vs Recall) &nbsp;·&nbsp; Models tab (pros/cons cards) &nbsp;·&nbsp; Consider: who pays for FP vs FN?
      </div>
    </div>
  </div>

  <!-- Q3 -->
  <div class="rq2-card">
    <div class="rq2-left">
      <div class="rq2-num">03</div>
    </div>
    <div class="rq2-body">
      <div class="rq2-title">The Reproducibility Trap</div>
      <p class="rq2-context">
        Across 30 independent train/val splits, ClinicalBERT's AUC ranges from
        <strong>0.816 to 0.846</strong> — a spread of 0.030. Random Forest spans
        <strong>0.810 to 0.845</strong>. No model is consistently "the best" across every run.
        The winner changes depending on which split you happened to draw.
      </p>
      <div class="rq2-question">
        A colleague shows you a paper claiming Model A outperforms Model B by 0.015 AUC on a
        single held-out split. Using evidence from the Reproducibility tab, construct a specific
        statistical argument for why this claim may not hold. What minimum experimental design
        — number of runs, split strategy, and reporting standard — would you require before
        accepting such a comparison as valid? Justify each requirement.
      </div>
      <div class="rq2-where">
        <span class="rq2-where-icon">→</span>
        Reproducibility tab (AUC range chart &amp; per-iteration slider) &nbsp;·&nbsp; Compare tab (std dev column)
      </div>
    </div>
  </div>

  <!-- Q4 -->
  <div class="rq2-card">
    <div class="rq2-left">
      <div class="rq2-num">04</div>
    </div>
    <div class="rq2-body">
      <div class="rq2-title">The Medication Text Paradox</div>
      <p class="rq2-context">
        TF-IDF LogReg (medication text <em>only</em>) achieves AUC <strong>0.775</strong> — worst of seven.
        Random Forest (zero text features) achieves AUC <strong>0.830</strong>. Yet
        medication-derived features (tfidf_opioid, topic_1, pca_1…) rank among the top predictors
        in the Random Forest importance chart.
      </p>
      <div class="rq2-question">
        Explain the contradiction. Why does medication text help Random Forest but not
        TF-IDF LogReg as a standalone model? What does this reveal about
        <em>individual feature value</em> vs. <em>feature value in context</em>?
        Follow-up: if forced to drop exactly half the features (22 of 45), which would you keep and why?
      </div>
      <div class="rq2-where">
        <span class="rq2-where-icon">→</span>
        Feature Engineering tab (importance chart) &nbsp;·&nbsp; Models tab (TF-IDF description)
      </div>
    </div>
  </div>

  <!-- Q5 -->
  <div class="rq2-card">
    <div class="rq2-left">
      <div class="rq2-num">05</div>
    </div>
    <div class="rq2-body">
      <div class="rq2-title">The Invisible 30%</div>
      <p class="rq2-context">
        Every top model misses ~<strong>30% of actual chronic pain patients</strong> (Recall ≈ 0.70).
        These are real people whose pain goes undetected by all seven models.
        You cannot run code to answer this question.
      </p>
      <div class="rq2-question">
        Using only what's visible in this UI, hypothesize what kind of patient is most
        likely to be in the missed 30%. Consider age, medication patterns, encounter count,
        QALY scores. Then propose one specific change — to features or training — that would
        most reduce false negatives, and explain what you sacrifice in return.
      </div>
      <div class="rq2-where">
        <span class="rq2-where-icon">→</span>
        Feature Engineering tab (QALY, DALY, pain_severity) &nbsp;·&nbsp; Dataset tab → patients.csv &nbsp;·&nbsp; Compare tab (Recall across models)
      </div>
    </div>
  </div>

  <!-- ── Final Project ── -->
  <div class="rq-section-label" style="margin-top:36px">Final Project <span class="rq-section-note">— graded coding component</span></div>

  <div class="fp2-hero">
    <div class="fp2-hero-inner">
      <div class="fp2-hero-top">
        <div>
          <div class="fp2-hero-title">Beat the Baseline</div>
          <div class="fp2-hero-sub">Start with the starter file. Modify any code as needed. Your best model counts for your grade.</div>
        </div>
        <div class="fp2-badge-col">
          <div class="fp2-file-chip">📄 final_project_starter.py</div>
          <div class="fp2-path">ai-sandbox/final-project-ui/challenges/</div>
        </div>
      </div>
      <div class="fp2-stats-row">
        <div class="fp2-stat">
          <div class="fp2-stat-val">0.741</div>
          <div class="fp2-stat-lbl">Baseline F1 (RF)</div>
        </div>
        <div class="fp2-stat-div"></div>
        <div class="fp2-stat">
          <div class="fp2-stat-val">45</div>
          <div class="fp2-stat-lbl">Starting features</div>
        </div>
        <div class="fp2-stat-div"></div>
        <div class="fp2-stat">
          <div class="fp2-stat-val">6</div>
          <div class="fp2-stat-lbl">Raw CSV files</div>
        </div>
        <div class="fp2-stat-div"></div>
      </div>
    </div>
  </div>

</div>
</div><!-- /tab-challenges -->


<script>
// ── Dataset stats injected from Python ──────────────────────────────────────
const DATASET_STATS = {{ stats }};

// ── My Model metrics — re-read from CSV on every page load ───────────────────
const MY_MODEL_STATS = {{ my_model }};

// ── Per-iteration metrics (all 30 runs, all 7 models) ────────────────────────
const ITER_DATA = {"iter1":{"random_forest":{"auc":0.8259,"accuracy":0.7422,"precision":0.758,"recall":0.7185,"f1":0.7378},"xgboost":{"auc":0.821,"accuracy":0.7329,"precision":0.7464,"recall":0.7129,"f1":0.7293},"tfidf_logreg":{"auc":0.7726,"accuracy":0.6959,"precision":0.8216,"recall":0.5875,"f1":0.6851},"lstm":{"auc":0.832,"accuracy":0.7487,"precision":0.777,"recall":0.7041,"f1":0.7387},"gru":{"auc":0.8293,"accuracy":0.7475,"precision":0.762,"recall":0.7265,"f1":0.7438},"transformer":{"auc":0.8302,"accuracy":0.7454,"precision":0.7508,"recall":0.7418,"f1":0.7463},"clinicalbert":{"auc":0.8367,"accuracy":0.7504,"precision":0.839,"recall":0.689,"f1":0.7566}},"iter2":{"random_forest":{"auc":0.8187,"accuracy":0.7361,"precision":0.7567,"recall":0.7033,"f1":0.729},"xgboost":{"auc":0.8131,"accuracy":0.7297,"precision":0.7439,"recall":0.7081,"f1":0.7256},"tfidf_logreg":{"auc":0.7603,"accuracy":0.687,"precision":0.8161,"recall":0.5796,"f1":0.6778},"lstm":{"auc":0.8256,"accuracy":0.7382,"precision":0.8049,"recall":0.6351,"f1":0.71},"gru":{"auc":0.8264,"accuracy":0.7369,"precision":0.7953,"recall":0.6447,"f1":0.7121},"transformer":{"auc":0.8211,"accuracy":0.7268,"precision":0.7263,"recall":0.7362,"f1":0.7312},"clinicalbert":{"auc":0.8242,"accuracy":0.7351,"precision":0.8152,"recall":0.69,"f1":0.7474}},"iter3":{"random_forest":{"auc":0.8338,"accuracy":0.7479,"precision":0.7727,"recall":0.7089,"f1":0.7394},"xgboost":{"auc":0.8315,"accuracy":0.7442,"precision":0.7695,"recall":0.7041,"f1":0.7353},"tfidf_logreg":{"auc":0.7739,"accuracy":0.6993,"precision":0.8169,"recall":0.603,"f1":0.6938},"lstm":{"auc":0.837,"accuracy":0.756,"precision":0.784,"recall":0.7129,"f1":0.7467},"gru":{"auc":0.841,"accuracy":0.7519,"precision":0.7757,"recall":0.7153,"f1":0.7443},"transformer":{"auc":0.8343,"accuracy":0.7592,"precision":0.8165,"recall":0.6744,"f1":0.7387},"clinicalbert":{"auc":0.8461,"accuracy":0.7514,"precision":0.8498,"recall":0.6803,"f1":0.7557}},"iter4":{"random_forest":{"auc":0.8285,"accuracy":0.7491,"precision":0.7576,"recall":0.7394,"f1":0.7484},"xgboost":{"auc":0.8284,"accuracy":0.7406,"precision":0.7459,"recall":0.737,"f1":0.7414},"tfidf_logreg":{"auc":0.7681,"accuracy":0.6889,"precision":0.8012,"recall":0.5976,"f1":0.6846},"lstm":{"auc":0.8288,"accuracy":0.741,"precision":0.8008,"recall":0.648,"f1":0.7163},"gru":{"auc":0.8256,"accuracy":0.7394,"precision":0.7685,"recall":0.6921,"f1":0.7283},"transformer":{"auc":0.823,"accuracy":0.7285,"precision":0.7951,"recall":0.6223,"f1":0.6982},"clinicalbert":{"auc":0.8268,"accuracy":0.7347,"precision":0.8124,"recall":0.6897,"f1":0.746}},"iter5":{"random_forest":{"auc":0.825,"accuracy":0.7487,"precision":0.7717,"recall":0.7129,"f1":0.7411},"xgboost":{"auc":0.8249,"accuracy":0.7418,"precision":0.7565,"recall":0.7201,"f1":0.7379},"tfidf_logreg":{"auc":0.7617,"accuracy":0.6781,"precision":0.7943,"recall":0.5728,"f1":0.6656},"lstm":{"auc":0.8238,"accuracy":0.7353,"precision":0.7738,"recall":0.672,"f1":0.7193},"gru":{"auc":0.8264,"accuracy":0.7349,"precision":0.7454,"recall":0.7209,"f1":0.733},"transformer":{"auc":0.8136,"accuracy":0.724,"precision":0.7703,"recall":0.6455,"f1":0.7024},"clinicalbert":{"auc":0.8239,"accuracy":0.7335,"precision":0.8149,"recall":0.6775,"f1":0.7399}},"iter6":{"random_forest":{"auc":0.8288,"accuracy":0.741,"precision":0.7506,"recall":0.7289,"f1":0.7396},"xgboost":{"auc":0.8217,"accuracy":0.7374,"precision":0.7351,"recall":0.7498,"f1":0.7424},"tfidf_logreg":{"auc":0.7777,"accuracy":0.7118,"precision":0.8285,"recall":0.6135,"f1":0.705},"lstm":{"auc":0.8353,"accuracy":0.7418,"precision":0.7931,"recall":0.6608,"f1":0.7209},"gru":{"auc":0.8384,"accuracy":0.7519,"precision":0.7866,"recall":0.6977,"f1":0.7395},"transformer":{"auc":0.8306,"accuracy":0.7454,"precision":0.8078,"recall":0.6504,"f1":0.7206},"clinicalbert":{"auc":0.8377,"accuracy":0.7544,"precision":0.8332,"recall":0.7034,"f1":0.7628}},"iter7":{"random_forest":{"auc":0.8283,"accuracy":0.7386,"precision":0.7461,"recall":0.7306,"f1":0.7382},"xgboost":{"auc":0.8271,"accuracy":0.7418,"precision":0.7458,"recall":0.741,"f1":0.7434},"tfidf_logreg":{"auc":0.7697,"accuracy":0.6959,"precision":0.8185,"recall":0.5883,"f1":0.6846},"lstm":{"auc":0.8196,"accuracy":0.7422,"precision":0.7743,"recall":0.6905,"f1":0.73},"gru":{"auc":0.8292,"accuracy":0.7382,"precision":0.7717,"recall":0.6832,"f1":0.7248},"transformer":{"auc":0.8199,"accuracy":0.7382,"precision":0.7768,"recall":0.6752,"f1":0.7224},"clinicalbert":{"auc":0.8345,"accuracy":0.7514,"precision":0.8352,"recall":0.6938,"f1":0.758}},"iter8":{"random_forest":{"auc":0.8276,"accuracy":0.7475,"precision":0.7706,"recall":0.7113,"f1":0.7398},"xgboost":{"auc":0.8237,"accuracy":0.7442,"precision":0.763,"recall":0.7153,"f1":0.7384},"tfidf_logreg":{"auc":0.7957,"accuracy":0.7116,"precision":0.8355,"recall":0.6015,"f1":0.6995},"lstm":{"auc":0.833,"accuracy":0.7539,"precision":0.7795,"recall":0.7145,"f1":0.7456},"gru":{"auc":0.8293,"accuracy":0.7511,"precision":0.7981,"recall":0.6784,"f1":0.7334},"transformer":{"auc":0.8278,"accuracy":0.7454,"precision":0.761,"recall":0.7225,"f1":0.7413},"clinicalbert":{"auc":0.8459,"accuracy":0.764,"precision":0.8545,"recall":0.6955,"f1":0.7668}},"iter9":{"random_forest":{"auc":0.8369,"accuracy":0.7515,"precision":0.7627,"recall":0.737,"f1":0.7496},"xgboost":{"auc":0.8282,"accuracy":0.7483,"precision":0.7692,"recall":0.7161,"f1":0.7417},"tfidf_logreg":{"auc":0.7779,"accuracy":0.6899,"precision":0.8302,"recall":0.5673,"f1":0.674},"lstm":{"auc":0.838,"accuracy":0.7471,"precision":0.7874,"recall":0.6832,"f1":0.7316},"gru":{"auc":0.8378,"accuracy":0.7584,"precision":0.7866,"recall":0.7153,"f1":0.7493},"transformer":{"auc":0.8315,"accuracy":0.7471,"precision":0.7613,"recall":0.7265,"f1":0.7435},"clinicalbert":{"auc":0.8414,"accuracy":0.7555,"precision":0.8507,"recall":0.688,"f1":0.7608}},"iter10":{"random_forest":{"auc":0.8286,"accuracy":0.7446,"precision":0.7588,"recall":0.7241,"f1":0.7411},"xgboost":{"auc":0.8193,"accuracy":0.7321,"precision":0.7524,"recall":0.6993,"f1":0.7249},"tfidf_logreg":{"auc":0.7616,"accuracy":0.6936,"precision":0.8075,"recall":0.6021,"f1":0.6898},"lstm":{"auc":0.8327,"accuracy":0.7459,"precision":0.7901,"recall":0.676,"f1":0.7286},"gru":{"auc":0.8268,"accuracy":0.7495,"precision":0.7973,"recall":0.6752,"f1":0.7312},"transformer":{"auc":0.8274,"accuracy":0.7426,"precision":0.7604,"recall":0.7153,"f1":0.7372},"clinicalbert":{"auc":0.8305,"accuracy":0.7484,"precision":0.8363,"recall":0.6906,"f1":0.7565}},"iter11":{"random_forest":{"auc":0.8384,"accuracy":0.7483,"precision":0.7633,"recall":0.7265,"f1":0.7445},"xgboost":{"auc":0.8209,"accuracy":0.7276,"precision":0.7184,"recall":0.757,"f1":0.7372},"tfidf_logreg":{"auc":0.7729,"accuracy":0.7017,"precision":0.8515,"recall":0.5666,"f1":0.6805},"lstm":{"auc":0.8376,"accuracy":0.7531,"precision":0.803,"recall":0.6768,"f1":0.7346},"gru":{"auc":0.8385,"accuracy":0.7548,"precision":0.7809,"recall":0.7145,"f1":0.7462},"transformer":{"auc":0.8279,"accuracy":0.7402,"precision":0.7923,"recall":0.6576,"f1":0.7187},"clinicalbert":{"auc":0.8334,"accuracy":0.7445,"precision":0.8303,"recall":0.6841,"f1":0.7501}},"iter12":{"random_forest":{"auc":0.8315,"accuracy":0.745,"precision":0.7556,"recall":0.7314,"f1":0.7433},"xgboost":{"auc":0.8297,"accuracy":0.7463,"precision":0.757,"recall":0.7322,"f1":0.7444},"tfidf_logreg":{"auc":0.7727,"accuracy":0.6971,"precision":0.7993,"recall":0.6143,"f1":0.6947},"lstm":{"auc":0.8329,"accuracy":0.7483,"precision":0.7875,"recall":0.6864,"f1":0.7335},"gru":{"auc":0.8299,"accuracy":0.7434,"precision":0.7867,"recall":0.6744,"f1":0.7263},"transformer":{"auc":0.824,"accuracy":0.7357,"precision":0.7513,"recall":0.7121,"f1":0.7312},"clinicalbert":{"auc":0.8357,"accuracy":0.7492,"precision":0.8189,"recall":0.71,"f1":0.7606}},"iter13":{"random_forest":{"auc":0.8294,"accuracy":0.7374,"precision":0.7459,"recall":0.7273,"f1":0.7365},"xgboost":{"auc":0.8226,"accuracy":0.7293,"precision":0.7357,"recall":0.7233,"f1":0.7295},"tfidf_logreg":{"auc":0.7741,"accuracy":0.697,"precision":0.8189,"recall":0.5994,"f1":0.6922},"lstm":{"auc":0.8284,"accuracy":0.7406,"precision":0.7672,"recall":0.6977,"f1":0.7308},"gru":{"auc":0.8325,"accuracy":0.745,"precision":0.7709,"recall":0.7041,"f1":0.736},"transformer":{"auc":0.8186,"accuracy":0.7313,"precision":0.7511,"recall":0.6993,"f1":0.7243},"clinicalbert":{"auc":0.831,"accuracy":0.7389,"precision":0.8229,"recall":0.6888,"f1":0.7499}},"iter14":{"random_forest":{"auc":0.8309,"accuracy":0.7471,"precision":0.76,"recall":0.7289,"f1":0.7442},"xgboost":{"auc":0.8262,"accuracy":0.745,"precision":0.7569,"recall":0.7289,"f1":0.7426},"tfidf_logreg":{"auc":0.771,"accuracy":0.6913,"precision":0.8078,"recall":0.5943,"f1":0.6848},"lstm":{"auc":0.8259,"accuracy":0.7422,"precision":0.802,"recall":0.6496,"f1":0.7178},"gru":{"auc":0.8288,"accuracy":0.741,"precision":0.7888,"recall":0.6648,"f1":0.7215},"transformer":{"auc":0.8255,"accuracy":0.7434,"precision":0.7622,"recall":0.7145,"f1":0.7376},"clinicalbert":{"auc":0.827,"accuracy":0.7445,"precision":0.8239,"recall":0.6959,"f1":0.7545}},"iter15":{"random_forest":{"auc":0.8364,"accuracy":0.7491,"precision":0.7597,"recall":0.7354,"f1":0.7474},"xgboost":{"auc":0.8237,"accuracy":0.7353,"precision":0.7436,"recall":0.7257,"f1":0.7346},"tfidf_logreg":{"auc":0.781,"accuracy":0.7001,"precision":0.8229,"recall":0.5954,"f1":0.6909},"lstm":{"auc":0.8361,"accuracy":0.7426,"precision":0.798,"recall":0.656,"f1":0.7201},"gru":{"auc":0.8334,"accuracy":0.7442,"precision":0.7695,"recall":0.7041,"f1":0.7353},"transformer":{"auc":0.8321,"accuracy":0.7325,"precision":0.8547,"recall":0.5662,"f1":0.6811},"clinicalbert":{"auc":0.8354,"accuracy":0.7442,"precision":0.8274,"recall":0.6894,"f1":0.7521}},"iter16":{"random_forest":{"auc":0.8358,"accuracy":0.7515,"precision":0.7721,"recall":0.7201,"f1":0.7452},"xgboost":{"auc":0.8232,"accuracy":0.7382,"precision":0.7582,"recall":0.7065,"f1":0.7314},"tfidf_logreg":{"auc":0.7717,"accuracy":0.6898,"precision":0.8112,"recall":0.579,"f1":0.6757},"lstm":{"auc":0.8373,"accuracy":0.7576,"precision":0.7967,"recall":0.6977,"f1":0.7439},"gru":{"auc":0.8333,"accuracy":0.7479,"precision":0.7938,"recall":0.676,"f1":0.7302},"transformer":{"auc":0.8273,"accuracy":0.7386,"precision":0.8133,"recall":0.6255,"f1":0.7072},"clinicalbert":{"auc":0.8348,"accuracy":0.7495,"precision":0.8283,"recall":0.6952,"f1":0.7559}},"iter17":{"random_forest":{"auc":0.8245,"accuracy":0.739,"precision":0.7551,"recall":0.7145,"f1":0.7342},"xgboost":{"auc":0.8281,"accuracy":0.7426,"precision":0.7548,"recall":0.7257,"f1":0.74},"tfidf_logreg":{"auc":0.776,"accuracy":0.6921,"precision":0.8132,"recall":0.5865,"f1":0.6815},"lstm":{"auc":0.8325,"accuracy":0.7454,"precision":0.7605,"recall":0.7233,"f1":0.7415},"gru":{"auc":0.8328,"accuracy":0.7467,"precision":0.7872,"recall":0.6824,"f1":0.7311},"transformer":{"auc":0.8193,"accuracy":0.741,"precision":0.7752,"recall":0.6856,"f1":0.7277},"clinicalbert":{"auc":0.828,"accuracy":0.7462,"precision":0.8337,"recall":0.6848,"f1":0.7519}},"iter18":{"random_forest":{"auc":0.8433,"accuracy":0.7568,"precision":0.7723,"recall":0.7346,"f1":0.753},"xgboost":{"auc":0.8371,"accuracy":0.7511,"precision":0.7528,"recall":0.7546,"f1":0.7537},"tfidf_logreg":{"auc":0.7771,"accuracy":0.6986,"precision":0.8142,"recall":0.6116,"f1":0.6985},"lstm":{"auc":0.8456,"accuracy":0.7604,"precision":0.811,"recall":0.6848,"f1":0.7426},"gru":{"auc":0.8431,"accuracy":0.7608,"precision":0.8112,"recall":0.6856,"f1":0.7432},"transformer":{"auc":0.8385,"accuracy":0.7576,"precision":0.8176,"recall":0.6688,"f1":0.7358},"clinicalbert":{"auc":0.8361,"accuracy":0.7437,"precision":0.8326,"recall":0.6896,"f1":0.7544}},"iter19":{"random_forest":{"auc":0.8282,"accuracy":0.7418,"precision":0.7623,"recall":0.7097,"f1":0.735},"xgboost":{"auc":0.8179,"accuracy":0.7285,"precision":0.747,"recall":0.6985,"f1":0.7219},"tfidf_logreg":{"auc":0.7723,"accuracy":0.6874,"precision":0.8107,"recall":0.587,"f1":0.681},"lstm":{"auc":0.8278,"accuracy":0.7386,"precision":0.7926,"recall":0.6528,"f1":0.7159},"gru":{"auc":0.8295,"accuracy":0.7365,"precision":0.7614,"recall":0.6961,"f1":0.7273},"transformer":{"auc":0.8231,"accuracy":0.7365,"precision":0.7775,"recall":0.6696,"f1":0.7195},"clinicalbert":{"auc":0.8284,"accuracy":0.7384,"precision":0.8286,"recall":0.6806,"f1":0.7473}},"iter20":{"random_forest":{"auc":0.8292,"accuracy":0.7442,"precision":0.7478,"recall":0.7442,"f1":0.746},"xgboost":{"auc":0.8177,"accuracy":0.7349,"precision":0.7239,"recall":0.7674,"f1":0.745},"tfidf_logreg":{"auc":0.7822,"accuracy":0.7137,"precision":0.8313,"recall":0.6204,"f1":0.7105},"lstm":{"auc":0.8343,"accuracy":0.7507,"precision":0.7897,"recall":0.6897,"f1":0.7363},"gru":{"auc":0.8344,"accuracy":0.7539,"precision":0.7752,"recall":0.7217,"f1":0.7475},"transformer":{"auc":0.8273,"accuracy":0.7434,"precision":0.7789,"recall":0.6864,"f1":0.7298},"clinicalbert":{"auc":0.831,"accuracy":0.7457,"precision":0.8223,"recall":0.7028,"f1":0.7579}},"iter21":{"random_forest":{"auc":0.8301,"accuracy":0.7454,"precision":0.761,"recall":0.7225,"f1":0.7413},"xgboost":{"auc":0.8195,"accuracy":0.7321,"precision":0.7423,"recall":0.7185,"f1":0.7302},"tfidf_logreg":{"auc":0.7741,"accuracy":0.6866,"precision":0.8089,"recall":0.5743,"f1":0.6717},"lstm":{"auc":0.8346,"accuracy":0.7459,"precision":0.7906,"recall":0.6752,"f1":0.7284},"gru":{"auc":0.8371,"accuracy":0.7511,"precision":0.7832,"recall":0.7009,"f1":0.7397},"transformer":{"auc":0.8302,"accuracy":0.7414,"precision":0.8187,"recall":0.6263,"f1":0.7097},"clinicalbert":{"auc":0.8314,"accuracy":0.7385,"precision":0.8303,"recall":0.6682,"f1":0.7405}},"iter22":{"random_forest":{"auc":0.8447,"accuracy":0.7624,"precision":0.7797,"recall":0.7378,"f1":0.7581},"xgboost":{"auc":0.8151,"accuracy":0.7232,"precision":0.7768,"recall":0.6335,"f1":0.6979},"tfidf_logreg":{"auc":0.7908,"accuracy":0.6976,"precision":0.8238,"recall":0.598,"f1":0.693},"lstm":{"auc":0.833,"accuracy":0.7515,"precision":0.7989,"recall":0.6784,"f1":0.7337},"gru":{"auc":0.8305,"accuracy":0.7446,"precision":0.7601,"recall":0.7217,"f1":0.7404},"transformer":{"auc":0.8281,"accuracy":0.7487,"precision":0.7953,"recall":0.676,"f1":0.7308},"clinicalbert":{"auc":0.835,"accuracy":0.7512,"precision":0.8387,"recall":0.6983,"f1":0.762}},"iter23":{"random_forest":{"auc":0.8309,"accuracy":0.7426,"precision":0.7716,"recall":0.6961,"f1":0.7319},"xgboost":{"auc":0.8216,"accuracy":0.7272,"precision":0.7468,"recall":0.6953,"f1":0.7201},"tfidf_logreg":{"auc":0.7674,"accuracy":0.6884,"precision":0.8304,"recall":0.5641,"f1":0.6718},"lstm":{"auc":0.8335,"accuracy":0.7479,"precision":0.7977,"recall":0.6704,"f1":0.7285},"gru":{"auc":0.8339,"accuracy":0.7434,"precision":0.7724,"recall":0.6969,"f1":0.7327},"transformer":{"auc":0.8263,"accuracy":0.7382,"precision":0.8171,"recall":0.6199,"f1":0.705},"clinicalbert":{"auc":0.8325,"accuracy":0.7405,"precision":0.8331,"recall":0.6765,"f1":0.7467}},"iter24":{"random_forest":{"auc":0.8283,"accuracy":0.7378,"precision":0.7634,"recall":0.6961,"f1":0.7282},"xgboost":{"auc":0.8254,"accuracy":0.7454,"precision":0.7692,"recall":0.7081,"f1":0.7374},"tfidf_logreg":{"auc":0.7876,"accuracy":0.7036,"precision":0.863,"recall":0.5663,"f1":0.6838},"lstm":{"auc":0.8335,"accuracy":0.7454,"precision":0.7937,"recall":0.6696,"f1":0.7264},"gru":{"auc":0.8327,"accuracy":0.7463,"precision":0.7549,"recall":0.7362,"f1":0.7454},"transformer":{"auc":0.8288,"accuracy":0.7325,"precision":0.822,"recall":0.5998,"f1":0.6936},"clinicalbert":{"auc":0.8314,"accuracy":0.7419,"precision":0.8316,"recall":0.6821,"f1":0.7495}},"iter25":{"random_forest":{"auc":0.8389,"accuracy":0.7511,"precision":0.7642,"recall":0.733,"f1":0.7483},"xgboost":{"auc":0.8323,"accuracy":0.7386,"precision":0.7657,"recall":0.6945,"f1":0.7283},"tfidf_logreg":{"auc":0.7814,"accuracy":0.7072,"precision":0.8628,"recall":0.582,"f1":0.6951},"lstm":{"auc":0.838,"accuracy":0.7572,"precision":0.8055,"recall":0.684,"f1":0.7398},"gru":{"auc":0.8393,"accuracy":0.7612,"precision":0.8005,"recall":0.7017,"f1":0.7479},"transformer":{"auc":0.838,"accuracy":0.7479,"precision":0.821,"recall":0.6399,"f1":0.7192},"clinicalbert":{"auc":0.8407,"accuracy":0.7524,"precision":0.8333,"recall":0.7104,"f1":0.7669}},"iter26":{"random_forest":{"auc":0.8225,"accuracy":0.7382,"precision":0.75,"recall":0.7217,"f1":0.7356},"xgboost":{"auc":0.8215,"accuracy":0.7353,"precision":0.7452,"recall":0.7225,"f1":0.7337},"tfidf_logreg":{"auc":0.7764,"accuracy":0.6985,"precision":0.8092,"recall":0.6063,"f1":0.6932},"lstm":{"auc":0.8235,"accuracy":0.7382,"precision":0.7712,"recall":0.684,"f1":0.725},"gru":{"auc":0.8269,"accuracy":0.7382,"precision":0.7874,"recall":0.6592,"f1":0.7176},"transformer":{"auc":0.8149,"accuracy":0.7333,"precision":0.7795,"recall":0.6576,"f1":0.7134},"clinicalbert":{"auc":0.8314,"accuracy":0.748,"precision":0.8242,"recall":0.701,"f1":0.7577}},"iter27":{"random_forest":{"auc":0.8102,"accuracy":0.7301,"precision":0.7474,"recall":0.7025,"f1":0.7243},"xgboost":{"auc":0.8109,"accuracy":0.7378,"precision":0.7371,"recall":0.7466,"f1":0.7418},"tfidf_logreg":{"auc":0.7757,"accuracy":0.6911,"precision":0.8089,"recall":0.5957,"f1":0.6861},"lstm":{"auc":0.8194,"accuracy":0.7361,"precision":0.7902,"recall":0.6496,"f1":0.713},"gru":{"auc":0.8262,"accuracy":0.7426,"precision":0.7725,"recall":0.6945,"f1":0.7314},"transformer":{"auc":0.8123,"accuracy":0.7285,"precision":0.7807,"recall":0.6423,"f1":0.7048},"clinicalbert":{"auc":0.8241,"accuracy":0.7358,"precision":0.8259,"recall":0.6764,"f1":0.7437}},"iter28":{"random_forest":{"auc":0.8283,"accuracy":0.7406,"precision":0.7635,"recall":0.7041,"f1":0.7326},"xgboost":{"auc":0.8241,"accuracy":0.7369,"precision":0.7692,"recall":0.684,"f1":0.7241},"tfidf_logreg":{"auc":0.7641,"accuracy":0.6806,"precision":0.7953,"recall":0.595,"f1":0.6807},"lstm":{"auc":0.8289,"accuracy":0.7406,"precision":0.773,"recall":0.6881,"f1":0.728},"gru":{"auc":0.8334,"accuracy":0.7442,"precision":0.7644,"recall":0.7129,"f1":0.7378},"transformer":{"auc":0.8113,"accuracy":0.7232,"precision":0.7917,"recall":0.6127,"f1":0.6908},"clinicalbert":{"auc":0.8159,"accuracy":0.7207,"precision":0.8238,"recall":0.6513,"f1":0.7275}},"iter29":{"random_forest":{"auc":0.8375,"accuracy":0.7535,"precision":0.7685,"recall":0.7322,"f1":0.7499},"xgboost":{"auc":0.841,"accuracy":0.7661,"precision":0.7762,"recall":0.7538,"f1":0.7648},"tfidf_logreg":{"auc":0.7828,"accuracy":0.6917,"precision":0.8127,"recall":0.5912,"f1":0.6844},"lstm":{"auc":0.8421,"accuracy":0.7584,"precision":0.7982,"recall":0.6977,"f1":0.7445},"gru":{"auc":0.8399,"accuracy":0.7535,"precision":0.7853,"recall":0.7041,"f1":0.7425},"transformer":{"auc":0.841,"accuracy":0.7535,"precision":0.7981,"recall":0.6848,"f1":0.7372},"clinicalbert":{"auc":0.8398,"accuracy":0.7599,"precision":0.8245,"recall":0.7311,"f1":0.775}},"iter30":{"random_forest":{"auc":0.8168,"accuracy":0.7398,"precision":0.7568,"recall":0.7137,"f1":0.7346},"xgboost":{"auc":0.8076,"accuracy":0.722,"precision":0.7393,"recall":0.6937,"f1":0.7158},"tfidf_logreg":{"auc":0.7682,"accuracy":0.6872,"precision":0.8288,"recall":0.553,"f1":0.6634},"lstm":{"auc":0.8183,"accuracy":0.7341,"precision":0.7831,"recall":0.6544,"f1":0.713},"gru":{"auc":0.812,"accuracy":0.7293,"precision":0.7623,"recall":0.6736,"f1":0.7152},"transformer":{"auc":0.8178,"accuracy":0.7374,"precision":0.7258,"recall":0.7706,"f1":0.7476},"clinicalbert":{"auc":0.822,"accuracy":0.7264,"precision":0.8057,"recall":0.6709,"f1":0.7322}}};

// ── Model data ──────────────────────────────────────────────────────────────
const MODELS = [
  {
    id: "random_forest",
    name: "Random Forest",
    icon: "🌲",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#16a34a",
    description: "An ensemble of decision trees, each trained on a random sample of patients and features. The majority vote across hundreds of trees smooths out individual errors and prevents overfitting.",
    input: "44 structured + text features",
    auc: 0.830,
    accuracy: 0.745,
    f1: 0.741,
    precision: 0.761,
    recall: 0.722,
    trainTimeSec: 5,
    gpuHrs: 0,
    gpuRequired: false,
    params: "500 trees",
    interpretable: true,
    costPerRun: "~$0 (CPU)",
    pros: [
      "No feature scaling needed — handles raw numeric ranges",
      "Feature importance is built-in and easy to interpret",
      "Robust to outliers and missing values",
      "Fastest to train — ~5 sec per iteration on CPU",
      "No GPU required — runs on any laptop"
    ],
    cons: [
      "Cannot read text or sequence data directly",
      "Trees are independent — misses temporal patterns across visits",
      "Model can be very large in memory (hundreds of trees)",
      "Doesn't generalize well beyond the feature space it was trained on"
    ]
  },
  {
    id: "xgboost",
    name: "XGBoost",
    icon: "⚡",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#ea580c",
    description: "Gradient boosted trees that train sequentially — each new tree corrects the mistakes of the previous one. XGBoost adds regularization to reduce overfitting, making it one of the most powerful tabular models.",
    input: "44 structured + text features",
    auc: 0.824,
    accuracy: 0.738,
    f1: 0.735,
    precision: 0.752,
    recall: 0.719,
    trainTimeSec: 5,
    gpuHrs: 0,
    gpuRequired: false,
    params: "300 boosted trees",
    interpretable: true,
    costPerRun: "~$0 (CPU)",
    pros: [
      "Often the best-performing model for tabular data across Kaggle competitions",
      "Built-in regularization reduces overfitting",
      "Handles class imbalance with scale_pos_weight",
      "Fast training — ~5 sec per iteration",
      "Feature importance available via gain or SHAP values"
    ],
    cons: [
      "Sequential tree building limits parallelism compared to Random Forest",
      "More hyperparameters to tune (learning rate, depth, subsampling)",
      "Cannot process text sequences or temporal ordering",
      "Slightly worse AUC than sequence models on this dataset"
    ]
  },
  {
    id: "tfidf_logreg",
    name: "TF-IDF + LogReg",
    icon: "📝",
    category: "cpu",
    categoryLabel: "CPU Baseline",
    color: "#0891b2",
    description: "A logistic regression classifier that works on bag-of-words medication text. TF-IDF converts each patient's medication list into a vector of word frequencies, then logistic regression finds a linear decision boundary.",
    input: "Medication note text only",
    auc: 0.775,
    accuracy: 0.695,
    f1: 0.686,
    precision: 0.820,
    recall: 0.590,
    trainTimeSec: 5,
    gpuHrs: 0,
    gpuRequired: false,
    params: "~50K vocab weights",
    interpretable: true,
    costPerRun: "~$0 (CPU)",
    pros: [
      "Fully interpretable — coefficients directly show which drug terms predict pain",
      "Trains in under 5 seconds",
      "High precision (0.82) — when it predicts pain, it is usually right",
      "Works on raw text without needing structured feature extraction"
    ],
    cons: [
      "Lowest AUC (0.775) — text alone is weaker than structured clinical features",
      "Bag-of-words ignores word order — 'no pain' treated same as 'pain'",
      "Fixed vocabulary — rare or misspelled drug names are invisible",
      "Low recall (0.59) — misses nearly half of actual pain patients"
    ]
  },
  {
    id: "lstm",
    name: "LSTM",
    icon: "🔁",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#7c3aed",
    description: "Long Short-Term Memory networks read patient visits one at a time, in order. A gating mechanism decides what to remember and what to forget as the model moves through the sequence, giving it a useful long-term memory.",
    input: "Per-visit sequence arrays",
    auc: 0.832,
    accuracy: 0.746,
    f1: 0.730,
    precision: 0.789,
    recall: 0.679,
    trainTimeSec: 26,
    gpuHrs: 0.007,
    gpuRequired: true,
    params: "~120K parameters",
    interpretable: false,
    costPerRun: "~$0.03 GPU",
    pros: [
      "Captures temporal patterns — learns how a patient's condition evolves over time",
      "Handles variable-length visit histories with padding",
      "Highest AUC (tied at 0.832) with relatively simple architecture",
      "Memory gates prevent vanishing gradient on moderate-length sequences"
    ],
    cons: [
      "Requires a GPU — cannot run efficiently on CPU",
      "Black box — no direct feature importance scores",
      "Sequential computation prevents parallelism during training",
      "Struggles with very long sequences (100+ visits)"
    ]
  },
  {
    id: "gru",
    name: "GRU",
    icon: "⚙️",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#9333ea",
    description: "Gated Recurrent Units are a streamlined version of LSTM with fewer gates, making them faster to train while retaining most of the sequential memory capability. On this dataset they match LSTM's performance.",
    input: "Per-visit sequence arrays",
    auc: 0.832,
    accuracy: 0.746,
    f1: 0.735,
    precision: 0.779,
    recall: 0.696,
    trainTimeSec: 17,
    gpuHrs: 0.005,
    gpuRequired: true,
    params: "~95K parameters",
    interpretable: false,
    costPerRun: "~$0.02 GPU",
    pros: [
      "Fewer parameters than LSTM — trains in ~17 sec vs 26 sec",
      "Full suite of metrics available (AUC, F1, Precision, Recall)",
      "Matched LSTM's AUC at 0.832 with a simpler architecture",
      "Better recall (0.70) than TF-IDF — catches more true pain patients"
    ],
    cons: [
      "Requires GPU",
      "Black box with no direct interpretability",
      "Marginally less expressive than LSTM on very long or complex sequences",
      "Gains over tree models are modest given the added infrastructure"
    ]
  },
  {
    id: "transformer",
    name: "Transformer",
    icon: "🔮",
    category: "gpu",
    categoryLabel: "GPU Sequence",
    color: "#4f46e5",
    description: "A from-scratch Transformer encoder with multi-head self-attention. Instead of reading visits one-by-one like LSTM/GRU, it looks at all visits simultaneously and learns which pairs of visits are most relevant to each other.",
    input: "Per-visit sequence arrays (masked)",
    auc: 0.826,
    accuracy: 0.740,
    f1: 0.722,
    precision: 0.785,
    recall: 0.671,
    trainTimeSec: 31,
    gpuHrs: 0.008,
    gpuRequired: true,
    params: "~200K parameters",
    interpretable: false,
    costPerRun: "~$0.03 GPU",
    pros: [
      "Attention mechanism can attend to any pair of visits — no sequential bottleneck",
      "Parallelizable training — all time steps processed simultaneously",
      "Attention weights give some post-hoc interpretability",
      "Comparable training time to LSTM (~31 sec) despite larger architecture"
    ],
    cons: [
      "Slightly lower AUC (0.826) than LSTM/GRU on this small dataset",
      "Needs more data to outperform RNNs — here the dataset may be too small",
      "Requires positional encoding for sequence ordering",
      "Most complex custom model — hardest to implement correctly"
    ]
  },
  {
    id: "clinicalbert",
    name: "ClinicalBERT",
    icon: "🧠",
    category: "bert",
    categoryLabel: "Pretrained BERT",
    color: "#1d4ed8",
    description: "BERT pre-trained on 2 billion words from real clinical notes (MIMIC-III). We fine-tune the final classification layer on our medication note sequences. The 110M pre-trained parameters give it deep clinical language understanding from day one.",
    input: "Tokenized medication notes + structured features",
    auc: 0.832,
    accuracy: 0.745,
    f1: 0.753,
    precision: 0.829,
    recall: 0.690,
    trainTimeSec: 49,
    gpuHrs: 0.014,
    gpuRequired: true,
    params: "110M (pretrained BERT)",
    interpretable: false,
    costPerRun: "~$0.05 GPU (fine-tune only)",
    pros: [
      "Highest F1 (0.753) and Precision (0.829) — best at being confident and correct",
      "Transfer learning from 2B clinical words — understands medical language nuances",
      "Fuses both text embeddings and structured tabular features",
      "Pre-training cost is paid once by the research community; we only pay fine-tuning"
    ],
    cons: [
      "110M parameters — by far the largest model; inference is expensive",
      "CLS embedding precomputation is a one-time but significant upfront GPU cost",
      "Requires specialized clinical tokenizer — cannot use off-the-shelf text tools",
      "Least interpretable — 110M weights make it a true black box",
      "Needs high-memory GPU (A100 80 GB) for full batch sizes"
    ]
  },
  {
    id: "my_model",
    name: "My Model",
    icon: "🧪",
    category: "my_model",
    categoryLabel: "My Submission",
    color: "#dc2626",
    description: "Your model trained on the Synthea chronic pain dataset. Modify features or the classifier in final_project_starter.py and re-run to improve this score.",
    input: "Custom feature set",
    auc:       MY_MODEL_STATS.auc,
    accuracy:  MY_MODEL_STATS.accuracy,
    f1:        MY_MODEL_STATS.f1,
    precision: MY_MODEL_STATS.precision,
    recall:    MY_MODEL_STATS.recall,
    trainTimeSec: null,
    gpuHrs: null,
    gpuRequired: false,
    params: "—",
    interpretable: null,
    costPerRun: "—",
    pros: [
      "This is your work — you control every feature and every parameter",
      `Precision ${MY_MODEL_STATS.precision.toFixed(3)} — check vs CPU baselines`,
      "No GPU required in the baseline starter"
    ],
    cons: [
      `F1 ${MY_MODEL_STATS.f1.toFixed(3)} — improve in final_project_starter.py`,
      `Recall ${MY_MODEL_STATS.recall.toFixed(3)} — consider cost-sensitive training to catch more patients`,
      `Accuracy ${MY_MODEL_STATS.accuracy.toFixed(3)}`
    ]
  }
];

// ── Tier thresholds (F1-based) ─────────────────────────────────────────────
const CPU_MODELS   = ['random_forest','xgboost','tfidf_logreg'];
const GPU_MODELS   = ['lstm','gru','transformer','clinicalbert'];
const CPU_BEST_F1  = Math.max(...CPU_MODELS.map(id => MODELS.find(m => m.id === id).f1));
const GPU_BEST_F1  = Math.max(...GPU_MODELS.map(id => MODELS.find(m => m.id === id).f1));
const MY_MODEL_F1  = MODELS.find(m => m.id === 'my_model').f1;
const gpuUnlocked  = MY_MODEL_F1 > CPU_BEST_F1;
const lbUnlocked   = MY_MODEL_F1 > GPU_BEST_F1;

// ── Leaderboard data ──────────────────────────────────────────────────────
const LEADERBOARD = [
  { name: "Instructor Baseline (ClinicalBERT)", f1: 0.753, model: "ClinicalBERT fine-tune", tag: "baseline", notes: "Beat this to lead the class" },
  { name: "Student A",  f1: 0.761, model: "RF + temporal features",  tag: "student", notes: "" },
  { name: "Student B",  f1: 0.758, model: "XGBoost + SMOTE",         tag: "student", notes: "" },
  { name: "Student C",  f1: 0.755, model: "Stacking ensemble",        tag: "student", notes: "" },
  { name: "Student D",  f1: 0.748, model: "LightGBM + cost-sensitive",tag: "student", notes: "" },
];

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
     <text x="${xScale(t)}" y="${tickLineY2 + 16}" text-anchor="middle" fill="#94a3b8" font-size="11">${t.toFixed(2)}</text>`
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
        <text x="${padL - 8}" y="${cy + 4}" text-anchor="end" fill="${labelColor}" font-size="11.5" font-style="italic" font-weight="600">${s.label}${gpuTag}</text>
        <line x1="${padL}" y1="${cy}" x2="${padL + innerW}" y2="${cy}" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>
        <text x="${padL + innerW/2}" y="${cy + 4}" text-anchor="middle" fill="#334155" font-size="11">🔒 Unlock GPU tier to reveal</text>
      `;
    }
    const dots = s.vals.map(a =>
      `<circle cx="${xScale(a)}" cy="${cy}" r="3" fill="${color}" fill-opacity="0.35"/>`
    ).join('');
    return `
      <text x="${padL - 8}" y="${cy + 4}" text-anchor="end" fill="${labelColor}" font-size="12" font-weight="700">${s.label}${gpuTag}</text>
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
    <text x="${W/2}" y="16" text-anchor="middle" fill="#94a3b8" font-size="12">F1-Score Range across 30 Iterations — dots = individual runs, bar = min→max, filled circle = mean</text>
    ${tickLines}
    ${dividerLine}
    ${rows}
    <text x="${padL + innerW/2}" y="${H - 4}" text-anchor="middle" fill="#64748b" font-size="11">F1-Score (harmonic mean of precision &amp; recall)</text>
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
    Each iteration uses a <em>different random train/val split</em> — notice how rankings shift between runs.`;
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
          <div class="tlb-title" style="font-size:.8rem;color:#4ade80">GPU Tier Unlocked!</div>
          <div class="tlb-sub" style="font-size:.7rem">
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
          <div class="tlb-title" style="font-size:.8rem">GPU Models Locked</div>
          <div class="tlb-sub" style="font-size:.7rem">
            Beat CPU best F1 (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>) with your model to unlock GPU rows.
            Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span>
          </div>
        </div>
        <div class="tlb-badge locked">⬆ BEAT RF</div>
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
          <div class="tlb-title" style="font-size:.78rem;color:#4ade80">GPU Stats Unlocked!</div>
          <div class="tlb-sub" style="font-size:.68rem">
            Your F1 <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> beat the CPU best
            (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>). Full 30-run GPU statistics are now shown below.
          </div>
        </div>
        <div class="tlb-badge" style="background:#16a34a;color:#fff;font-size:.6rem">✓ UNLOCKED</div>
      </div>
    </td>
  </tr>`
    : `<tr style="background:#0f172a">
    <td colspan="8" style="padding:8px 14px">
      <div class="tier-lock-banner" style="margin:0;padding:10px 16px">
        <div class="tlb-icon" style="font-size:1.3rem">🔒</div>
        <div class="tlb-body">
          <div class="tlb-title" style="font-size:.78rem">GPU model stats hidden</div>
          <div class="tlb-sub" style="font-size:.68rem">
            Beat CPU best F1 (<span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span>) to reveal all 30-run GPU statistics.
            Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span>
          </div>
        </div>
        <div class="tlb-badge locked" style="font-size:.6rem">⬆ BEAT RF</div>
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
  document.getElementById('st-total').textContent = s.n_total.toLocaleString();
  document.getElementById('st-pos').textContent = s.n_positive.toLocaleString();
  document.getElementById('st-neg').textContent = s.n_negative.toLocaleString();
  const posP = s.pos_rate;
  const negP = (100 - posP).toFixed(1);
  document.getElementById('pos-bar').style.width = posP + '%';
  document.getElementById('neg-bar').style.width = negP + '%';
  document.getElementById('bar-pos-pct').textContent = posP + '%';
  document.getElementById('bar-neg-pct').textContent = negP + '%';
}

// ── Raw data sub-tabs ──────────────────────────────────────────────────────────
const RAW_META = {
  patients:     { icon: '🧍', desc: 'One row per patient — demographics, birthdate, address, insurance, race, ethnicity.' },
  conditions:   { icon: '🩺', desc: 'Active and resolved clinical conditions per patient, with ICD-equivalent codes and descriptions.' },
  encounters:   { icon: '🏥', desc: 'Every clinical visit — type (outpatient, emergency, etc.), provider, payer, cost.' },
  medications:  { icon: '💊', desc: 'Prescriptions and dispensing events — drug name, dose, cost, payer coverage.' },
  procedures:   { icon: '🔬', desc: 'Clinical procedures performed per encounter — code, description, base cost.' },
  observations: { icon: '📋', desc: 'Lab results and vital signs per encounter — LOINC code, value, units.' },
};

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
    const header = d.cols.map(c => `<th>${c}</th>`).join('');
    const body = d.rows.map((row, ri) => {
      const tds = d.cols.map(c => {
        let v = row[c] || '';
        // truncate long UUIDs for readability
        if (/^[0-9a-f]{8}-[0-9a-f]{4}-/i.test(v)) v = v.slice(0, 8) + '…';
        return `<td title="${row[c] || ''}">${v || '—'}</td>`;
      }).join('');
      return `<tr><td>${ri + 1}</td>${tds}</tr>`;
    }).join('');

    return `
    <div class="raw-panel${i === 0 ? ' active' : ''}" id="rp-${key}">
      <div class="raw-file-meta">
        <span class="rfm-name">${meta.icon} ${key}.csv</span>
        <span class="raw-col-count">${d.cols.length} columns</span>
        <span class="rfm-desc">${meta.desc}</span>
        <span class="rfm-count">Showing 20 of ${d.total.toLocaleString()} rows</span>
      </div>
      <div class="raw-table-wrap">
        <table>
          <thead><tr><th>#</th>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
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
  const f1Val = locked ? '🔒' : (m.f1 != null ? m.f1.toFixed(3) : '—');
  const f1Color = locked ? '#94a3b8' : m.color;
  return `
  <div class="model-card${locked ? ' locked-card' : ''}" id="mc-${m.id}" onclick="${locked ? '' : "toggleModel('"+m.id+"')"}">
    <div class="mc-check" id="mc-check-${m.id}">✓</div>
    <div class="mc-icon">${m.icon}</div>
    <div><span class="mc-category ${catClass}">${m.categoryLabel}</span></div>
    <div class="mc-name">${m.name}</div>
    <div class="mc-auc" style="color:${f1Color}">${f1Val}</div>
    <div class="mc-auc-lbl">${locked ? 'Unlock to reveal' : 'F1-Score (avg 30 runs)'}</div>
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
        Need to beat: <span class="tlb-need">${CPU_BEST_F1.toFixed(3)}</span> (Random Forest)
        ${cpuBeat ? `&nbsp;·&nbsp; You lead by <span class="tlb-need">+${cpuDelta}</span>` : `&nbsp;·&nbsp; Gap: <span class="tlb-got">${cpuDelta}</span>`}
        <br/>Improve your model and re-run to unlock GPU models (if locked).
      </div>
    </div>
    <div class="tlb-badge ${cpuBeat ? '' : 'locked'}">${cpuBeat ? '✓ UNLOCKED' : '⬆ BEAT RF F1'}</div>
  </div>`;

  // Section 2 — GPU Models (locked or unlocked)
  html += `<div class="model-section-hdr" style="margin-top:4px">🧠 Tier 2 — GPU Sequence Models</div>`;
  html += `<div class="models-grid" style="margin-bottom:8px">`;
  gpuModels.forEach(m => { html += modelCard(m, !gpuUnlocked); });
  html += `</div>`;

  if (gpuUnlocked) {
    const gpuDelta = (MY_MODEL_F1 - GPU_BEST_F1).toFixed(3);
    const gpuBeat = MY_MODEL_F1 > GPU_BEST_F1;
    html += `<div class="tier-lock-banner">
      <div class="tlb-icon">${gpuBeat ? '🏆' : '🔒'}</div>
      <div class="tlb-body">
        <div class="tlb-title">${gpuBeat ? 'Leaderboard Unlocked!' : 'Leaderboard Locked'}</div>
        <div class="tlb-sub">
          Your F1: <span class="tlb-got">${MY_MODEL_F1.toFixed(3)}</span> &nbsp;·&nbsp;
          Need to beat: <span class="tlb-need">${GPU_BEST_F1.toFixed(3)}</span> (ClinicalBERT)
          ${gpuBeat ? `&nbsp;·&nbsp; You're on the board! Gap: <span class="tlb-need">+${Math.abs(gpuDelta)}</span>` : `&nbsp;·&nbsp; Gap: <span class="tlb-got">${gpuDelta}</span>`}
        </div>
      </div>
      <div class="tlb-badge ${gpuBeat ? '' : 'locked'}">${gpuBeat ? '🏆 ON BOARD' : '⬆ BEAT BERT F1'}</div>
    </div>`;
  }

  grid.innerHTML = html;
  renderLeaderboard();
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

function renderLeaderboard() {
  const threshEl = document.getElementById('lb-threshold-val');
  if (threshEl) threshEl.textContent = GPU_BEST_F1.toFixed(3);

  const body = document.getElementById('lb-body');
  if (!body) return;

  if (!lbUnlocked) {
    body.innerHTML = `<div class="lb-locked-msg">
      <div class="lb-locked-icon">🔒</div>
      <strong>Leaderboard locked.</strong><br/>
      Beat all GPU models (F1 &gt; ${GPU_BEST_F1.toFixed(3)}) to appear on the leaderboard.<br/>
      Your current F1: <strong style="color:#dc2626">${MY_MODEL_F1.toFixed(3)}</strong> &nbsp;·&nbsp;
      Gap: <strong>${(MY_MODEL_F1 - GPU_BEST_F1).toFixed(3)}</strong>
    </div>`;
    return;
  }

  // Build ranked list with My Model inserted
  const entries = [...LEADERBOARD, {
    name: "You (My Model)", f1: MY_MODEL_F1, model: "final_project_starter.py", tag: "mine", notes: "Your submission"
  }].sort((a, b) => b.f1 - a.f1);

  const rankClass = i => i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : '';
  const tagHtml = tag => {
    if (tag === 'baseline') return `<span class="lb-tag lb-tag-baseline">Baseline</span>`;
    if (tag === 'mine')     return `<span class="lb-tag lb-tag-mine">You</span>`;
    return '';
  };

  body.innerHTML = `<table class="lb-table">
    <thead><tr>
      <th>Rank</th><th>Name</th><th>Model</th>
      <th>F1-Score ★</th><th>Notes</th>
    </tr></thead>
    <tbody>
      ${entries.map((e, i) => `
        <tr class="${e.tag === 'baseline' ? 'lb-row-baseline' : e.tag === 'mine' ? 'lb-row-mine' : ''}">
          <td class="lb-rank ${rankClass(i)}">${i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '#' + (i+1)}</td>
          <td><strong>${e.name}</strong>${tagHtml(e.tag)}</td>
          <td style="color:#64748b;font-size:.75rem">${e.model}</td>
          <td class="lb-f1" style="color:${e.tag==='mine'?'#dc2626':e.tag==='baseline'?'#92400e':'#1d4ed8'}">${e.f1.toFixed(3)}</td>
          <td style="color:#94a3b8;font-size:.72rem">${e.notes}</td>
        </tr>`).join('')}
    </tbody>
  </table>`;
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
    wrap.innerHTML = '<div style="text-align:center;padding:40px;color:#9a7a5c;font-size:.8rem">No data logged for this metric for selected models.</div>';
    return;
  }

  const maxVal = Math.max(...vals.filter(v => v !== null), 0.95);
  const minVal = Math.min(...vals.filter(v => v !== null), 0.6);
  const scale = v => chartH - (v - minVal) / (maxVal - minVal + 0.05) * chartH;

  let svgBars = selected.map((m, i) => {
    const v = m[metric];
    const cx = PAD_L + gap * i + gap / 2;
    if (v === null) {
      return `<text x="${cx}" y="${PAD_T + chartH / 2}" text-anchor="middle" font-size="10" fill="#9a7a5c">—</text>`;
    }
    const y = PAD_T + scale(v);
    const barH = (PAD_T + chartH) - y;
    return `
      <rect x="${cx - barW / 2}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${m.color}" opacity=".85"/>
      <text x="${cx}" y="${y - 5}" text-anchor="middle" font-size="10" fill="#2e1f10" font-weight="700">${v.toFixed(3)}</text>
    `;
  }).join('');

  let svgLabels = selected.map((m, i) => {
    const cx = PAD_L + gap * i + gap / 2;
    return `<text x="${cx}" y="${H - 8}" text-anchor="middle" font-size="10" fill="${m.color}" font-weight="700">${m.name}</text>`;
  }).join('');

  // Y-axis ticks
  const ticks = [minVal, (minVal + maxVal) / 2, maxVal].map(v => {
    const y = PAD_T + scale(v);
    return `<text x="${PAD_L - 5}" y="${y + 3}" text-anchor="end" font-size="9" fill="#9a7a5c">${v.toFixed(2)}</text>
            <line x1="${PAD_L}" y1="${y}" x2="${W - PAD_R}" y2="${y}" stroke="#f0e8d8" stroke-width="1"/>`;
  }).join('');

  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
    ${ticks}
    <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${PAD_T + chartH}" stroke="#e8ddd0" stroke-width="1.5"/>
    <line x1="${PAD_L}" y1="${PAD_T + chartH}" x2="${W - PAD_R}" y2="${PAD_T + chartH}" stroke="#e8ddd0" stroke-width="1.5"/>
    ${svgBars}
    ${svgLabels}
    <text x="${W / 2}" y="12" text-anchor="middle" font-size="10" fill="#7a5c3c" font-weight="700">${METRIC_LABELS[metric]}</text>
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
      return `<td${isF1 ? ' style="background:#fef9f0"' : ''}><strong${isF1 ? ' style="color:#c2410c"' : ''}>${v.toFixed(3)}</strong> <span style="color:#9a7a5c;font-size:.65rem">(${pct}%)</span></td>`;
    }).join('');
    const rowStyle = isF1 ? ' style="background:#fef9f0"' : '';
    const labelStyle = isF1 ? ' style="font-weight:900;color:#c2410c"' : 'style="font-weight:700;color:#3d2a18"';
    return `<tr${rowStyle}><td ${labelStyle}>${colLabels[i]}${isF1 ? ' ★' : ''}</td>${cells}</tr>`;
  }).join('');
}

function renderComputeTable(selected) {
  const thead = document.getElementById('comp-thead-row');
  const tbody = document.getElementById('comp-tbody');

  thead.innerHTML = '<th>Attribute</th>' + selected.map(m =>
    `<th style="color:${m.color}">${m.icon} ${m.name}</th>`).join('');

  const rows = [
    {label: 'Requires GPU', fn: m => m.id === 'my_model' ? '<span class="na-chip">—</span>' : m.gpuRequired ? '<span class="yes-chip">✓ Yes</span>' : '<span class="no-chip">✗ No</span>'},
    {label: 'Training Time / Iter', fn: m => m.trainTimeSec == null ? '<span class="na-chip">—</span>' : m.trainTimeSec < 60 ? m.trainTimeSec + ' sec' : Math.round(m.trainTimeSec / 60) + ' min ' + (m.trainTimeSec % 60) + ' sec'},
    {label: 'GPU Hours / Iter', fn: m => m.gpuHrs == null ? '<span class="na-chip">—</span>' : m.gpuHrs === 0 ? '0 (CPU only)' : m.gpuHrs.toFixed(3) + ' hr'},
    {label: 'Parameters', fn: m => m.params},
    {label: 'Interpretable?', fn: m => m.id === 'my_model' ? '<span class="na-chip">—</span>' : m.interpretable ? '<span class="yes-chip">✓ Yes</span>' : '<span class="no-chip">✗ No</span>'},
    {label: 'Cost / Run (A100)', fn: m => m.costPerRun},
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
      ? `<div style="color:#9a7a5c;font-size:.78rem;font-style:italic;padding:16px 0">
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
const FEATURES = [
  // ── Demographics & Identity ──────────────────────────────────────────────
  {group:"Demographics & Identity"},
  {n:"race",         desc:"Racial category of the patient (white / black / asian / other)",          src:"patients",     cols:"RACE",                                          calc:"Label-encoded to integer for modeling"},
  {n:"ethnicity",    desc:"Hispanic or non-Hispanic ethnic background",                               src:"patients",     cols:"ETHNICITY",                                     calc:"Binary-encoded (0 = non-Hispanic, 1 = Hispanic)"},
  {n:"age",          desc:"Patient age in years at time of data extraction",                          src:"patients",     cols:"BIRTHDATE",                                     calc:"(extraction_date − BIRTHDATE) / 365.25"},
  {n:"is_female",    desc:"Binary gender indicator",                                                  src:"patients",     cols:"GENDER",                                        calc:"1 if GENDER == 'F', else 0"},
  // ── Financial ──────────────────────────────────────────────────────────
  {group:"Financial & Quality of Life"},
  {n:"healthcare_expenses", desc:"Total cumulative healthcare expenses billed over patient lifetime ($)", src:"patients",  cols:"HEALTHCARE_EXPENSES",                           calc:"Direct field — running total from all claims"},
  {n:"healthcare_coverage", desc:"Total insurance coverage amount received ($)",                          src:"patients",  cols:"HEALTHCARE_COVERAGE",                           calc:"Direct field — payer contributions to date"},
  {n:"qaly",         desc:"Quality-Adjusted Life Year — 1 QALY = one year in perfect health",        src:"observations", cols:"VALUE, DESCRIPTION",                            calc:"Row where DESCRIPTION contains 'QALY'; latest VALUE per patient"},
  {n:"daly",         desc:"Disability-Adjusted Life Year — years of healthy life lost to disease",    src:"observations", cols:"VALUE, DESCRIPTION",                            calc:"Row where DESCRIPTION contains 'DALY'; latest VALUE per patient"},
  {n:"qols",         desc:"Quality of Life Score — composite wellbeing measure",                      src:"observations", cols:"VALUE, DESCRIPTION",                            calc:"Row where DESCRIPTION contains 'Quality'; latest VALUE per patient"},
  // ── Medical History ──────────────────────────────────────────────────────
  {group:"Medical History"},
  {n:"num_conditions",         desc:"Total number of clinical conditions ever recorded for the patient",          src:"conditions",  cols:"PATIENT",              calc:"COUNT(*) grouped by PATIENT"},
  {n:"avg_condition_duration", desc:"Average duration of resolved conditions in days",                            src:"conditions",  cols:"START, STOP, PATIENT", calc:"mean((STOP − START).days) per patient; open conditions excluded"},
  {n:"num_encounters",         desc:"Total number of clinical visits (all types) across the patient's history",   src:"encounters",  cols:"PATIENT",              calc:"COUNT(*) grouped by PATIENT"},
  {n:"num_procedures",         desc:"Total number of procedures performed across all encounters",                  src:"procedures",  cols:"PATIENT",              calc:"COUNT(*) grouped by PATIENT"},
  {n:"unique_procedures",      desc:"Number of distinct procedure types ever performed on the patient",           src:"procedures",  cols:"CODE, PATIENT",        calc:"COUNT(DISTINCT CODE) grouped by PATIENT"},
  // ── Medications ─────────────────────────────────────────────────────────
  {group:"Medications"},
  {n:"num_unique_meds", desc:"Number of distinct medications ever prescribed or dispensed",                    src:"medications", cols:"CODE, PATIENT",        calc:"COUNT(DISTINCT CODE) grouped by PATIENT"},
  {n:"num_pain_meds",   desc:"Number of pain-targeted medication entries (opioids, NSAIDs, gabapentinoids)", src:"medications", cols:"DESCRIPTION, PATIENT",  calc:"COUNT(*) where DESCRIPTION matches pain drug keyword list"},
  // ── Vitals & Pain ───────────────────────────────────────────────────────
  {group:"Clinical Vitals & Pain Score"},
  {n:"pain_severity_0_10_verbal_numeric_rating_score_reported", desc:"Self-reported pain intensity on 0–10 verbal numeric scale", src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows where DESCRIPTION matches 'Pain severity'; mean(VALUE) per patient"},
  {n:"body_height",              desc:"Patient height in centimeters",                   src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows where DESCRIPTION == 'Body Height'; latest VALUE"},
  {n:"body_weight",              desc:"Patient weight in kilograms",                     src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows where DESCRIPTION == 'Body Weight'; latest VALUE"},
  {n:"body_mass_index",          desc:"Body Mass Index (kg/m²)",                         src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows where DESCRIPTION == 'Body Mass Index'; latest VALUE"},
  {n:"systolic_blood_pressure",  desc:"Systolic (peak) arterial pressure in mmHg",       src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows matching 'Systolic Blood Pressure' (LOINC 8480-6); mean(VALUE)"},
  {n:"diastolic_blood_pressure", desc:"Diastolic (resting) arterial pressure in mmHg",   src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows matching 'Diastolic Blood Pressure' (LOINC 8462-4); mean(VALUE)"},
  {n:"heart_rate",               desc:"Heart rate in beats per minute",                   src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows matching 'Heart rate' (LOINC 8867-4); mean(VALUE)"},
  {n:"respiratory_rate",         desc:"Breathing rate in breaths per minute",             src:"observations", cols:"VALUE, DESCRIPTION, PATIENT", calc:"Rows matching 'Respiratory rate' (LOINC 9279-1); mean(VALUE)"},
  // ── TF-IDF Drug Terms ───────────────────────────────────────────────────
  {group:"TF-IDF Medication Term Frequencies"},
  {n:"tfidf_opioid",       desc:"How prominently 'opioid' appears in this patient's medication history vs. all patients", src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF: tf(term, doc) × log(N / df(term)); fit across all patient medication notes"},
  {n:"tfidf_acetaminophen",desc:"TF-IDF weight for acetaminophen (Tylenol — pain/fever reliever)",      src:"medications", cols:"DESCRIPTION, PATIENT", calc:"Same TF-IDF formula applied to term 'acetaminophen'"},
  {n:"tfidf_ibuprofen",    desc:"TF-IDF weight for ibuprofen (Advil — NSAID anti-inflammatory)",        src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'ibuprofen' in concatenated medication descriptions per patient"},
  {n:"tfidf_gabapentin",   desc:"TF-IDF weight for gabapentin (Neurontin — nerve pain / anticonvulsant)",src:"medications",cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'gabapentin'"},
  {n:"tfidf_morphine",     desc:"TF-IDF weight for morphine (strong opioid analgesic)",                  src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'morphine'"},
  {n:"tfidf_tramadol",     desc:"TF-IDF weight for tramadol (moderate opioid-like pain reliever)",       src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'tramadol'"},
  {n:"tfidf_oxycodone",    desc:"TF-IDF weight for oxycodone (OxyContin — opioid)",                      src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'oxycodone'"},
  {n:"tfidf_hydrocodone",  desc:"TF-IDF weight for hydrocodone (Vicodin — opioid)",                      src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'hydrocodone'"},
  {n:"tfidf_meperidine",   desc:"TF-IDF weight for meperidine (Demerol — opioid, often post-surgery)",   src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'meperidine'"},
  {n:"tfidf_fentanyl",     desc:"TF-IDF weight for fentanyl (high-potency synthetic opioid)",             src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'fentanyl'"},
  {n:"tfidf_pregabalin",   desc:"TF-IDF weight for pregabalin (Lyrica — gabapentinoid for nerve pain)",  src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'pregabalin'"},
  {n:"tfidf_naproxen",     desc:"TF-IDF weight for naproxen (Aleve — longer-acting NSAID)",              src:"medications", cols:"DESCRIPTION, PATIENT", calc:"TF-IDF for 'naproxen'"},
  // ── LDA Topics ──────────────────────────────────────────────────────────
  {group:"LDA Topic Model (latent medication themes)"},
  {n:"topic_1", desc:"Patient's affinity for latent medication topic 1 (e.g. cardiovascular drugs)",       src:"medications", cols:"DESCRIPTION, PATIENT", calc:"LDA(n_topics=5) on all medication notes; per-patient document-topic proportion, topic 1"},
  {n:"topic_2", desc:"Patient's affinity for latent medication topic 2 (e.g. pain management regimens)",   src:"medications", cols:"DESCRIPTION, PATIENT", calc:"LDA document-topic proportion, topic 2"},
  {n:"topic_3", desc:"Patient's affinity for latent medication topic 3 (e.g. mental health / psych meds)", src:"medications", cols:"DESCRIPTION, PATIENT", calc:"LDA document-topic proportion, topic 3"},
  {n:"topic_4", desc:"Patient's affinity for latent medication topic 4 (e.g. antibiotics / infection)",    src:"medications", cols:"DESCRIPTION, PATIENT", calc:"LDA document-topic proportion, topic 4"},
  {n:"topic_5", desc:"Patient's affinity for latent medication topic 5 (e.g. metabolic / diabetes meds)",  src:"medications", cols:"DESCRIPTION, PATIENT", calc:"LDA document-topic proportion, topic 5"},
  // ── PCA & UMAP ──────────────────────────────────────────────────────────
  {group:"Dimensionality Reduction (PCA & UMAP of TF-IDF space)"},
  {n:"pca_1",  desc:"First principal component of the 12-column TF-IDF matrix",           src:"medications", cols:"tfidf_* (all 12)",  calc:"PCA(n_components=2) fit on TF-IDF features; component 1 — max variance direction"},
  {n:"pca_2",  desc:"Second principal component — orthogonal to pca_1",                   src:"medications", cols:"tfidf_* (all 12)",  calc:"PCA component 2 — second highest variance direction"},
  {n:"umap_1", desc:"First UMAP dimension of medication embedding",                        src:"medications", cols:"tfidf_* (all 12)",  calc:"UMAP(n_components=2, n_neighbors=15) on TF-IDF; dim 1 — local neighborhood structure"},
  {n:"umap_2", desc:"Second UMAP dimension of medication embedding",                       src:"medications", cols:"tfidf_* (all 12)",  calc:"UMAP dim 2 — second non-linear structure in medication space"},
];

const SRC_CLASS = {
  patients:"src-patients", conditions:"src-conditions", encounters:"src-encounters",
  medications:"src-medications", procedures:"src-procedures", observations:"src-observations"
};

function buildFeatTable() {
  const tbody = document.getElementById('feat-tbody');
  if (!tbody) return;
  let rowNum = 0;
  tbody.innerHTML = FEATURES.map(f => {
    if (f.group) {
      return `<tr class="feat-group-row"><td colspan="5">${f.group}</td></tr>`;
    }
    rowNum++;
    const srcLabel = f.src + '.csv';
    const colPills = f.cols.split(',').map(c =>
      `<code style="font-size:.63rem;background:#f0e8d8;padding:1px 5px;border-radius:3px;white-space:nowrap;color:#5c3d22">${c.trim()}</code>`
    ).join(' ');
    return `<tr>
      <td>${rowNum}</td>
      <td><span class="feat-name">${f.n}</span></td>
      <td style="font-size:.72rem;color:#4b3320">${f.desc}</td>
      <td><span class="src-badge ${SRC_CLASS[f.src]}">${srcLabel}</span></td>
      <td style="line-height:1.8">${colPills}</td>
      <td style="font-size:.7rem;color:#6b5440">${f.calc}</td>
    </tr>`;
  }).join('');
}

// ── Init ──────────────────────────────────────────────────────────────────────
[initStats, buildRawTabs, buildFeatTable, renderModelsGrid].forEach(fn => {
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
