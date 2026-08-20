"""
Final Project — Improve Chronic Pain Prediction
================================================
Your starting point is a Random Forest baseline that achieves AUC = 0.830
on 30-iteration stratified cross-validation.

Your goal: beat it. Use any technique you want.

GRADING
-------
  Your score is based on the AUC reported by run_evaluation() at the bottom.
  The same 5-fold stratified CV is used for everyone — no cherry-picking splits.
  Include your final printed output in your submission.

WHAT YOU CAN CHANGE
-------------------
  - Feature engineering  (add, remove, transform any features)
  - Model architecture   (any sklearn, xgboost, lightgbm, pytorch model)
  - Training strategy    (class weights, sampling, ensembling, stacking)
  - Hyperparameters      (tune anything)

WHAT YOU CANNOT CHANGE
-----------------------
  - The evaluation harness below the "DO NOT MODIFY" line
  - The dataset (synthea_enriched_features.csv + the 6 raw CSVs)
  - The label column (binary_label)

HOW TO SUBMIT
-------------
  1. Fill in build_features() and build_model() below.
  2. Run this script: python final_project_starter.py
  3. Copy the printed output block into your report.

HOW TO RUN
----------
  python final_project_starter.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
RAW     = ROOT / "data"
DERIVED = ROOT / "data"

LABEL_COL  = "binary_label"
PATIENT_ID = "id"

# ---------------------------------------------------------------------------
# Load base dataset (45 pre-engineered features)
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DERIVED / "synthea_enriched_features.csv")
print(f"  {len(df):,} patients  |  {df[LABEL_COL].mean():.1%} chronic pain positive")

# Raw files available if you want to engineer new features
patients    = pd.read_csv(RAW / "patients.csv")
conditions  = pd.read_csv(RAW / "conditions.csv",   parse_dates=["START", "STOP"])
encounters  = pd.read_csv(RAW / "encounters.csv",   parse_dates=["START", "STOP"])
medications = pd.read_csv(RAW / "medications.csv",  parse_dates=["START", "STOP"])
procedures  = pd.read_csv(RAW / "procedures.csv",   parse_dates=["DATE"])
observations= pd.read_csv(RAW / "observations.csv", parse_dates=["DATE"])

# ---------------------------------------------------------------------------
# Baseline feature set (what the benchmark uses — 24 structured features)
# ---------------------------------------------------------------------------
BASELINE_COLS = [
    "race", "ethnicity", "healthcare_expenses", "healthcare_coverage",
    "age", "is_female", "num_conditions", "avg_condition_duration",
    "num_unique_meds", "num_pain_meds", "num_encounters", "num_procedures",
    "unique_procedures",
    "pain_severity_0_10_verbal_numeric_rating_score_reported",
    "body_height", "body_weight", "body_mass_index",
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate", "qaly", "daly", "qols",
]

# All 45 features (structured + TF-IDF + topics + PCA/UMAP)
ALL_45_COLS = BASELINE_COLS + [
    "tfidf_opioid", "tfidf_acetaminophen", "tfidf_ibuprofen", "tfidf_gabapentin",
    "tfidf_morphine", "tfidf_tramadol", "tfidf_oxycodone", "tfidf_hydrocodone",
    "tfidf_meperidine", "tfidf_fentanyl", "tfidf_pregabalin", "tfidf_naproxen",
    "topic_1", "topic_2", "topic_3", "topic_4", "topic_5",
    "pca_1", "pca_2", "umap_1", "umap_2",
]


# ===========================================================================
#  PART 1 — FEATURE ENGINEERING
#  Add, remove, or transform features here.
#  Return a DataFrame with one row per patient, indexed by patient id.
# ===========================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Start from the 45 existing features and optionally add new ones.
    Returns the feature matrix X as a DataFrame (no label column).

    The baseline uses BASELINE_COLS (24 features). You have access to:
      - df            : the enriched CSV with all 45 pre-built features
      - patients      : raw patient demographics
      - conditions    : all diagnoses with start/stop dates
      - encounters    : all clinical visits
      - medications   : all prescriptions
      - procedures    : all procedures
      - observations  : all lab results and vitals

    Ideas:
      - Use ALL_45_COLS instead of BASELINE_COLS
      - Engineer temporal features (slopes, rates, recency)
      - Add interaction terms (age × num_pain_meds, BMI × systolic_BP)
      - Extract from raw files: condition trajectory, vital trends, etc.
      - Log-transform skewed features (healthcare_expenses)
    """
    # ── TODO: modify this feature set ──────────────────────────────────────
    feature_df = df[BASELINE_COLS].copy()   # <-- change BASELINE_COLS to ALL_45_COLS
                                             #     or add your own engineered features

    # Example: log-transform healthcare expenses (right-skewed)
    # feature_df["log_expenses"] = np.log1p(df["healthcare_expenses"])

    # Example: interaction term
    # feature_df["age_x_pain_meds"] = df["age"] * df["num_pain_meds"]

    # Example: temporal feature from raw data
    # first_enc = encounters.groupby("PATIENT")["START"].min()
    # feature_df["days_active"] = (pd.Timestamp("2026-01-01") - first_enc).dt.days
    # feature_df = feature_df.join(days_active.rename("days_active"), on=PATIENT_ID)

    return feature_df.fillna(0)


# ===========================================================================
#  PART 2 — MODEL
#  Replace or extend the classifier here.
# ===========================================================================
def build_model():
    """
    Return an unfitted sklearn-compatible classifier.
    Must implement fit(X, y) and predict_proba(X).

    The baseline is RandomForest with 500 trees. Try:
      - More trees / deeper trees
      - XGBClassifier, LGBMClassifier
      - StackingClassifier combining multiple base learners
      - Tuned hyperparameters via RandomizedSearchCV
      - class_weight="balanced" (already in baseline)
    """
    # ── TODO: replace or tune this model ───────────────────────────────────
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ===========================================================================
#  DO NOT MODIFY BELOW THIS LINE — Evaluation harness
# ===========================================================================
N_FOLDS = 5
RANDOM_STATE = 42

def run_evaluation():
    print("\n" + "=" * 62)
    print("  FINAL PROJECT EVALUATION")
    print("  5-fold stratified cross-validation, fixed random seed")
    print("=" * 62)

    X = build_features(df)
    y = df[LABEL_COL].values
    feature_names = X.columns.tolist()
    X = X.values

    print(f"\n  Features used: {len(feature_names)}")
    print(f"  Baseline features: {len(BASELINE_COLS)}")
    new_feats = [f for f in feature_names if f not in BASELINE_COLS]
    if new_feats:
        print(f"  New features added: {new_feats}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        fold_results.append({
            "auc":       roc_auc_score(y[val_idx], probs),
            "f1":        f1_score(y[val_idx], preds, zero_division=0),
            "precision": precision_score(y[val_idx], preds, zero_division=0),
            "recall":    recall_score(y[val_idx], preds, zero_division=0),
        })
        print(f"  Fold {fold}  AUC={fold_results[-1]['auc']:.4f}  "
              f"F1={fold_results[-1]['f1']:.4f}  "
              f"P={fold_results[-1]['precision']:.4f}  "
              f"R={fold_results[-1]['recall']:.4f}")

    elapsed = time.time() - t0
    means = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
    stds  = {k: np.std( [r[k] for r in fold_results]) for k in fold_results[0]}

    BASELINE_AUC = 0.830
    delta = means["auc"] - BASELINE_AUC

    print("\n" + "─" * 62)
    print(f"  {'Metric':<12} {'Mean':>8}  {'Std':>8}")
    print("  " + "─" * 30)
    for k in ["auc", "f1", "precision", "recall"]:
        print(f"  {k:<12} {means[k]:>8.4f}  {stds[k]:>8.4f}")
    print("─" * 62)
    print(f"  Baseline AUC      : {BASELINE_AUC:.4f}")
    print(f"  Your AUC          : {means['auc']:.4f}")
    print(f"  Improvement       : {delta:+.4f}")
    print(f"  Train time        : {elapsed:.1f} sec")
    print("─" * 62)
    if delta >= 0.015:
        print("  EXCELLENT — strong improvement over baseline")
    elif delta >= 0.008:
        print("  GOOD — meaningful improvement")
    elif delta >= 0.003:
        print("  MODEST — small but real improvement")
    elif delta >= -0.003:
        print("  NEUTRAL — within noise; try different features or model")
    else:
        print("  BELOW BASELINE — something went wrong; check your features")
    print("=" * 62 + "\n")
    return means["auc"]


if __name__ == "__main__":
    run_evaluation()
