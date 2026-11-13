"""Loads the trained models once and exposes small prediction helpers."""

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"

# Names derived from reports/policy_segment_profiles.csv
SEGMENT_INFO = {
    0: ("Young high-risk urban drivers",
        "Average age ~31, high Bonus-Malus (~85), dense areas. Highest claim rate."),
    1: ("Experienced low-density drivers",
        "Average age ~49, low Bonus-Malus (~53), rural / small towns. Lowest claim rate."),
    2: ("Experienced big-city drivers",
        "Average age ~50, low Bonus-Malus, very dense areas (e.g. Ile-de-France). Medium claim rate."),
}


@lru_cache(maxsize=1)
def _frequency():
    return joblib.load(ARTIFACTS / "best_claim_frequency_model.pkl")


@lru_cache(maxsize=1)
def _severity():
    return joblib.load(ARTIFACTS / "best_individual_claim_model_mae.pkl")


@lru_cache(maxsize=1)
def _segmentation():
    return joblib.load(ARTIFACTS / "policy_segmentation_model.pkl")


def risk_level(claims_per_year: float) -> str:
    # ~0.10 claims/year is the portfolio average (baseline_rate in the saved model).
    if claims_per_year < 0.08:
        return "LOW"
    if claims_per_year < 0.15:
        return "MEDIUM"
    return "HIGH"


def predict_frequency(policy: dict) -> dict:
    saved = _frequency()
    row = pd.DataFrame([policy])[saved["feature_columns"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_year = float(saved["model"].predict(saved["preprocessor"].transform(row))[0])
    per_year = max(per_year, 0.0)
    return {
        "expected_claims_per_year": round(per_year, 4),
        "expected_claims_for_period": round(per_year * policy["Exposure"], 4),
        "risk_level": risk_level(per_year),
        "model": saved["model_name"],
    }


def predict_severity(policy: dict) -> dict:
    saved = _severity()
    row = pd.DataFrame([policy])[saved["feature_columns"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amount = float(saved["model"].predict(row)[0])
    return {"predicted_claim_amount": round(max(amount, 0.0), 2),
            "model": saved["model_name"]}


def predict_segment(policy: dict) -> dict:
    saved = _segmentation()
    row = pd.DataFrame([policy])[saved["feature_columns"]].copy()
    if saved.get("density_log_transform"):
        row["Density"] = np.log1p(row["Density"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster = int(saved["model"].predict(saved["preprocessor"].transform(row))[0])
    name, description = SEGMENT_INFO.get(cluster, (f"Segment {cluster}", ""))
    return {"cluster": cluster, "segment_name": name, "description": description}


def recommended_action(level: str, amount: float) -> str:
    if level == "HIGH" or amount > 5000:
        return "MANUAL_REVIEW"
    if level == "MEDIUM":
        return "STANDARD_PROCESSING"
    return "FAST_TRACK"
