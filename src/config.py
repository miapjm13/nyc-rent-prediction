import numpy as np
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / \
    "rental_report_final.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS_METRICS = PROJECT_ROOT / "reports" / "metrics"

# Columns
ID_COLS = ["nta2020", "ntaname"]
CRIME_COLS = [
    "murder", "rape", "robbery", "felony_assault",
    "burglary", "grand_larceny", "grand_larceny_auto"
]
RAW_TARGET_COLS = ["median_monthly_rent",
                   "avg_monthly_rent", "rent_num_properties"]
# Columns dropped due to data quality issues
DROPPED_FEATURES = ["school_count", "avg_ats"]
TARGET_LOG = "log_median_rent"
BOROUGH_CODES = ["BX", "BK", "MN", "QN", "SI"]

# NTAs that are parks, cemeteries, airports, or other non-residential land —
# excluded because they have no meaningful residential housing stock and
# nobody rents an apartment there. Identified via anomalous per-unit crime
# rates (near-zero avg_unitstotal inflates rate denominators).
NON_RESIDENTIAL_NTAS = [
    "QN8191", "BK1892", "SI9593", "SI0291", "BX1271", "BX0492", "QN8491",
    "BX1071", "QN0271", "QN0871", "MN1191", "QN1191", "MN0191", "MN1291",
    "QN0791", "BK0891", "BK1391", "BX2891", "BK5693", "QN0571", "QN0151",
    "QN8081", "BK5591", "BK0261", "BK0571", "BK1893", "SI9561", "BX0391",
    "QN1491", "MN1292", "QN1371", "BX0491", "BX0991", "QN8291", "BX0291",
    "BK5691", "QN0572", "BK1891", "BK0771", "BX1091", "QN0574", "BK1091",
    "BX2791", "BX2691"
]

def add_borough_dummies(df):
    df["borough"] = df["nta2020"].str[:2]
    df["borough"] = pd.Categorical(df["borough"], categories=BOROUGH_CODES)
    dummies = pd.get_dummies(df["borough"], prefix="borough", drop_first=True)
    return pd.concat([df, dummies], axis=1)


# Output files
FINAL_MODEL_PATH = MODELS_DIR / "03_final_model.joblib"
METRICS_JSON_PATH = REPORTS_METRICS / "03_metrics.json"
TEST_PRED_PATH = DATA_PROCESSED / "03_test_predictions.csv"
FINAL_PRED_PATH = DATA_PROCESSED / "04_final_nta_rent_predictions.csv"

FIG_ACTUAL_VS_PRED = REPORTS_FIGURES / "03_actual_vs_pred.png"
FIG_FEATURE_IMPORTANCE = REPORTS_FIGURES / "03_feature_importance.png"

LOWER_MODEL_PATH = MODELS_DIR / "03_final_model_q10.joblib"
UPPER_MODEL_PATH = MODELS_DIR / "03_final_model_q90.joblib"


def add_crime_rates(df, per=1000):
    total_units = df["avg_unitstotal"] * df["num_bbls"]
    total_units = total_units.replace(0, np.nan)  # avoid divide-by-zero
    for c in CRIME_COLS:
        df[f"{c}_rate"] = df[c] / (total_units / per)
    return df
