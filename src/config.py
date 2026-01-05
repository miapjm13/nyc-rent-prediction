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
RAW_TARGET_COLS = ["median_monthly_rent",
                   "avg_monthly_rent", "rent_num_properties"]
TARGET_LOG = "log_median_rent"

# Output files
FINAL_MODEL_PATH = MODELS_DIR / "03_final_model.joblib"
METRICS_JSON_PATH = REPORTS_METRICS / "03_metrics.json"
TEST_PRED_PATH = DATA_PROCESSED / "03_test_predictions.csv"
FINAL_PRED_PATH = DATA_PROCESSED / "04_final_nta_rent_predictions.csv"

FIG_ACTUAL_VS_PRED = REPORTS_FIGURES / "03_actual_vs_pred.png"
FIG_FEATURE_IMPORTANCE = REPORTS_FIGURES / "03_feature_importance.png"
