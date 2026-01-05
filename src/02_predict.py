import numpy as np
import pandas as pd
import joblib

from config import (
    DATA_RAW, FINAL_MODEL_PATH, FINAL_PRED_PATH,
    RAW_TARGET_COLS, TARGET_LOG
)


def main():
    df = pd.read_csv(DATA_RAW)

    exclude = ["nta2020", "ntaname", *RAW_TARGET_COLS, TARGET_LOG]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN columns to match training behavior
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    model = joblib.load(FINAL_MODEL_PATH)

    df["pred_log_median_rent"] = model.predict(X)
    df["pred_median_rent"] = np.exp(df["pred_log_median_rent"])

    final_output = df[[
        "nta2020", "ntaname", "median_monthly_rent", "pred_median_rent"
    ]].copy()

    final_output["rent_source"] = np.where(
        final_output["median_monthly_rent"].notna(),
        "actual",
        "predicted"
    )

    final_output.to_csv(FINAL_PRED_PATH, index=False)

    print("✅ Predictions saved:", FINAL_PRED_PATH)
    print(final_output["rent_source"].value_counts())


if __name__ == "__main__":
    main()
