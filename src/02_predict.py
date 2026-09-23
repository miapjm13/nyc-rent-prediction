import numpy as np
import pandas as pd
import joblib

from config import (
    CRIME_COLS, DATA_PROCESSED, DATA_RAW, DROPPED_FEATURES, FINAL_MODEL_PATH, FINAL_PRED_PATH, LOWER_MODEL_PATH, NON_RESIDENTIAL_NTAS,
    RAW_TARGET_COLS, TARGET_LOG, UPPER_MODEL_PATH, add_crime_rates, add_borough_dummies
)


def main():
    df = pd.read_csv(DATA_RAW)
    df = add_crime_rates(df)
    df = add_borough_dummies(df)

    exclude = ["nta2020", "ntaname", *RAW_TARGET_COLS,
               TARGET_LOG, *DROPPED_FEATURES, *CRIME_COLS]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    model = joblib.load(FINAL_MODEL_PATH)

    df["pred_log_median_rent"] = model.predict(X)
    df["pred_median_rent"] = np.exp(df["pred_log_median_rent"])
    lower_model = joblib.load(LOWER_MODEL_PATH)
    upper_model = joblib.load(UPPER_MODEL_PATH)

    df["pred_low_median_rent"] = np.exp(lower_model.predict(X))
    df["pred_high_median_rent"] = np.exp(upper_model.predict(X))

    df = df[~df["nta2020"].isin(NON_RESIDENTIAL_NTAS)].copy()

    final_output = df[[
        "nta2020", "ntaname", "borough", "median_monthly_rent",
        "pred_median_rent", "pred_low_median_rent", "pred_high_median_rent",
        "pct_rent_impairing", "grand_larceny_auto_rate", "avg_market_value"
    ]].copy()

    final_output["rent_source"] = np.where(
        final_output["median_monthly_rent"].notna(),
        "actual",
        "predicted"
    )

    final_output.to_csv(FINAL_PRED_PATH, index=False)

    print("Predictions saved:", FINAL_PRED_PATH)
    print(final_output["rent_source"].value_counts())


if __name__ == "__main__":
    main()
