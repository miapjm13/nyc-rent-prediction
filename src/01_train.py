import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import joblib
import matplotlib.pyplot as plt

from config import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_FIGURES, REPORTS_METRICS,
    RAW_TARGET_COLS, TARGET_LOG, FINAL_MODEL_PATH, METRICS_JSON_PATH,
    FIG_ACTUAL_VS_PRED, FIG_FEATURE_IMPORTANCE
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dirs():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)


def load_train_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    df = df[df["median_monthly_rent"].notna()].copy()
    df[TARGET_LOG] = np.log(df["median_monthly_rent"])
    return df


def make_X_y(df: pd.DataFrame):
    exclude = ["nta2020", "ntaname", *RAW_TARGET_COLS, TARGET_LOG]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN columns
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    feature_cols = X.columns.tolist()
    y = df[TARGET_LOG]
    return X, y, feature_cols


def cv_eval(model, X, y, name: str, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses, maes = [], []
    for tr, va in cv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmses.append(rmse(y_va, pred))
        maes.append(float(mean_absolute_error(y_va, pred)))

    out = {
        "name": name,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
    }
    return out


def plot_actual_vs_pred(y_test, y_pred, title, out_path: Path):
    y_true_d = np.exp(y_test)
    y_pred_d = np.exp(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_d, y_pred_d, alpha=0.7)
    mn = min(y_true_d.min(), y_pred_d.min())
    mx = max(y_true_d.max(), y_pred_d.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual Median Rent ($)")
    plt.ylabel("Predicted Median Rent ($)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_importance(importances: pd.Series, title: str, out_path: Path):
    plt.figure(figsize=(8, 6))
    importances.head(15).sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()

    df = load_train_data()
    X, y, feature_cols = make_X_y(df)

    # Models (same spirit as your notebook)
    rf_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=800,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ])

    hgb_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            random_state=42,
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.0
        ))
    ])

    rf_cv = cv_eval(rf_pipe, X, y, "RandomForest")
    hgb_cv = cv_eval(hgb_pipe, X, y, "HistGB")

    # choose by CV RMSE mean
    final_model = hgb_pipe if hgb_cv["rmse_mean"] <= rf_cv["rmse_mean"] else rf_pipe
    final_name = "HistGB" if final_model is hgb_pipe else "RandomForest"

    # held-out test split for reporting + plots
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    test_rmse_log = rmse(y_test, y_pred)
    test_mae_log = float(mean_absolute_error(y_test, y_pred))
    test_mae_dollars = float(np.mean(np.abs(np.exp(y_test) - np.exp(y_pred))))

    # Save model
    joblib.dump(final_model, FINAL_MODEL_PATH)

    # Save plot
    plot_actual_vs_pred(
        y_test, y_pred, f"Actual vs Predicted ({final_name})", FIG_ACTUAL_VS_PRED)

    # Importance plot (permutation works for any model)
    perm = permutation_importance(
        final_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
    importances = pd.Series(perm.importances_mean,
                            index=feature_cols).sort_values(ascending=False)
    plot_importance(
        importances, "Top 15 Feature Importances (Permutation)", FIG_FEATURE_IMPORTANCE)

    # Save test predictions
    pred_df = pd.DataFrame({
        "y_true_log": y_test.values,
        "y_pred_log": y_pred,
        "y_true": np.exp(y_test.values),
        "y_pred": np.exp(y_pred)
    })
    pred_df.to_csv(DATA_PROCESSED / "03_test_predictions.csv", index=False)

    # Save metrics JSON
    metrics = {
        "final_model": final_name,
        "rf_cv": rf_cv,
        "hgb_cv": hgb_cv,
        "test_rmse_log": test_rmse_log,
        "test_mae_log": test_mae_log,
        "test_mae_dollars": test_mae_dollars,
        "n_rows_train": int(len(df)),
        "n_features": int(len(feature_cols)),
        "top_features": importances.head(10).to_dict()
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2))

    print("✅ Training complete")
    print("Model:", FINAL_MODEL_PATH)
    print("Metrics:", METRICS_JSON_PATH)
    print("Figures:", REPORTS_FIGURES)


if __name__ == "__main__":
    main()
