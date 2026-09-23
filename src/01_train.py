import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import joblib
import matplotlib.pyplot as plt

from config import (
    CRIME_COLS, DATA_RAW, DATA_PROCESSED, DROPPED_FEATURES, LOWER_MODEL_PATH,
    MODELS_DIR, REPORTS_FIGURES, REPORTS_METRICS, RAW_TARGET_COLS, TARGET_LOG,
    FINAL_MODEL_PATH, METRICS_JSON_PATH, FIG_ACTUAL_VS_PRED, FIG_FEATURE_IMPORTANCE,
    UPPER_MODEL_PATH, add_borough_dummies, add_crime_rates
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
    df = add_crime_rates(df)
    df = add_borough_dummies(df)
    df[TARGET_LOG] = np.log(df["median_monthly_rent"])
    return df


def make_X_y(df: pd.DataFrame):
    exclude = ["nta2020", "ntaname", "borough", *RAW_TARGET_COLS,
               TARGET_LOG, *DROPPED_FEATURES, *CRIME_COLS]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    feature_cols = X.columns.tolist()
    y = df[TARGET_LOG]
    return X, y, feature_cols


def cv_evaluate(model, X, y, name, cv):
    rmses, maes = [], []
    for tr, va in cv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmses.append(rmse(y_va, pred))
        maes.append(float(mean_absolute_error(y_va, pred)))
    print(f"{name} CV RMSE (log): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"{name} CV MAE  (log): {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    return float(np.mean(rmses)), float(np.mean(maes))


def tune(pipe, param_dist, X, y, cv, name):
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=30, cv=cv,
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    print(f"Best {name} params:", search.best_params_)
    print(f"Best {name} CV RMSE (log):", -search.best_score_)
    return search.best_estimator_


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

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_pipe = tune(
        Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        {"model__n_estimators": randint(200, 1000), "model__max_depth": randint(3, 30),
         "model__min_samples_leaf": randint(1, 10), "model__max_features": uniform(0.3, 0.7)},
        X, y, cv, "RF"
    )
    hgb_pipe = tune(
        Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("model", HistGradientBoostingRegressor(random_state=42))]),
        {"model__learning_rate": uniform(0.01, 0.19), "model__max_depth": randint(2, 10),
         "model__max_leaf_nodes": randint(15, 63), "model__min_samples_leaf": randint(5, 30),
         "model__l2_regularization": uniform(0.0, 1.0)},
        X, y, cv, "HGB"
    )
    lgbm_pipe = tune(
        Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("model", LGBMRegressor(random_state=42, verbosity=-1))]),
        {"model__learning_rate": uniform(0.01, 0.19), "model__max_depth": randint(2, 10),
         "model__num_leaves": randint(15, 63), "model__min_child_samples": randint(5, 30),
         "model__reg_lambda": uniform(0.0, 1.0)},
        X, y, cv, "LightGBM"
    )
    xgb_pipe = tune(
        Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("model", XGBRegressor(random_state=42))]),
        {"model__learning_rate": uniform(0.01, 0.19), "model__max_depth": randint(2, 10),
         "model__min_child_weight": randint(1, 10), "model__subsample": uniform(0.6, 0.4),
         "model__reg_lambda": uniform(0.0, 2.0)},
        X, y, cv, "XGBoost"
    )

    rf_cv_rmse, rf_cv_mae = cv_evaluate(rf_pipe, X, y, "RandomForest", cv)
    hgb_cv_rmse, hgb_cv_mae = cv_evaluate(hgb_pipe, X, y, "HistGB", cv)
    lgbm_cv_rmse, lgbm_cv_mae = cv_evaluate(lgbm_pipe, X, y, "LightGBM", cv)
    xgb_cv_rmse, xgb_cv_mae = cv_evaluate(xgb_pipe, X, y, "XGBoost", cv)

    candidates = {
        "RandomForest": (rf_pipe, rf_cv_rmse),
        "HistGB": (hgb_pipe, hgb_cv_rmse),
        "LightGBM": (lgbm_pipe, lgbm_cv_rmse),
        "XGBoost": (xgb_pipe, xgb_cv_rmse),
    }
    final_name = min(candidates, key=lambda k: candidates[k][1])
    final_model = candidates[final_name][0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    test_rmse_log = rmse(y_test, y_pred)
    test_mae_log = float(mean_absolute_error(y_test, y_pred))
    test_mae_dollars = float(np.mean(np.abs(np.exp(y_test) - np.exp(y_pred))))

    print("Final model:", final_name)
    print("Test RMSE (log):", test_rmse_log)
    print("Test MAE (log):", test_mae_log)
    print("Test MAE ($):", test_mae_dollars)

    joblib.dump(final_model, FINAL_MODEL_PATH)

    # Quantile models for prediction intervals
    def make_quantile_pipe(quantile):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, random_state=42,
                learning_rate=0.05, max_depth=6, max_leaf_nodes=31, min_samples_leaf=10
            ))
        ])

    lower_pipe = make_quantile_pipe(0.10)
    upper_pipe = make_quantile_pipe(0.90)
    lower_pipe.fit(X_train, y_train)
    upper_pipe.fit(X_train, y_train)
    joblib.dump(lower_pipe, LOWER_MODEL_PATH)
    joblib.dump(upper_pipe, UPPER_MODEL_PATH)

    plot_actual_vs_pred(y_test, y_pred, f"Actual vs Predicted ({final_name})", FIG_ACTUAL_VS_PRED)

    if final_name == "RandomForest":
        model = final_model.named_steps["model"]
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    else:
        perm = permutation_importance(final_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
        importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

    plot_importance(importances, f"Top 15 Feature Importances ({final_name})", FIG_FEATURE_IMPORTANCE)

    pred_df = pd.DataFrame({
        "y_true_log": y_test.values, "y_pred_log": y_pred,
        "y_true": np.exp(y_test.values), "y_pred": np.exp(y_pred)
    })
    pred_df.to_csv(DATA_PROCESSED / "03_test_predictions.csv", index=False)

    metrics = {
        "final_model": final_name,
        "rf_cv_rmse_log": rf_cv_rmse,
        "hgb_cv_rmse_log": hgb_cv_rmse,
        "lgbm_cv_rmse_log": lgbm_cv_rmse,
        "xgb_cv_rmse_log": xgb_cv_rmse,
        "test_rmse_log": test_rmse_log,
        "test_mae_log": test_mae_log,
        "test_mae_dollars": test_mae_dollars,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "top_features": importances.head(10).to_dict()
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2))

    print("Training complete")
    print("Model:", FINAL_MODEL_PATH)
    print("Metrics:", METRICS_JSON_PATH)
    print("Figures:", REPORTS_FIGURES)


if __name__ == "__main__":
    main()