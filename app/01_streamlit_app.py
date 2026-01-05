import pandas as pd
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NYC Rent Prediction (NTA)", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "rental_report_final.csv"
MODEL_PATH = ROOT / "models" / "03_final_model.joblib"
st.write("DATA_PATH:", str(DATA_PATH))
st.write("Exists?", DATA_PATH.exists())
st.title("NYC Neighborhood Rent Prediction (NTA)")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

# Dropdown
options = df[["nta2020", "ntaname"]].copy()
options["label"] = options["nta2020"] + " — " + options["ntaname"]
selected = st.selectbox("Select a neighborhood (NTA):",
                        options["label"].tolist())

row = options[options["label"] == selected].iloc[0]
nta_code = row["nta2020"]
nta_name = row["ntaname"]

st.subheader(f"{nta_name} ({nta_code})")

# Build features like training did
exclude = ["nta2020", "ntaname", "median_monthly_rent",
           "avg_monthly_rent", "rent_num_properties", "log_median_rent"]
feature_cols = [c for c in df.columns if c not in exclude]
X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
all_nan_cols = X_all.columns[X_all.isna().all()].tolist()
if all_nan_cols:
    X_all = X_all.drop(columns=all_nan_cols)

X_row = X_all[df["nta2020"] == nta_code]
pred_log = model.predict(X_row)[0]
pred = float(np.exp(pred_log))

actual = df.loc[df["nta2020"] == nta_code, "median_monthly_rent"].values[0]
source = "actual" if pd.notna(actual) else "predicted"

col1, col2, col3 = st.columns(3)
col1.metric("Predicted median rent", f"${pred:,.0f}")
col2.metric("Actual median rent (if available)",
            "-" if pd.isna(actual) else f"${float(actual):,.0f}")
col3.metric("Rent label source", source)

st.divider()

st.caption("This demo uses the trained model artifact saved in /models and the full feature table exported from SQL.")
