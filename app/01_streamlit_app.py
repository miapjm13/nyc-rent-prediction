import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="NYC Rent Prediction (NTA)", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "processed" / "04_final_nta_rent_predictions.csv"
METRICS_PATH = ROOT / "reports" / "metrics" / "03_metrics.json"
GEOJSON_PATH = ROOT / "data" / "processed" / "nta_boundaries.geojson"

BOROUGH_NAMES = {"BX": "Bronx", "BK": "Brooklyn", "MN": "Manhattan", "QN": "Queens", "SI": "Staten Island"}
BOROUGH_COLORS = {"Bronx": "#C23B22", "Brooklyn": "#F2622E", "Manhattan": "#4F6DF5",
                   "Queens": "#2FA84F", "Staten Island": "#6E7B8B"}
GOLD = "#F2B705"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Public Sans', sans-serif; }
    div[data-testid="stMetricValue"] { font-weight: 900; }
    hr { border-color: rgba(232,234,237,0.15); }
    .borough-tag {
        display: inline-block; padding: 2px 10px; border-radius: 3px;
        font-size: 0.85rem; font-weight: 700; color: #0E1826;
    }
    .predicted-banner {
        background-color: rgba(242,183,5,0.12);
        border-left: 3px solid #F2B705;
        padding: 10px 14px; border-radius: 2px; margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions():
    df = pd.read_csv(PRED_PATH)
    df["borough_name"] = df["borough"].map(BOROUGH_NAMES)
    df["rent_percentile"] = df["pred_median_rent"].rank(pct=True) * 100
    return df


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_nta_geojson():
    with open(GEOJSON_PATH) as f:
        return json.load(f)


df = load_predictions()
metrics = load_metrics()

st.title("NYC Neighborhood Rent Prediction (NTA)")
st.caption(
    "Predicts median monthly rent across NYC's residential neighborhoods using housing quality, "
    "crime, and property market signals — no unit-level data."
)

with st.expander("How to read this"):
    st.markdown(f"""
    - **NTA** (Neighborhood Tabulation Area) — NYC's standard neighborhood boundary system, used by the Census and city agencies.
    - **{len(df)} residential neighborhoods shown** — 44 NTAs representing parks, cemeteries, airports, and similar non-residential land (e.g. Central Park, LaGuardia Airport, Green-Wood Cemetery) were excluded, since they have no real rental market.
    - **Actual vs. predicted** — of these, most have real reported rent data. The rest have no reliable rent statistics, so their number shown is the model's estimate.
    - **80% prediction interval** — the model's likely range for a neighborhood's rent, not just a single guess. Wider ranges mean less confidence.
    - **What the model doesn't know** — apartment size, bedroom count, amenities, floor, exact address, or unit condition. A neighborhood estimate is not a quote for any specific apartment in it.
    - **Model accuracy** — typical error is about 20–25% of actual rent, reflecting the neighborhood-level (not unit-level) data available. Full methodology in the project README.
    """)

st.divider()

# --- Map ---
st.subheader("Explore NYC by neighborhood")
map_metric = st.radio("Shade by:", ["Predicted rent", "Prediction uncertainty"], horizontal=True)

geojson = load_nta_geojson()
map_df = df.copy()

if map_metric == "Predicted rent":
    color_col, color_label, scale = "pred_median_rent", "Predicted rent ($)", [[0, "#16233A"], [1, "#F2B705"]]
else:
    map_df["interval_width"] = map_df["pred_high_median_rent"] - map_df["pred_low_median_rent"]
    color_col, color_label, scale = "interval_width", "Interval width ($)", [[0, "#16233A"], [1, "#F2622E"]]

fig_map = px.choropleth(
    map_df, geojson=geojson, locations="nta2020", featureidkey="properties.NTA2020",
    color=color_col, color_continuous_scale=scale, hover_name="ntaname",
    hover_data={"borough_name": True, "pred_median_rent": ":$,.0f", color_col: False},
    labels={color_col: color_label}
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(
    paper_bgcolor="#0E1826", plot_bgcolor="#0E1826", font_color="#E8EAED",
    font_family="Public Sans", margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig_map, width="stretch")

st.divider()

# --- Budget finder ---
st.subheader("Find neighborhoods in your budget")
st.caption("See which neighborhoods fit your budget, ranked by housing quality and safety — not just price.")

max_budget = st.slider("Max monthly rent:", min_value=1000, max_value=6000, value=2500, step=100)
in_budget = df[df["pred_median_rent"] <= max_budget].copy()

if in_budget.empty:
    st.write("No neighborhoods currently fit that budget in this dataset.")
else:
    in_budget["quality_rank"] = in_budget["pct_rent_impairing"].rank()
    in_budget = in_budget.sort_values("quality_rank")
    show = in_budget[["ntaname", "borough_name", "pred_median_rent",
                       "pct_rent_impairing", "grand_larceny_auto_rate"]].head(15).copy()
    show.columns = ["Neighborhood", "Borough", "Est. Rent", "Rent-Impairing Violations", "Auto Theft Rate"]
    show["Est. Rent"] = show["Est. Rent"].map("${:,.0f}".format)
    show["Rent-Impairing Violations"] = show["Rent-Impairing Violations"].map("{:.1%}".format)
    show["Auto Theft Rate"] = show["Auto Theft Rate"].map("{:.2f}".format)
    st.write(f"**{len(in_budget)} neighborhoods** fit your budget. Top 15 by housing quality:")
    st.dataframe(show, width="stretch", hide_index=True)

st.divider()

# --- Neighborhood lookup ---
st.subheader("Look up a neighborhood")
col_filter, col_select = st.columns([1, 2])
with col_filter:
    borough_filter = st.selectbox("Borough:", ["All boroughs"] + sorted(BOROUGH_NAMES.values()))

filtered = df if borough_filter == "All boroughs" else df[df["borough_name"] == borough_filter]

with col_select:
    options = filtered.copy()
    options["label"] = options["nta2020"] + " — " + options["ntaname"]
    selected = st.selectbox("Neighborhood:", options["label"].tolist())

row = options[options["label"] == selected].iloc[0]
nta_code, nta_name, borough_name = row["nta2020"], row["ntaname"], row["borough_name"]
b_color = BOROUGH_COLORS.get(borough_name, "#6E7B8B")

st.markdown(f"### {nta_name} ({nta_code})")
st.markdown(f'<span class="borough-tag" style="background-color:{b_color};">{borough_name}</span>',
            unsafe_allow_html=True)

pred, pred_low, pred_high = row["pred_median_rent"], row["pred_low_median_rent"], row["pred_high_median_rent"]
actual = row["median_monthly_rent"]

if pd.isna(actual):
    st.markdown(
        '<div class="predicted-banner">⚠️ No reported rent data exists for this neighborhood. '
        'The figure below is a <b>model estimate</b>, not an observed value.</div>',
        unsafe_allow_html=True
    )

col1, col2, col3 = st.columns(3)
col1.metric("Predicted median rent", f"${pred:,.0f}")
col2.metric("Actual median rent", "—" if pd.isna(actual) else f"${float(actual):,.0f}")
col3.metric("80% interval width", f"${pred_high - pred_low:,.0f}")

st.caption(f"Likely range: ${pred_low:,.0f} – ${pred_high:,.0f}")
st.write(f"More expensive than **{row['rent_percentile']:.0f}%** of NYC neighborhoods.")

fr = row  # underlying signals already present on this row
st.markdown("**Underlying signals for this neighborhood:**")
fc1, fc2, fc3 = st.columns(3)
fc1.metric("Rent-impairing violations", f"{fr['pct_rent_impairing']:.1%}" if pd.notna(fr.get("pct_rent_impairing")) else "—")
fc2.metric("Auto theft rate (per 1k units)", f"{fr['grand_larceny_auto_rate']:.2f}" if pd.notna(fr.get("grand_larceny_auto_rate")) else "—")
fc3.metric("Avg. market value", f"${fr['avg_market_value']:,.0f}" if pd.notna(fr.get("avg_market_value")) else "—")

st.divider()

# --- Citywide table ---
st.subheader("Browse all neighborhoods")
table_borough = st.selectbox("Filter by borough:", ["All boroughs"] + sorted(BOROUGH_NAMES.values()), key="table_filter")
table_df = df if table_borough == "All boroughs" else df[df["borough_name"] == table_borough]

display_df = table_df[["nta2020", "ntaname", "borough_name", "pred_median_rent", "median_monthly_rent"]].copy()
display_df.columns = ["NTA Code", "Neighborhood", "Borough", "Predicted Rent", "Actual Rent"]
display_df = display_df.sort_values("Predicted Rent", ascending=False)
st.dataframe(display_df, width="stretch", hide_index=True)

st.download_button(
    "Download full predictions (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="nyc_nta_rent_predictions.csv",
    mime="text/csv"
)

st.divider()

# --- Model performance (tucked away, not headline) ---
with st.expander("About this model"):
    st.write(
        f"Model: {metrics['final_model']}. Typical error: ${metrics['test_mae_dollars']:,.0f} "
        f"(about 20–25% of median rent). Full methodology, feature engineering, and model "
        f"comparison are documented in the project README."
    )

    observed = df[df["median_monthly_rent"].notna()].copy()
    observed["abs_error"] = (observed["median_monthly_rent"] - observed["pred_median_rent"]).abs()
    worst = observed.sort_values("abs_error", ascending=False).head(10)
    worst_show = worst[["ntaname", "borough_name", "median_monthly_rent", "pred_median_rent", "abs_error"]].copy()
    worst_show.columns = ["Neighborhood", "Borough", "Actual", "Predicted", "Error"]
    for c in ["Actual", "Predicted", "Error"]:
        worst_show[c] = worst_show[c].map("${:,.0f}".format)

    st.markdown("**Where the model struggles most** (largest errors among neighborhoods with observed rent):")
    st.dataframe(worst_show, width="stretch", hide_index=True)
    st.caption(
        "These are typically high-rent or otherwise atypical neighborhoods that are "
        "underrepresented in the training data — a known limitation, not a data error."
    )

st.divider()
st.caption("This demo uses precomputed predictions from the trained model pipeline "
           "(see src/01_train.py, src/02_predict.py). Methodology and limitations are "
           "documented in the project README.")