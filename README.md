# NYC Neighborhood Rent Prediction (NTA-level)

Predict median monthly rent across NYC neighborhoods (NTA 2020 boundaries) using housing stock, maintenance violations, crime, schools, and property valuation signals.

## Why this project
I wanted a real-world, messy-data project that goes beyond notebooks: integrate multiple NYC datasets, engineer neighborhood-level features in SQL, train ML models with cross-validation, and generate predictions for neighborhoods with missing rent labels.

## Data & Features (high level)
All features are aggregated to **NTA (Neighborhood Tabulation Area)**, producing one row per neighborhood.

Sources integrated:
- Housing stock / built environment (PLUTO aggregates)
- Maintenance & rent-impairing violations
- Crime counts by category
- School ATS performance aggregated to NTA
- Property valuation aggregates
- Rent labels aggregated to NTA (median rent)

## Method
- Built a modeling table in Postgres at NTA level (one row per NTA)
- Trained baseline (Ridge) and tree/boosting models
- Selected final model via 5-fold cross-validation
- Generated predictions for NTAs missing rent labels

## Results
- Final model: **HistGradientBoostingRegressor**
- Test MAE: **~$470** (median rent at NTA level)
- Key drivers: housing quality (rent-impairing violations), property crime, market value, violations per building

See plots in `reports/figures/`.

## Reproducible pipeline
1) Put the exported dataset at:
`data/raw/final_modeling_table_with_target.csv`

2) Install deps:
```bash
pip install -r requirements.txt
