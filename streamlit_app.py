
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HDB Rental Price Predictor", layout="wide")

MODEL_RMSE_SGD = 508.39
MODEL_FILE = Path("rental_price_model.joblib")
FEATURE_FILE = Path("model_features.joblib")
TOWN_MAPPING_FILE = Path("town_avg_rent_mapping.joblib")
DATA_FILE = Path("RentingOutofFlatsfromJan2021.csv")

FLAT_TYPE_MAP = {
    "1-ROOM": 1,
    "2-ROOM": 2,
    "3-ROOM": 3,
    "4-ROOM": 4,
    "5-ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7,
}
CENTRAL_TOWNS = {"CENTRAL AREA", "BUKIT TIMAH", "QUEENSTOWN", "TOA PAYOH", "BISHAN"}

@st.cache_data
def load_reference_data():
    if not DATA_FILE.exists():
        return pd.DataFrame({"town": sorted(CENTRAL_TOWNS), "flat_type": list(FLAT_TYPE_MAP.keys())})
    df = pd.read_csv(DATA_FILE)
    df["town"] = df["town"].astype(str).str.strip()
    df["flat_type"] = df["flat_type"].astype(str).str.strip()
    return df

@st.cache_resource
def load_model_artifacts():
    missing = [p.name for p in (MODEL_FILE, FEATURE_FILE, TOWN_MAPPING_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_FILE)
    town_avg_rent = joblib.load(TOWN_MAPPING_FILE)
    global_avg = float(np.mean(list(town_avg_rent.values()))) if town_avg_rent else 2500.0
    return model, feature_columns, town_avg_rent, global_avg

def build_feature_row(town, flat_type_label, year, month, feature_columns, town_avg_rent, global_avg):
    chosen_date = pd.Timestamp(year=year, month=month, day=1)
    months_since_2021 = int((chosen_date - pd.Timestamp("2021-01-01")).days / 30)
    row = {
        "flat_type_num": FLAT_TYPE_MAP[flat_type_label],
        "rent_year": int(year),
        "rent_month": int(month),
        "rent_quarter": int(((month - 1) // 3) + 1),
        "months_since_2021": months_since_2021,
        "is_central": int(town in CENTRAL_TOWNS),
        "town_avg_rent": float(town_avg_rent.get(town, global_avg)),
    }
    for col in feature_columns:
        row.setdefault(col, 0)
    town_col = f"town_{town}"
    if town_col in row:
        row[town_col] = 1
    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)

def validate_inputs(year, month):
    current_year = dt.date.today().year
    if year < 2021 or year > current_year + 3:
        return False, f"Year must be between 2021 and {current_year + 3}."
    if month < 1 or month > 12:
        return False, "Month must be between 1 and 12."
    return True, ""

st.title("Singapore HDB Rental Price Predictor")
st.caption("Predict monthly rent using your tuned Random Forest model.")

df_ref = load_reference_data()
towns = sorted(df_ref["town"].dropna().unique().tolist()) if "town" in df_ref.columns else sorted(CENTRAL_TOWNS)
flat_types = sorted([x for x in df_ref["flat_type"].dropna().unique().tolist() if x in FLAT_TYPE_MAP]) if "flat_type" in df_ref.columns else list(FLAT_TYPE_MAP.keys())
if not flat_types:
    flat_types = list(FLAT_TYPE_MAP.keys())

col1, col2 = st.columns([1.2, 0.8], gap="large")
with col1:
    town = st.selectbox("Town", options=towns)
    flat_type = st.selectbox("Flat Type", options=flat_types, index=min(2, len(flat_types) - 1))
    current_year = dt.date.today().year
    year = st.number_input("Target Year", min_value=2021, max_value=current_year + 3, value=current_year)
    month = st.slider("Target Month", min_value=1, max_value=12, value=1, step=1)
    is_valid, msg = validate_inputs(int(year), int(month))
    if not is_valid:
        st.error(msg)
    predict_clicked = st.button("Predict Monthly Rent", type="primary", use_container_width=True)

with col2:
    st.markdown(
        f"- **Model**: Tuned Random Forest\n"
        f"- **Metric**: RMSE\n"
        f"- **Expected error band**: +/- SGD {MODEL_RMSE_SGD:,.0f}"
    )

if predict_clicked:
    if not is_valid:
        st.stop()
    try:
        model, feature_columns, town_avg_rent, global_avg = load_model_artifacts()
        X_input = build_feature_row(town, flat_type, int(year), int(month), feature_columns, town_avg_rent, global_avg)
        pred = float(model.predict(X_input)[0])
        if pred < 0:
            st.error("Prediction failed validation (negative rent).")
            st.stop()
        low = max(0.0, pred - MODEL_RMSE_SGD)
        high = pred + MODEL_RMSE_SGD
        st.success(f"Estimated Monthly Rent: SGD {pred:,.2f}")
        st.info(f"Likely range: SGD {low:,.2f} to SGD {high:,.2f}")
        with st.expander("Show engineered input"):
            st.dataframe(X_input, use_container_width=True)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.caption("Run notebook model-saving cells first.")
    except Exception as exc:
        st.error("Unexpected error while generating prediction.")
        st.exception(exc)
