# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Enhanced Crop Yield Prediction")
st.caption("IEEE-grade ML model • India Agriculture Dataset")

@st.cache_resource
def load_artifacts():
    model, feature_columns = joblib.load("best_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    median_vals = joblib.load("median_values.pkl")
    agg_stats = joblib.load("agg_stats.pkl")
    yield_mean, yield_std = joblib.load("yield_stats.pkl")
    return model, feature_columns, label_encoders, scaler, median_vals, agg_stats, yield_mean, yield_std

(
    model,
    feature_columns,
    label_encoders,
    scaler,
    median_vals,
    agg_stats,
    yield_mean,
    yield_std
) = load_artifacts()

# ---------------- UI ---------------- #
st.subheader("📥 Input Parameters")

crop = st.selectbox("Crop", label_encoders["Crop"].classes_)
state = st.selectbox("State", label_encoders["State"].classes_)
season = st.selectbox("Season", label_encoders["Season"].classes_)

st.markdown("---")

if st.button("🔮 Predict Yield", use_container_width=True):
    input_dict = {}

    crop_avg = agg_stats["crop_avg_yield"].get(crop, yield_mean)
    state_avg = agg_stats["state_avg_yield"].get(state, yield_mean)
    season_avg = agg_stats["season_avg_yield"].get(season, yield_mean)

    for col in feature_columns:
        if col == "Crop":
            input_dict[col] = [label_encoders["Crop"].transform([crop])[0]]
        elif col == "State":
            input_dict[col] = [label_encoders["State"].transform([state])[0]]
        elif col == "Season":
            input_dict[col] = [label_encoders["Season"].transform([season])[0]]
        elif col == "Crop_Avg_Yield":
            input_dict[col] = [crop_avg]
        elif col == "State_Avg_Yield":
            input_dict[col] = [state_avg]
        elif col == "Season_Avg_Yield":
            input_dict[col] = [season_avg]
        elif col == "Rainfall_per_Fertilizer":
            input_dict[col] = [median_vals["Annual_Rainfall"] / (median_vals["Fertilizer"] + 1)]
        elif col == "Area_Production_Ratio":
            input_dict[col] = [median_vals["Area"] / (median_vals["Production"] + 1)]
        elif col == "Total_Input":
            input_dict[col] = [median_vals["Fertilizer"] + median_vals["Pesticide"]]
        elif col == "Fertilizer_Pesticide_Ratio":
            input_dict[col] = [median_vals["Fertilizer"] / (median_vals["Pesticide"] + 1)]
        else:
            input_dict[col] = [median_vals.get(col, 0)]

    X = pd.DataFrame(input_dict)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    st.success(f"🌱 **Predicted Yield:** `{prediction:.2f}` tons / hectare")

    if prediction > yield_mean + yield_std:
        st.markdown("🟢 **Assessment:** Above average yield expected")
    elif prediction < yield_mean - yield_std:
        st.markdown("🔴 **Assessment:** Below average yield expected")
    else:
        st.markdown("🟡 **Assessment:** Average yield expected")

    st.markdown("---")
    st.caption(f"📊 Historical Avg for {crop}: {crop_avg:.2f} t/ha")
    st.caption(f"📍 Historical Avg in {state}: {state_avg:.2f} t/ha")

