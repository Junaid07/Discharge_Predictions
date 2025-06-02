import streamlit as st
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from sklearn.preprocessing import MinMaxScaler
import os

# === Load the Keras model from local path ===
model_path = "lstm_discharge_model_2to1.h5"  # or .keras if you're using Keras v3 format
model = load_model(model_path, custom_objects={"Orthogonal": Orthogonal})

# === Load scaler from CSV ===
params = pd.read_csv("scaler_params.csv")
scaler = MinMaxScaler()
scaler.min_ = params["min_"].values
scaler.scale_ = params["scale_"].values

# === Load historical averages from Excel ===
averages = pd.read_excel("historical_averages.xlsx")
averages["Day"] = pd.to_numeric(averages["Day"], errors="coerce").fillna(0).astype(int)
averages_indexed = averages.set_index("Day")

# === Streamlit UI ===
st.title("2-Day Discharge Forecast App")
st.markdown("Enter **today's** and **yesterday's** discharge to forecast the **next 2 days**.")

# === User Inputs ===
today = st.number_input("Today's Discharge (Day 0)", min_value=0.0, step=100.0)
yesterday = st.number_input("Yesterday's Discharge (Day -1)", min_value=0.0, step=100.0)

if st.button("Forecast Next 2 Days"):
    if today == 0.0 or yesterday == 0.0:
        st.warning("Please enter both values to forecast.")
    else:
        # === Scale and reshape input ===
        input_scaled = scaler.transform(np.array([yesterday, today]).reshape(-1, 1)).flatten()
        sequence = np.array(input_scaled).reshape(1, 2, 1)

        predictions_scaled = []
        prediction_dates = []
        base_date = datetime.date.today()

        for i in range(2):
            pred_scaled = model.predict(sequence, verbose=0)[0][0]
            predictions_scaled.append(pred_scaled)
            sequence = np.append(sequence.flatten()[1:], pred_scaled).reshape(1, 2, 1)
            prediction_dates.append(base_date + datetime.timedelta(days=i + 1))

        # === Inverse transform predictions ===
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        # === Compute day-of-year and lookup averages ===
        doy = [d.timetuple().tm_yday for d in prediction_dates]
        avg_vals = [averages_indexed["Q"].get(day, np.nan) for day in doy]

        # === Create output dataframe ===
        forecast_df = pd.DataFrame({
            "Date": prediction_dates,
            "Predicted Discharge (cfs)": np.round(predictions, 2),
            "Historical Average (cfs)": [round(val, 2) if not pd.isna(val) else "N/A" for val in avg_vals]
        })

        st.success("Forecast Complete!")
        st.dataframe(forecast_df)
