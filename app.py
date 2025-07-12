import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧")
st.title("💧 Smart Water Quality Analyzer")
st.write("Upload your water test results to check safety based on WHO guidelines and AI prediction.")

# --- Upload CSVs ---
phys_file = st.file_uploader("Upload Physical Parameter CSV", type=["csv"])
bact_file = st.file_uploader("Upload Bacterial Test CSV", type=["csv"])

if phys_file and bact_file:
    phys_df = pd.read_csv(phys_file)
    bact_df = pd.read_csv(bact_file)
    st.success("✅ Files uploaded!")

    # --- Preprocess Physical Data ---
    try:
        phys_df[['EC_val', 'Temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
    except:
        st.error("⚠️ Failed to split 'EC' column into EC_val and Temp. Please check format.")

    # --- Merge Data ---
    df = pd.merge(phys_df, bact_df, on="Sample", how="inner")

    # --- Clean column names just in case ---
    df.columns = df.columns.str.strip()

    # --- WHO Checks ---
    df["pH_Status"] = df["pH"].apply(lambda x: "✅ OK" if 6.5 <= x <= 8.5 else "⚠️ Out of Range")
    df["TDS_Status"] = df["TDS"].apply(lambda x: "✅ OK" if x <= 1000 else "⚠️ High")
    df["EC_Status"] = df["EC_val"].apply(lambda x: "✅ OK" if x <= 1400 else "⚠️ High")

    # Handle missing coliform column gracefully
    if "Coliform" in df.columns:
        df["Coliform_Status"] = df["Coliform"].apply(lambda x: "✅ Safe" if x == 0 else "🚨 Unsafe")
    else:
        df["Coliform_Status"] = "⚠️ Missing"
        st.warning("Column 'Coliform' not found. Skipping microbial safety check.")

    # --- Load Model & Scaler ---
    try:
        model = load_model("water_quality_ann.h5")
        scaler = joblib.load("scaler.pkl")
        features = df[["EC_val", "Temp", "pH", "TDS"]]
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        df["Prediction"] = np.argmax(prediction, axis=1)
        df["Interpretation"] = df["Prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})
        st.success("🎯 AI prediction complete!")
    except Exception as e:
        st.error(f"❌ Model or scaler failed to load: {e}")
        df["Interpretation"] = "Unavailable"

    # --- Display Results ---
    st.subheader("📋 Full Analysis")
    st.dataframe(df[[
        "Sample", "pH", "pH_Status", "TDS", "TDS_Status", 
        "EC_val", "EC_Status", "Coliform_Status", "Interpretation"
    ]])

    # --- Safety Summary ---
    if "🚨 Unsafe" in df["Coliform_Status"].values:
        st.error("⚠️ Microbial contamination detected in one or more samples.")
    elif "⚠️ Out of Range" in df["pH_Status"].values or "⚠️ High" in df["TDS_Status"].values or "⚠️ High" in df["EC_Status"].values:
        st.warning("⚠️ Some physico-chemical values exceed WHO guidelines.")
    else:
        st.success("✅ All parameters within WHO safety thresholds.")
else:
    st.info("📂 Please upload both Physical and Bacterial CSV files to begin.")









