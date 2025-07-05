import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧")
st.title("💧 Smart Water Quality Analyzer")
st.write("Upload your water test results to check safety and get predictions based on WHO guidelines and machine learning.")

# Upload physical & bacterial CSV files
phys_file = st.file_uploader("Upload Physical Parameter CSV", type=["csv"])
bact_file = st.file_uploader("Upload Bacterial Test CSV", type=["csv"])

if phys_file and bact_file:
    # Load the files
    phys_df = pd.read_csv(phys_file)
    bact_df = pd.read_csv(bact_file)

    st.success("✅ Files uploaded successfully!")

    # Clean EC column
    phys_df[['EC_val', 'Temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)

    # Merge physical & bacterial data
    df = pd.merge(phys_df, bact_df, on="Sample", how="inner")

    # WHO guideline checks
    df["pH_Status"] = df["pH"].apply(lambda x: "✅ OK" if 6.5 <= x <= 8.5 else "⚠️ Out of Range")
    df["TDS_Status"] = df["TDS"].apply(lambda x: "✅ OK" if x <= 1000 else "⚠️ High")
    df["EC_Status"] = df["EC_val"].apply(lambda x: "✅ OK" if x <= 1400 else "⚠️ High")
    df["Coliform_Status"] = df["Coliform"].apply(lambda x: "✅ Safe" if x == 0 else "🚨 Unsafe")

    # Load model & scaler
    try:
        model = load_model("water_quality_ann.h5")
        scaler = joblib.load("scaler.pkl")

        features = df[['EC_val', 'Temp', 'pH', 'TDS']]
        scaled = scaler.transform(features)

        # Predict and classify
        predictions = model.predict(scaled)
        df["Prediction"] = np.argmax(predictions, axis=1)  # Optional: map 0/1/2 to labels
        df["Interpretation"] = df["Prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})

        st.subheader("📊 Prediction & WHO Compliance Report")
        st.dataframe(df[[
            "Sample", "pH", "pH_Status", "TDS", "TDS_Status",
            "EC_val", "EC_Status", "Coliform", "Coliform_Status",
            "Interpretation"
        ]])

        # Optional: summary
        if "🚨 Unsafe" in df["Coliform_Status"].values:
            st.error("⚠️ One or more samples show microbial contamination.")
        else:
            st.success("✅ All samples meet WHO microbial safety standards.")

    except Exception as e:
        st.error(f"Model loading or prediction failed: {e}")

else:
    st.warning("📂 Please upload both Physical and Bacterial CSV files to continue.")

