import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧")
st.title("💧 Smart Water Quality Analyzer")
st.write("Upload your water test results to assess safety based on WHO guidelines and AI prediction.")

# 🧠 Educational Notes
with st.expander("ℹ️ WHO Guidelines & Water Parameters Explained"):
    st.markdown("""
    **✅ Health-Based Parameters (WHO)**  
    - **Coliforms**: 0 CFU/100 mL — presence indicates possible fecal contamination  
    - **pH**: 6.5–8.5 — outside this range may cause corrosion or scaling  
    - **TDS**: ≤ 1000 mg/L — affects taste and acceptability  
    - **EC**: ≤ 1400 µS/cm — correlates with salinity and minerals  

    **🧱 Aesthetic / Operational Indicators**  
    - **Hardness** (as CaCO₃):  
        - 0–60 → 💧 Soft  
        - 61–120 → 🧂 Moderately Hard  
        - 121–180 → 🪨 Hard  
        - >180 → ⚠️ Very Hard  
    - **DO (Dissolved Oxygen)**:  
        - >6 mg/L → ✅ Good  
        - <6 mg/L → ⚠️ Low (possible stagnation or pollution)
    """)

# --- File Upload ---
phys_file = st.file_uploader("Upload Physical Parameter CSV", type=["csv"])
bact_file = st.file_uploader("Upload Bacterial Test CSV", type=["csv"])

if phys_file and bact_file:
    phys_df = pd.read_csv(phys_file)
    bact_df = pd.read_csv(bact_file)
    st.success("✅ Files uploaded and recognized!")

    # Preprocess EC column
    try:
        phys_df[['EC_val', 'Temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
    except Exception as e:
        st.error("⚠️ Error splitting 'EC' into EC_val and Temp — please check the format (e.g. '1240/25.5')")

    # Merge & Clean
    df = pd.merge(phys_df, bact_df, on="Sample", how="inner")
    df.columns = df.columns.str.strip()  # Remove whitespace

    # WHO Checks
    df["pH_Status"] = df["pH"].apply(lambda x: "✅ OK" if 6.5 <= x <= 8.5 else "⚠️ Out of Range")
    df["TDS_Status"] = df["TDS"].apply(lambda x: "✅ OK" if x <= 1000 else "⚠️ High")
    df["EC_Status"] = df["EC_val"].apply(lambda x: "✅ OK" if x <= 1400 else "⚠️ High")

    if "Coliform" in df.columns:
        df["Coliform_Status"] = df["Coliform"].apply(lambda x: "✅ Safe" if x == 0 else "🚨 Unsafe")
    else:
        df["Coliform_Status"] = "⚠️ Missing"
        st.warning("Column 'Coliform' not found — skipping microbial safety check.")

    if "Hardness" in df.columns:
        def classify_hardness(h):
            if h <= 60:
                return "💧 Soft"
            elif h <= 120:
                return "🧂 Moderately Hard"
            elif h <= 180:
                return "🪨 Hard"
            else:
                return "⚠️ Very Hard"
        df["Hardness_Status"] = df["Hardness"].apply(classify_hardness)
    else:
        df["Hardness_Status"] = "⚠️ Missing"

    if "DO" in df.columns:
        df["DO_Status"] = df["DO"].apply(lambda x: "✅ Good" if x >= 6 else "⚠️ Low")
    else:
        df["DO_Status"] = "⚠️ Missing"

    # Load Model & Predict
    try:
        model = load_model("water_quality_ann.h5")
        scaler = joblib.load("scaler.pkl")
        features = df[["EC_val", "Temp", "pH", "TDS"]]
        X_scaled = scaler.transform(features)
        preds = model.predict(X_scaled)
        df["Prediction"] = np.argmax(preds, axis=1)
        df["Interpretation"] = df["Prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})
        st.success("🧠 AI model prediction complete!")
    except Exception as e:
        df["Interpretation"] = "Unavailable"
        st.error(f"❌ Failed to load model or scaler: {e}")

    # Display Results
    st.subheader("📋 Analysis Report")
    st.dataframe(df[[
        "Sample", "pH", "pH_Status", "TDS", "TDS_Status",
        "EC_val", "EC_Status", "DO", "DO_Status",
        "Hardness", "Hardness_Status", "Coliform_Status", "Interpretation"
    ]])

    # 🚨 Boiling Advisory
    if "🚨 Unsafe" in df["Coliform_Status"].values:
        st.error("🚨 Coliform contamination detected!")
        with st.warning("💡 Boiling Water Advisory"):
            st.markdown("""
            One or more samples show microbial contamination.  
            Please **boil water for at least 1 minute at a rolling boil** before any household use like drinking, brushing, or cooking.  
            Special care should be taken for children, elderly, and immunocompromised members.
            """)

    # Summary
    issues = []
    if "⚠️ Out of Range" in df["pH_Status"].values:
        issues.append("pH out of range")
    if "⚠️ High" in df["TDS_Status"].values or "⚠️ High" in df["EC_Status"].values:
        issues.append("High salinity")
    if "⚠️ Very Hard" in df.get("Hardness_Status", []).values:
        issues.append("Very hard water")
    if "⚠️ Low" in df.get("DO_Status", []).values:
        issues.append("Low oxygen levels")

    if not issues and "🚨 Unsafe" not in df["Coliform_Status"].values:
        st.success("✅ All parameters are within safe and acceptable ranges.")
    else:
        st.warning("⚠️ Other potential issues detected: " + ", ".join(issues) if issues else "⚠️")

else:
    st.info("📂 Please upload both Physical and Bacterial CSV files to begin.")




