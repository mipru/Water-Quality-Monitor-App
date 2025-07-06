import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧")
st.title("💧 Water Safety Dashboard")
st.write("Upload your test results to evaluate safety using WHO guidelines and AI predictions.")

# 🧠 Educational Section
with st.expander("ℹ️ WHO Guidelines & Key Indicators"):
    st.markdown("""
    - **Coliforms**: 0 CFU/100 mL → 🚨 Unsafe if > 0  
    - **pH**: 6.5–8.5  
    - **TDS**: ≤ 1000 mg/L  
    - **EC**: ≤ 1400 µS/cm  
    - **Hardness**: >500 mg/L may cause scaling  
    - **DO (Dissolved Oxygen)**: >6 mg/L preferred for freshness  
    """)

# Upload CSVs
phys_file = st.file_uploader("Upload Physical Parameter CSV", type=["csv"])
bact_file = st.file_uploader("Upload Bacterial Test CSV", type=["csv"])

if phys_file and bact_file:
    phys_df = pd.read_csv(phys_file)
    bact_df = pd.read_csv(bact_file)
    st.success("✅ Files uploaded!")

    # Process EC field
    try:
        phys_df[['EC_val', 'Temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
    except Exception as e:
        st.error("⚠️ Unable to split 'EC' into EC_val and Temp. Ensure format like '1200/24.5'.")

    df = pd.merge(phys_df, bact_df, on="Sample", how="inner")
    df.columns = df.columns.str.strip().str.lower()

    # WHO checks (all lowercase)
    df["ph_status"] = df["pH"].apply(lambda x: "✅ OK" if 6.5 <= x <= 8.5 else "⚠️ Out of Range")
    df["tds_status"] = df["tds"].apply(lambda x: "✅ OK" if x <= 1000 else "⚠️ High")
    df["ec_status"] = df["ec_val"].apply(lambda x: "✅ OK" if x <= 1400 else "⚠️ High")
    
    if "coliform" in df.columns:
        df["coliform_status"] = df["coliform"].apply(lambda x: "✅ Safe" if x == 0 else "🚨 Unsafe")
    else:
        df["coliform_status"] = "⚠️ Missing"

    if "hardness" in df.columns:
        df["hardness_status"] = df["hardness"].apply(
            lambda x: "💧 Soft" if x <= 60 else
                      "🧂 Moderate" if x <= 120 else
                      "🪨 Hard" if x <= 180 else
                      "⚠️ Very Hard"
        )
    else:
        df["hardness_status"] = "⚠️ Missing"

    if "do" in df.columns:
        df["do_status"] = df["do"].apply(lambda x: "✅ Good" if x >= 6 else "⚠️ Low")
    else:
        df["do_status"] = "⚠️ Missing"

    # AI prediction
    try:
        model = load_model("water_quality_ann.h5")
        scaler = joblib.load("scaler.pkl")
        features = df[["ec_val", "temp", "pH", "tds"]]
        X_scaled = scaler.transform(features)
        pred = model.predict(X_scaled)
        df["prediction"] = np.argmax(pred, axis=1)
        df["interpretation"] = df["prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})
        st.success("🧠 AI predictions generated!")
    except:
        df["interpretation"] = "Unavailable"
        st.warning("⚠️ Model or scaler missing.")

    # 🔵 Generate Pie Charts for WHO Parameters
    def pie_chart(column, title):
        counts = df[column].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        colors = ["#4CAF50" if "✅" in l or "💧" in l else
                  "#FFC107" if "⚠️" in l or "🧂" in l else
                  "#F44336" for l in labels]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.subheader(f"📊 {title}")
        st.pyplot(fig)

    pie_chart("ph_status", "pH Compliance")
    pie_chart("tds_status", "TDS Levels")
    pie_chart("ec_status", "EC Compliance")
    if "coliform_status" in df.columns:
        pie_chart("coliform_status", "Coliform Presence")
    pie_chart("hardness_status", "Hardness Distribution")
    pie_chart("do_status", "Dissolved Oxygen")

    # 🚨 Advisory if coliform present
    if "🚨 Unsafe" in df["coliform_status"].values:
        st.error("🚨 Coliform bacteria detected!")
        with st.warning("💡 Boiling Water Advisory"):
            st.markdown("""
            Boil water for **at least 1 minute at a rolling boil** before use.  
            Protect infants, elders, and those with low immunity.  
            Microbial contamination suggests a sanitation risk.
            """)

    # 📌 Optional: Summary text
    st.markdown("---")
    st.subheader("🧾 Parameter Overview")
    param_summary = []
    for col in ["ph_status", "tds_status", "ec_status", "do_status", "coliform_status", "hardness_status"]:
        if col in df.columns:
            safe = df[col].str.contains("✅|💧|🧂|🪨").sum()
            total = df.shape[0]
            pct = (safe / total) * 100
            param_summary.append(f"**{col.replace('_status','').upper()}**: {pct:.1f}% samples in acceptable range.")

    st.markdown("\n".join(param_summary))

else:
    st.info("📂 Please upload both Physical and Bacterial CSV files to begin.")






