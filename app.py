import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧")

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.radio("📌 Navigation", ["Dashboard", "History", "Map"])

# ---------------------------
# Initialize Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------
# Dashboard Page
# ---------------------------
if page == "Dashboard":
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

        # Split EC into EC_val and Temp
        try:
            phys_df[['ec_val', 'temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
        except Exception:
            st.error("⚠️ Unable to split 'EC'. Please ensure it's in 'value/temp' format (e.g., '1400/25').")

        # Merge data
        df = pd.merge(phys_df, bact_df, on="Sample", how="inner")
        df.columns = df.columns.str.strip().str.lower()

        # WHO Checks
        if 'ph' in df.columns:
            df["ph_status"] = df["ph"].apply(lambda x: "✅ OK" if 6.5 <= x <= 7.5 else "⚠️ Out of Range")
        if 'tds' in df.columns:
            df["tds_status"] = df["tds"].apply(lambda x: "✅ OK" if x <= 300 else "⚠️ High")
        if 'ec_val' in df.columns:
            df["ec_status"] = df["ec_val"].apply(lambda x: "✅ OK" if x <= 400 else "⚠️ High")
        if 'coliform' in df.columns:
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
            features = df[["ec_val", "temp", "ph", "tds"]]
            X_scaled = scaler.transform(features)
            prediction = model.predict(X_scaled)
            df["prediction"] = np.argmax(prediction, axis=1)
            df["interpretation"] = df["prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})
            st.success("🧠 AI predictions generated!")
        except Exception as e:
            st.warning(f"⚠️ Model/scaler issue: {e}")
            df["interpretation"] = "Unavailable"

        # Pie chart function
        def pie_chart(col, title):
            if col in df.columns:
                counts = df[col].value_counts()
                labels = counts.index.tolist()
                sizes = counts.values.tolist()
                colors = ["#4CAF50" if "✅" in l or "💧" in l else
                          "#FFC107" if "⚠️" in l or "🧂" in l or "🪨" in l else
                          "#F44336" for l in labels]
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')
                st.subheader(f"📊 {title}")
                st.pyplot(fig)

        pie_chart("ph_status", "pH Compliance")
        pie_chart("tds_status", "TDS Compliance")
        pie_chart("ec_status", "Electrical Conductivity")
        pie_chart("coliform_status", "Coliform Presence")
        pie_chart("hardness_status", "Water Hardness")
        pie_chart("do_status", "Dissolved Oxygen")

        # Advisory
        if "🚨 Unsafe" in df["coliform_status"].values:
            st.error("🚨 Coliform bacteria detected!")
            with st.warning("💡 Boiling Water Advisory"):
                st.markdown("""
                One or more samples show microbial contamination.  
                **Please boil water for at least 1 minute at a rolling boil** before drinking or cooking.  
                Vulnerable groups (infants, elderly, immunocompromised) are especially at risk.
                """)

        # Text summary
        st.subheader("📋 Overall Safety Summary")
        param_columns = [
            "ph_status", "tds_status", "ec_status",
            "coliform_status", "hardness_status", "do_status"
        ]
        for col in param_columns:
            if col in df.columns:
                total = len(df)
                safe = df[col].str.contains("✅|💧|🧂|🪨").sum()
                st.markdown(f"**{col.replace('_status','').upper()}**: {safe/total*100:.1f}% samples within acceptable range.")

        # Save to history
        st.session_state["history"].append(df)

    else:
        st.info("📂 Please upload both Physical and Bacterial CSV files to continue.")

# ---------------------------
# History Page
# ---------------------------
elif page == "History":
    st.title("📜 Result History")
    if st.session_state["history"]:
        for i, past_df in enumerate(st.session_state["history"], 1):
            with st.expander(f"Result {i}"):
                st.dataframe(past_df)
    else:
        st.info("No history yet. Upload files in Dashboard first.")

# ---------------------------
# Map Page
# ---------------------------
elif page == "Map":
    st.title("🗺️ Map View of Sampling Points")

    if st.session_state["history"]:
        latest_df = st.session_state["history"][-1]  # show map for latest upload
        if "lat" in latest_df.columns and "lon" in latest_df.columns:
            m = folium.Map(location=[latest_df["lat"].mean(), latest_df["lon"].mean()], zoom_start=10)
            for _, row in latest_df.iterrows():
                popup_text = f"Sample: {row['sample']}<br>Status: {row['interpretation']}"
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    popup=popup_text,
                    icon=folium.Icon(color="green" if row["interpretation"]=="Good" else "red")
                ).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.info("🌍 No coordinates found in data. Add 'lat' and 'lon' columns to enable map view.")
    else:
        st.info("No data available. Upload files in Dashboard first.")

    # --- Google Earth Button ---
    st.subheader("🌐 Explore in Google Earth")
    earth_url = (
        "https://earth.google.com/web/@23.01551068,91.97356712,"
        "200.53061734a,2699.9592911d,35y,-0h,0t,0r/"
        "data=CgRCAggBMikKJwolCiExV0NSdUd0VXlXX0s0eHBQaHlRNzFvTzBLcU1UN0YzS0cgAToDCgEwQgII"










