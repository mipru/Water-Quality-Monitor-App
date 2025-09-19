import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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
        except Exception as e:
            st.error(f"⚠️ Unable to split 'EC'. Please ensure it's in 'value/temp' format (e.g., '1400/25'). Error: {e}")
            st.stop()

        # Merge data
        df = pd.merge(phys_df, bact_df, on="Sample", how="inner")
        df.columns = df.columns.str.strip().str.lower()

        # WHO Checks
        if 'ph' in df.columns:
            df["ph_status"] = df["ph"].apply(lambda x: "✅ OK" if 6.5 <= x <= 8.5 else "⚠️ Out of Range")
        else:
            df["ph_status"] = "⚠️ Missing"
            
        if 'tds' in df.columns:
            df["tds_status"] = df["tds"].apply(lambda x: "✅ OK" if x <= 1000 else "⚠️ High")
        else:
            df["tds_status"] = "⚠️ Missing"
            
        if 'ec_val' in df.columns:
            df["ec_status"] = df["ec_val"].apply(lambda x: "✅ OK" if x <= 1400 else "⚠️ High")
        else:
            df["ec_status"] = "⚠️ Missing"
            
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
            df["prediction"] = -1

        # Pie chart function
        def pie_chart(col, title):
            if col in df.columns:
                counts = df[col].value_counts()
                if len(counts) > 0:
                    labels = counts.index.tolist()
                    sizes = counts.values.tolist()
                    colors = ["#4CAF50" if "✅" in str(l) or "💧" in str(l) else
                              "#FFC107" if "⚠️" in str(l) or "🧂" in str(l) or "🪨" in str(l) else
                              "#F44336" for l in labels]
                    fig, ax = plt.subplots()
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                    ax.axis('equal')
                    st.subheader(f"📊 {title}")
                    st.pyplot(fig)
                else:
                    st.write(f"No data for {title}")

        # Display pie charts
        pie_chart("ph_status", "pH Compliance")
        pie_chart("tds_status", "TDS Compliance")
        pie_chart("ec_status", "Electrical Conductivity")
        pie_chart("coliform_status", "Coliform Presence")
        pie_chart("hardness_status", "Water Hardness")
        pie_chart("do_status", "Dissolved Oxygen")

        # Advisory
        if "coliform_status" in df.columns and "🚨 Unsafe" in df["coliform_status"].values:
            st.error("🚨 Coliform bacteria detected!")
            with st.expander("💡 Boiling Water Advisory"):
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
                # Use string methods safely
                safe_count = df[col].astype(str).str.contains("✅|💧|🧂|🪨", regex=True).sum()
                st.markdown(f"**{col.replace('_status','').upper()}**: {safe_count/total*100:.1f}% samples within acceptable range.")

        # Save to history
        st.session_state["history"].append(df.copy())

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
    
    # Google Earth Link Section
    st.subheader("🌐 Explore in Google Earth")
    st.markdown("""
    Click the button below to view the water quality results in Google Earth. 
    This will open your Google Earth application or web version with the sampling locations and results.
    """)
    
    # Your Google Earth link
    earth_url = "https://earth.google.com/earth/d/1BvhbCLWeRHEIu19l6V2YTBDXzrejSlgs?usp=sharing"
    
    st.markdown(f"""
    <a href="{earth_url}" target="_blank">
        <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            🗺️ Open in Google Earth
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.info("💡 Make sure you have Google Earth installed or use the web version at earth.google.com")

    # Show sample data preview if available
    if st.session_state["history"]:
        latest_df = st.session_state["history"][-1]
        st.subheader("📍 Sample Data Preview")
        
        # Check if coordinates exist
        if all(col in latest_df.columns for col in ["lat", "lon"]):
            st.write("Coordinates found in your data:")
            coord_df = latest_df[["sample", "lat", "lon", "interpretation"]].dropna(subset=["lat", "lon"])
            st.dataframe(coord_df.head())
        else:
            st.info("No coordinate data ('lat' and 'lon' columns) found in your dataset.")
    else:
        st.info("No data available. Upload files in Dashboard first.")












