import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
import base64
from io import BytesIO
import time

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧")

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.radio("📌 Navigation", ["Dashboard", "History", "Map"])

# ---------------------------
# Initialize Shared History (using session state as persistent storage)
# ---------------------------
if "shared_history" not in st.session_state:
    st.session_state["shared_history"] = []

# ---------------------------
# Helper function to convert matplotlib figure to base64 for storage
# ---------------------------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def base64_to_fig(base64_str):
    buf = BytesIO(base64.b64decode(base64_str))
    return plt.imread(buf)

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
        - **TDS**: ≤ 300 mg/L  
        - **EC**: ≤ 750 µS/cm  
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
        def create_pie_chart(data, col, title):
            if col in data.columns:
                counts = data[col].value_counts()
                if len(counts) > 0:
                    labels = counts.index.tolist()
                    sizes = counts.values.tolist()
                    colors = ["#4CAF50" if "✅" in str(l) or "💧" in str(l) else
                              "#FFC107" if "⚠️" in str(l) or "🧂" in str(l) or "🪨" in str(l) else
                              "#F44336" for l in labels]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                    ax.axis('equal')
                    plt.title(title, fontsize=14, fontweight='bold')
                    return fig
            return None

        # Display pie charts
        st.subheader("📊 Analysis Results")
        
        charts_data = {}
        param_titles = {
            "ph_status": "pH Compliance",
            "tds_status": "TDS Compliance", 
            "ec_status": "Electrical Conductivity",
            "coliform_status": "Coliform Presence",
            "hardness_status": "Water Hardness",
            "do_status": "Dissolved Oxygen"
        }
        
        for param, title in param_titles.items():
            fig = create_pie_chart(df, param, title)
            if fig:
                charts_data[title] = fig_to_base64(fig)
                st.pyplot(fig)

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
        
        summary_data = {}
        for col in param_columns:
            if col in df.columns:
                total = len(df)
                safe_count = df[col].astype(str).str.contains("✅|💧|🧂|🪨", regex=True).sum()
                percentage = safe_count/total*100
                summary_data[col.replace('_status','').upper()] = percentage
                st.markdown(f"**{col.replace('_status','').upper()}**: {percentage:.1f}% samples within acceptable range.")

        # AI prediction summary
        if "interpretation" in df.columns:
            st.subheader("🤖 AI Quality Assessment")
            quality_counts = df["interpretation"].value_counts()
            for quality, count in quality_counts.items():
                st.markdown(f"**{quality}**: {count} samples ({count/len(df)*100:.1f}%)")

        # Save to shared history
        history_entry = {
            "timestamp": time.time(),
            "summary": summary_data,
            "charts": charts_data,
            "total_samples": len(df),
            "ai_quality": dict(df["interpretation"].value_counts()) if "interpretation" in df.columns else {},
            "has_coliform_alert": "🚨 Unsafe" in df["coliform_status"].values if "coliform_status" in df.columns else False,
            "user": f"User_{int(time.time()) % 10000}"  # Simple user identifier
        }
        
        st.session_state["shared_history"].append(history_entry)
        st.success("✅ Analysis completed and saved to shared history!")

    else:
        st.info("📂 Please upload both Physical and Bacterial CSV files to continue.")

# ---------------------------
# History Page (Shared across all users)
# ---------------------------
elif page == "History":
    st.title("📜 Shared Analysis History")
    st.info("🌐 Viewing all analysis results from all users")
    
    if st.session_state["shared_history"]:
        # Display in reverse order (newest first)
        for i, history_entry in enumerate(reversed(st.session_state["shared_history"]), 1):
            with st.expander(f"Analysis {i} - {time.strftime('%Y-%m-%d %H:%M', time.localtime(history_entry['timestamp']))} (by {history_entry['user']})"):
                
                # Display summary statistics
                st.subheader("📋 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", history_entry["total_samples"])
                
                with col2:
                    if history_entry["has_coliform_alert"]:
                        st.error("🚨 Coliform Detected")
                    else:
                        st.success("✅ No Coliform Issues")
                
                with col3:
                    st.write(f"User: {history_entry['user']}")
                
                # Display parameter percentages
                st.subheader("Parameter Compliance")
                for param, percentage in history_entry["summary"].items():
                    st.progress(float(percentage/100), text=f"{param}: {percentage:.1f}% compliant")
                
                # Display AI quality assessment
                if history_entry["ai_quality"]:
                    st.subheader("🤖 AI Quality Assessment")
                    for quality, count in history_entry["ai_quality"].items():
                        percentage = count/history_entry["total_samples"]*100
                        st.write(f"**{quality}**: {count} samples ({percentage:.1f}%)")
                
                # Display charts
                st.subheader("📊 Analysis Charts")
                if history_entry["charts"]:
                    cols = st.columns(2)
                    col_idx = 0
                    
                    for title, fig_base64 in history_entry["charts"].items():
                        # Create a temporary figure for display
                        fig, ax = plt.subplots(figsize=(6, 5))
                        img = base64_to_fig(fig_base64)
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(title, fontsize=12, fontweight='bold')
                        
                        with cols[col_idx]:
                            st.pyplot(fig)
                        
                        col_idx = (col_idx + 1) % 2
                        plt.close(fig)
                else:
                    st.info("No charts available for this analysis.")
                    
                st.markdown("---")
    else:
        st.info("No analysis history yet. Upload files in Dashboard first to generate results.")

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
    earth_url = "https://earth.google.com/earth/d/1Dl41EVJhvi4KIbI4dViC7mUyNVmNssVb?usp=sharing"
    
    st.markdown(f"""
    <a href="{earth_url}" target="_blank">
        <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            🗺️ Open in Google Earth
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.info("💡 Make sure you have Google Earth installed or use the web version at earth.google.com")

    # Show shared history info if available
    if st.session_state["shared_history"]:
        st.subheader("📊 Shared History Overview")
        total_analyses = len(st.session_state["shared_history"])
        latest_analysis = max(st.session_state["shared_history"], key=lambda x: x["timestamp"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.write(f"Latest: {time.strftime('%Y-%m-%d %H:%M', time.localtime(latest_analysis['timestamp']))}")
    else:
        st.info("No analysis data available. Upload files in Dashboard first.")
















