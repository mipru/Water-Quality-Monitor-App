import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧")
st.title("💧 Smart Water Quality Analyzer")
st.write("Upload your water test results to check safety based on WHO guidelines and AI prediction.")

# 🧠 Educational Section
with st.expander(ℹ️ WHO Guidelines & Key Indicators)
    st.markdown(
    - Coliforms 0 CFU100 mL → 🚨 Unsafe if  0  
    - pH 6.5–8.5  
    - TDS ≤ 1000 mgL  
    - EC ≤ 1400 µScm  
    - Hardness 500 mgL may cause scaling  
    - DO (Dissolved Oxygen) 6 mgL preferred for freshness  
    )

# Upload CSVs
phys_file = st.file_uploader(Upload Physical Parameter CSV, type=[csv])
bact_file = st.file_uploader(Upload Bacterial Test CSV, type=[csv])

if phys_file and bact_file
    phys_df = pd.read_csv(phys_file)
    bact_df = pd.read_csv(bact_file)
    st.success(✅ Files uploaded!)

    # Split EC into EC_val and Temp
    try
        phys_df[['ec_val', 'temp']] = phys_df['EC'].str.split('', expand=True).astype(float)
    except Exception
        st.error(⚠️ Unable to split 'EC'. Please ensure it's in 'valuetemp' format (e.g., '140025').)

    # Merge data
    df = pd.merge(phys_df, bact_df, on=Sample, how=inner)
    df.columns = df.columns.str.strip().str.lower()

    # WHO Checks
    if 'ph' in df.columns
        df[ph_status] = df[ph].apply(lambda x ✅ OK if 6.5 = x = 8.5 else ⚠️ Out of Range)
    if 'tds' in df.columns
        df[tds_status] = df[tds].apply(lambda x ✅ OK if x = 1000 else ⚠️ High)
    if 'ec_val' in df.columns
        df[ec_status] = df[ec_val].apply(lambda x ✅ OK if x = 1400 else ⚠️ High)
    if 'coliform' in df.columns
        df[coliform_status] = df[coliform].apply(lambda x ✅ Safe if x == 0 else 🚨 Unsafe)
    else
        df[coliform_status] = ⚠️ Missing

    if hardness in df.columns
        df[hardness_status] = df[hardness].apply(
            lambda x 💧 Soft if x = 60 else
                      🧂 Moderate if x = 120 else
                      🪨 Hard if x = 180 else
                      ⚠️ Very Hard
        )
    else
        df[hardness_status] = ⚠️ Missing

    if do in df.columns
        df[do_status] = df[do].apply(lambda x ✅ Good if x = 6 else ⚠️ Low)
    else
        df[do_status] = ⚠️ Missing

    # AI prediction
    try
        model = load_model(water_quality_ann.h5)
        scaler = joblib.load(scaler.pkl)
        features = df[[ec_val, temp, ph, tds]]
        X_scaled = scaler.transform(features)
        prediction = model.predict(X_scaled)
        df[prediction] = np.argmax(prediction, axis=1)
        df[interpretation] = df[prediction].map({0 Good, 1 Moderate, 2 Poor})
        st.success(🧠 AI predictions generated!)
    except Exception as e
        st.warning(f⚠️ Modelscaler issue {e})
        df[interpretation] = Unavailable

    # Pie chart function
    def pie_chart(col, title)
        if col in df.columns
            counts = df[col].value_counts()
            labels = counts.index.tolist()
            sizes = counts.values.tolist()
            colors = [#4CAF50 if ✅ in l or 💧 in l else
                      #FFC107 if ⚠️ in l or 🧂 in l or 🪨 in l else
                      #F44336 for l in labels]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.subheader(f📊 {title})
            st.pyplot(fig)

    pie_chart(ph_status, pH Compliance)
    pie_chart(tds_status, TDS Compliance)
    pie_chart(ec_status, Electrical Conductivity)
    pie_chart(coliform_status, Coliform Presence)
    pie_chart(hardness_status, Water Hardness)
    pie_chart(do_status, Dissolved Oxygen)

    # Advisory
    if 🚨 Unsafe in df[coliform_status].values
        st.error(🚨 Coliform bacteria detected!)
        with st.warning(💡 Boiling Water Advisory)
            st.markdown(
            One or more samples show microbial contamination.  
            Please boil water for at least 1 minute at a rolling boil before drinking or cooking.  
            Vulnerable groups (infants, elderly, immunocompromised) are especially at risk.
            )

    # Text summary
    st.subheader(📋 Overall Safety Summary)
    param_columns = [
        ph_status, tds_status, ec_status,
        coliform_status, hardness_status, do_status
    ]
    for col in param_columns
        if col in df.columns
            total = len(df)
            safe = df[col].str.contains(✅💧🧂🪨).sum()
            st.markdown(f{col.replace('_status','').upper()} {safetotal100.1f}% samples within acceptable range.)

else
    st.info(📂 Please upload both Physical and Bacterial CSV files to continue.)






