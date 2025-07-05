# 📦 Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# 📥 Load & Merge Sample Data
phys_path = input("Enter path to Physical Parameter CSV: ")
bact_path = input("Enter path to Bacterial Test CSV: ")

phys_df = pd.read_csv(phys_path)
bact_df = pd.read_csv(bact_path)
phys_df[['EC_val', 'Temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
df = pd.merge(phys_df, bact_df, on="Sample", how="inner")

# 🧠 Label Water Quality
def classify_quality(do):
    if do < 5: return 'Poor'
    elif do < 7: return 'Moderate'
    else: return 'Good'
df['Water_Quality'] = df['DO'].apply(classify_quality)

# 🔬 Binary encoding
df['Ecoli_Present'] = df['E. Coli'].notna().astype(int)
df['Salmonella_Present'] = df['Salmonella'].notna().astype(int)

# 🛠 Feature Engineering
df['Hardness'] = df['Hardness /ppm'].abs()
df['TDS_Hardness'] = df['TDS'] / df['Hardness'].replace(0, 0.1)
df['DO_EC'] = df['DO'] * df['EC_val']
df['pH_Deviation'] = abs(df['pH'] - 7)
df['Bacteria_Load'] = df['Ecoli_Present'] + df['Salmonella_Present']

# 🎯 Features
features = ['pH', 'DO', 'TDS', 'EC_val', 'Temp', 'Hardness',
            'Ecoli_Present', 'Salmonella_Present',
            'TDS_Hardness', 'DO_EC', 'pH_Deviation', 'Bacteria_Load']
X = df[features]
y = df['Water_Quality'].map({'Poor': 0, 'Moderate': 1, 'Good': 2})

# ⚖ Normalize + Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)

# 🧠 ANN Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# 🌲 Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_scaled)

# 💾 Save Files
model.save("water_quality_ann.h5")
joblib.dump(iso, "anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Files saved:")
print("- water_quality_ann.h5")
print("- anomaly_model.pkl")
print("- scaler.pkl")
