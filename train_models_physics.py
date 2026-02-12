import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# ----------------- CONFIGURATION -----------------
DATASET_PATH = r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv'
MODEL_DIR = r'c:\Users\SriHaran\Documents\Projects\evapp\backend\models_ml'
os.makedirs(MODEL_DIR, exist_ok=True)

# User's Battery Specs
NOMINAL_CAPACITY_USER = 1.200  # 1200mAh = 1.2Ah
# Dataset Specs (Inferred from BCt max ~1.99Ah)
NOMINAL_CAPACITY_DATASET = 2.000 

print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# ==============================================================================
# 1. PHYSICS-BASED FEATURE EXTRACTION
# ==============================================================================
print("\nExtracting physics-based features...")

# Feature 1: Internal Resistance (IR) proxy
# IR = Voltage Sag / Current Change
# During switch from Charge to Discharge, voltage drops significantly.
# We approximate IR using the diff in voltage averages relative to current averages.
# IR = (chV - disV) / (chI + disI)
# Note: This is an average cycle IR, not instantaneous.
df['IR'] = (df['chV'] - df['disV']) / (df['chI'] + df['disI'])

# Feature 2: Discharge Time (dt) proxy
# Capacity (Ah) = Current (A) * Time (h)  =>  Time = Capacity / Current
# We use 'BCt' as the measured capacity of the cycle.
df['dis_time'] = df['BCt'] / df['disI']

# Feature 3: Capacity Fade (Coulomb Counting)
# Cumulative sum of capacity loss is a strong indicator of age.
# First, determining the max capacity of each battery_id to calculate fade relative to it.
df['capacity_fade'] = 0.0
for bid in df['battery_id'].unique():
    subset = df[df['battery_id'] == bid]
    max_cap = subset['BCt'].max()
    # capacity_fade = Max Capacity - Current Capacity
    df.loc[df['battery_id'] == bid, 'capacity_fade'] = max_cap - df.loc[df['battery_id'] == bid, 'BCt']

# Additional Health Indicators (from previous phase, still useful)
df['temp_diff'] = df['disT'] - df['chT']
df['weighted_volts'] = df['chV'] * 0.5 + df['disV'] * 0.5

# ==============================================================================
# 2. TARGET DEFINITION
# ==============================================================================
# SOH: State of Health
# User definition: Current Capacity / Nominal (1200mAh).
# PROBLEM: The dataset is ~2Ah, user battery is 1.2Ah.
# SOLUTION: We train the model to predict SOH % relative to the DATASET's nominal (2.0Ah).
#           Then in the app, we apply this % to the user's 1.2Ah battery.
#           SOH = BCt / NOMINAL_CAPACITY_DATASET
df['SOH'] = (df['BCt'] / NOMINAL_CAPACITY_DATASET) * 100.0

# RUL: Remaining Useful Life
# User definition: Cycles remaining until SOH < 80%.
# We calculate the failure cycle for each battery where SOH drops below 80%.
print("Calculating RUL based on 80% SOH threshold...")
df['RUL'] = 0
for bid in df['battery_id'].unique():
    subset = df[df['battery_id'] == bid].sort_values('cycle')
    # Find cycle where SOH < 80% (or closest failure point)
    # The dataset batteries degrade from ~100% to ~37%.
    # 80% of 2.0Ah = 1.6Ah.
    failure_rows = subset[subset['SOH'] < 80]
    
    if len(failure_rows) > 0:
        failure_cycle = failure_rows.iloc[0]['cycle']
    else:
        failure_cycle = subset['cycle'].max() # If never fails, assume end of data
        
    # RUL = Failure Cycle - Current Cycle
    # If already passed failure, RUL = 0
    rul_vals = failure_cycle - subset['cycle']
    rul_vals = rul_vals.apply(lambda x: max(0, x))
    df.loc[df['battery_id'] == bid, 'RUL'] = rul_vals

# BCt: Capacity Target (Ah)
# We will predict the normalized relative capacity (0-1 current capacity factor)
# Target = BCt / NOMINAL_CAPACITY_DATASET
df['Capacity_Factor'] = df['BCt'] / NOMINAL_CAPACITY_DATASET

# ==============================================================================
# 3. TRAINING ML MODELS
# ==============================================================================
# Note: We exclude 'dis_time' and 'capacity_fade' from training keys because 
# they are derived from the target (BCt) and won't be available for real-time prediction.
features = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'IR']
targets = ['SOH', 'RUL', 'Capacity_Factor']

print(f"Features: {features}")
print(f"Targets: {targets}")

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler immediately
with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
    
# Save feature names
with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'w') as f:
    f.write(','.join(features))

results = {}

for target in targets:
    print(f"\n--- Training {target} Model ---")
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0)
    }
    
    best_name = ""
    best_score = -999
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"  {name}: R2={r2:.4f} MAE={mae:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model
            
    print(f"  🏆 Best {target}: {best_name} (R2={best_score:.4f})")
    results[target] = best_model
    
    # Save model
    fname = "bct_model.pkl" if target == "Capacity_Factor" else f"{target.lower()}_model.pkl"
    with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
        pickle.dump(best_model, f)

print("\nAll models trained and saved successfully!")
