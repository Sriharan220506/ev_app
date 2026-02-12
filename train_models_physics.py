"""
Research Paper-Based Battery SOH/RUL Model Training
====================================================
Based on: Patrizi et al. (2024), Sensors 24, 3382
"A Review of Degradation Models and RUL Prediction for Li-Ion Batteries"

Key Methods Applied:
1. Single Exponential Degradation: C_k = C0 + a * e^(b/k)  [Eq. 4]
2. SOH Definition: SOH_k = C_k / C_rated                    [Eq. 5]
3. Polynomial Capacity Fade: C(q) = b0 + b1*sqrt(q) + b2*q + b3*q^2  [Eq. 7]
4. RUL: cycles until SOH < 80% (failure threshold)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

# =================== CONFIGURATION ===================
DATASET_PATH = r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv'
MODEL_DIR = r'c:\Users\SriHaran\Documents\Projects\evapp\backend\models_ml'
os.makedirs(MODEL_DIR, exist_ok=True)

# Battery Specs
NOMINAL_CAPACITY_DATASET = 2.000  # Dataset nominal (Ah), inferred from BCt max ~1.99
RATED_CAPACITY_USER = 1.200       # User's battery (1200mAh)
FAILURE_THRESHOLD = 0.80          # 80% SOH = End of Life (from paper)

print("=" * 60)
print("RESEARCH PAPER-BASED MODEL TRAINING")
print("Patrizi et al. (2024), Sensors 24, 3382")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\nDataset: {len(df)} rows, Batteries: {df['battery_id'].nunique()}")
print(f"Cycle range: {df['cycle'].min()} - {df['cycle'].max()}")
print(f"BCt range: {df['BCt'].min():.4f} - {df['BCt'].max():.4f} Ah")

# =================== PHASE 1: EXPONENTIAL DEGRADATION FIT ===================
# Paper Eq. 4: C_k = C0 + a * exp(b / k)
# We fit this to each battery's capacity curve to extract degradation parameters.

def single_exp_model(k, C0, a, b):
    """Single exponential degradation model (Paper Eq. 4)"""
    return C0 + a * np.exp(b / k)

print("\n--- Phase 1: Fitting Single Exponential Degradation Model ---")
print("Model: C_k = C0 + a * exp(b/k)  [Paper Eq. 4]")

exp_params = {}
for bid in df['battery_id'].unique():
    subset = df[df['battery_id'] == bid].sort_values('cycle')
    cycles = subset['cycle'].values.astype(float)
    capacity = subset['BCt'].values
    
    # Initial guesses: C0 ~ min capacity, a ~ capacity range, b ~ negative (decay)
    try:
        popt, pcov = curve_fit(
            single_exp_model, cycles, capacity,
            p0=[capacity.min(), capacity.max() - capacity.min(), -50],
            maxfev=5000
        )
        C0_fit, a_fit, b_fit = popt
        # Calculate R² of the fit
        predicted = single_exp_model(cycles, *popt)
        r2 = r2_score(capacity, predicted)
        exp_params[bid] = {'C0': C0_fit, 'a': a_fit, 'b': b_fit, 'R2': r2}
        print(f"  Battery {bid}: C0={C0_fit:.4f}, a={a_fit:.4f}, b={b_fit:.2f}, R²={r2:.4f}")
    except Exception as e:
        print(f"  Battery {bid}: Fit failed ({e}), using defaults")
        exp_params[bid] = {'C0': capacity.min(), 'a': 0.5, 'b': -30, 'R2': 0.0}

# Average exponential parameters across all batteries
avg_C0 = np.mean([p['C0'] for p in exp_params.values()])
avg_a = np.mean([p['a'] for p in exp_params.values()])
avg_b = np.mean([p['b'] for p in exp_params.values()])
print(f"\n  Average params: C0={avg_C0:.4f}, a={avg_a:.4f}, b={avg_b:.2f}")

# Save exponential parameters for backend use
with open(os.path.join(MODEL_DIR, 'exp_params.json'), 'w') as f:
    json.dump({
        'C0': float(avg_C0), 'a': float(avg_a), 'b': float(avg_b),
        'nominal_capacity': NOMINAL_CAPACITY_DATASET,
        'failure_threshold': FAILURE_THRESHOLD
    }, f, indent=2)

# =================== PHASE 2: FEATURE ENGINEERING ===================
print("\n--- Phase 2: Engineering Features (Paper-Inspired) ---")

# Internal Resistance proxy (IR)
df['IR'] = (df['chV'] - df['disV']) / (df['chI'] + df['disI'])

# Polynomial capacity fade features [Paper Eq. 7]
# C(q) = b0 + b1*sqrt(q) + b2*q + b3*q²
df['sqrt_cycle'] = np.sqrt(df['cycle'])
df['cycle_sq'] = df['cycle'] ** 2

# Exponential model prediction at each cycle (using avg params)
df['exp_capacity'] = single_exp_model(df['cycle'].values.astype(float), avg_C0, avg_a, avg_b)

# Capacity deviation from exponential model (residual — captures battery-specific variation)
df['cap_residual'] = df['BCt'] - df['exp_capacity']

# Temperature stress indicator
df['temp_diff'] = df['disT'] - df['chT']

# =================== PHASE 3: TARGET DEFINITION ===================
print("\n--- Phase 3: Defining Targets (Paper Eq. 5) ---")

# SOH: Paper Eq. 5 — SOH_k = C_k / C_rated
df['SOH'] = (df['BCt'] / NOMINAL_CAPACITY_DATASET) * 100.0
print(f"  SOH range: {df['SOH'].min():.1f}% - {df['SOH'].max():.1f}%")

# RUL: Cycles until SOH < 80% (Paper failure threshold)
# For each battery, find the cycle where SOH drops below 80%
df['RUL'] = 0
for bid in df['battery_id'].unique():
    subset = df[df['battery_id'] == bid].sort_values('cycle')
    soh_values = subset['SOH'].values
    cycles = subset['cycle'].values
    
    # Find failure cycle (SOH < 80%)
    fail_mask = soh_values < (FAILURE_THRESHOLD * 100)
    if fail_mask.any():
        fail_cycle = cycles[fail_mask][0]
    else:
        # Use exponential model to extrapolate failure cycle
        # Solve: C0 + a*exp(b/k) = 0.8 * Nominal
        target_cap = FAILURE_THRESHOLD * NOMINAL_CAPACITY_DATASET
        params = exp_params[bid]
        # Extrapolate: find k where C_k = target_cap
        for k_test in range(int(cycles.max()), 1000):
            c_pred = single_exp_model(float(k_test), params['C0'], params['a'], params['b'])
            if c_pred <= target_cap:
                fail_cycle = k_test
                break
        else:
            fail_cycle = 500  # cap at 500 cycles
    
    rul_vals = np.maximum(0, fail_cycle - cycles)
    df.loc[df['battery_id'] == bid, 'RUL'] = rul_vals

print(f"  RUL range: {df['RUL'].min()} - {df['RUL'].max()} cycles")

# Capacity Factor (normalized 0-1)
df['Capacity_Factor'] = df['BCt'] / NOMINAL_CAPACITY_DATASET

# =================== PHASE 4: MODEL TRAINING ===================
print("\n--- Phase 4: Training ML Models ---")

# Features: cycle data + polynomial + IR + exponential residual + temp
features = [
    'cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT',  # 7 base
    'IR', 'sqrt_cycle', 'cycle_sq', 'cap_residual', 'temp_diff'  # 5 paper-inspired
]
targets = ['SOH', 'RUL', 'Capacity_Factor']

print(f"  Features ({len(features)}): {features}")

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature names
with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'w') as f:
    f.write(','.join(features))

results = {}

for target in targets:
    print(f"\n  --- {target} ---")
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
    }
    
    best_name, best_score, best_model = "", -999, None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"    {name:20s}: R²={r2:.6f}  MAE={mae:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model
    
    print(f"    🏆 Best: {best_name} (R²={best_score:.6f})")
    results[target] = {'name': best_name, 'r2': best_score, 'model': best_model}
    
    # Save model
    fname = {
        'SOH': 'soh_model.pkl',
        'RUL': 'rul_model.pkl',
        'Capacity_Factor': 'bct_model.pkl'
    }[target]
    with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
        pickle.dump(best_model, f)

# =================== SUMMARY ===================
print("\n" + "=" * 60)
print("TRAINING COMPLETE — Summary")
print("=" * 60)
for target, info in results.items():
    print(f"  {target:20s} → {info['name']:20s} (R²={info['r2']:.6f})")
print(f"\nFeature count: {len(features)}")
print(f"Models saved to: {MODEL_DIR}")
print(f"Exponential params saved: exp_params.json")
print(f"\nPaper methods applied:")
print(f"  ✓ Single exponential degradation fit (Eq. 4)")
print(f"  ✓ SOH = C_k / C_rated (Eq. 5)")
print(f"  ✓ Polynomial capacity fade features (Eq. 7)")
print(f"  ✓ RUL = cycles to 80% SOH threshold")
