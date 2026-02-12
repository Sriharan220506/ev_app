"""
=================================================================
EV Battery Dataset — Complete Analysis & ML Pipeline
=================================================================
Phase 1: Dataset Analysis (statistics, missing values, outliers)
Phase 2: Health Indicator Extraction (voltage rise time, peak temp, discharge capacity)
Phase 3: ML Model Training & Comparison (SOH, RUL, BCT)
Phase 4: Save Best Models
=================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# PHASE 1: DATASET ANALYSIS
# ===================================================================
print("=" * 70)
print("  PHASE 1: DATASET ANALYSIS")
print("=" * 70)

df = pd.read_csv(r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv')

print(f"\n📊 BASIC INFO:")
print(f"  Shape:    {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns:  {list(df.columns)}")
print(f"  Dtypes:")
for col in df.columns:
    print(f"    {col:15s}  {str(df[col].dtype):10s}  (non-null: {df[col].notna().sum()}/{len(df)})")

print(f"\n📊 SUMMARY STATISTICS:")
print(df.describe().round(4).to_string())

# --- Cycles per battery ---
print(f"\n📊 CYCLES PER BATTERY:")
for bid in sorted(df['battery_id'].unique()):
    sub = df[df['battery_id'] == bid]
    print(f"  {bid}: {sub['cycle'].nunique()} cycles  (cycle range: {sub['cycle'].min()} — {sub['cycle'].max()})")
print(f"  Total unique cycles across all batteries: {df['cycle'].nunique()}")

# --- Missing values ---
print(f"\n📊 MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values found in any column!")
else:
    for col in df.columns:
        if missing[col] > 0:
            print(f"  ⚠️ {col}: {missing[col]} missing ({missing[col]/len(df)*100:.1f}%)")

# --- Outlier detection (IQR method) ---
print(f"\n📊 OUTLIER DETECTION (IQR method, >1.5×IQR):")
numeric_cols = ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'SOH', 'RUL', 'BCt']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        print(f"  ⚠️ {col:6s}: {len(outliers):3d} outliers  (range: {df[col].min():.4f} — {df[col].max():.4f}, IQR bounds: {lower:.4f} — {upper:.4f})")
    else:
        print(f"  ✅ {col:6s}: 0 outliers    (range: {df[col].min():.4f} — {df[col].max():.4f})")

# --- Z-score outliers for temperature and voltage specifically ---
print(f"\n📊 Z-SCORE OUTLIERS (|z| > 3) — Temperature & Voltage:")
for col in ['chT', 'disT', 'chV', 'disV']:
    z = np.abs(stats.zscore(df[col]))
    z_outliers = (z > 3).sum()
    print(f"  {col:6s}: {z_outliers} values with |z| > 3  (mean={df[col].mean():.4f}, std={df[col].std():.4f})")

# ===================================================================
# PHASE 2: HEALTH INDICATOR EXTRACTION
# ===================================================================
print(f"\n\n{'=' * 70}")
print("  PHASE 2: HEALTH INDICATOR EXTRACTION")
print("=" * 70)

# Since this is a cycle-level dataset (1 row per cycle, not time-series),
# we derive health indicators from the available aggregate values.

health_df = df.copy()

# --- Feature 1: Voltage Rise Indicator ---
# Proxy: difference between charging voltage and discharge voltage
# Higher difference = more voltage drop = more degradation
health_df['voltage_rise'] = health_df['chV'] - health_df['disV']
print(f"\n✅ voltage_rise (chV - disV):   mean={health_df['voltage_rise'].mean():.4f}, range={health_df['voltage_rise'].min():.4f} — {health_df['voltage_rise'].max():.4f}")

# --- Feature 2: Peak Temperature During Discharge ---
# disT is already the discharge temperature; we add temp difference
health_df['temp_diff'] = health_df['disT'] - health_df['chT']
health_df['peak_temp'] = health_df[['chT', 'disT']].max(axis=1)
print(f"✅ peak_temp (max of chT, disT): mean={health_df['peak_temp'].mean():.4f}, range={health_df['peak_temp'].min():.4f} — {health_df['peak_temp'].max():.4f}")
print(f"✅ temp_diff (disT - chT):        mean={health_df['temp_diff'].mean():.4f}")

# --- Feature 3: Discharge Capacity (Proxy) ---
# In a cycle-level dataset, capacity ~ disI × voltage_factor
# Higher discharge current at lower voltage = used more capacity
health_df['discharge_capacity'] = health_df['disI'] * health_df['disV']
print(f"✅ discharge_capacity (disI × disV): mean={health_df['discharge_capacity'].mean():.4f}")

# --- Feature 4: Charging Energy ---
health_df['charge_energy'] = health_df['chI'] * health_df['chV']
print(f"✅ charge_energy (chI × chV):     mean={health_df['charge_energy'].mean():.4f}")

# --- Feature 5: Efficiency Ratio ---
health_df['efficiency'] = health_df['discharge_capacity'] / health_df['charge_energy']
print(f"✅ efficiency (dis_cap / ch_energy): mean={health_df['efficiency'].mean():.4f}")

# --- Feature 6: Current Ratio ---
health_df['current_ratio'] = health_df['disI'] / health_df['chI']
print(f"✅ current_ratio (disI / chI):    mean={health_df['current_ratio'].mean():.4f}")

# --- Feature 7: Temperature Stress ---
health_df['temp_stress'] = health_df['disT'] * health_df['disI']  # heat generation proxy
print(f"✅ temp_stress (disT × disI):     mean={health_df['temp_stress'].mean():.4f}")

# --- Feature 8: Cycle degradation rate (SOH change per cycle) ---
for bid in health_df['battery_id'].unique():
    mask = health_df['battery_id'] == bid
    health_df.loc[mask, 'soh_change'] = health_df.loc[mask, 'SOH'].diff().fillna(0)
print(f"✅ soh_change (SOH diff per cycle): mean={health_df['soh_change'].mean():.4f}")

print(f"\n  Cycle-level DataFrame shape: {health_df.shape}")
print(f"  New features added: voltage_rise, temp_diff, peak_temp, discharge_capacity, charge_energy, efficiency, current_ratio, temp_stress, soh_change")

# ===================================================================
# PHASE 3: ML MODEL TRAINING & COMPARISON
# ===================================================================
print(f"\n\n{'=' * 70}")
print("  PHASE 3: ML MODEL TRAINING & COMPARISON")
print("=" * 70)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import pickle, os

# All features (original + health indicators)
feature_cols = [
    'cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT',
    'voltage_rise', 'temp_diff', 'peak_temp',
    'discharge_capacity', 'charge_energy', 'efficiency',
    'current_ratio', 'temp_stress'
]

X = health_df[feature_cols].values
y_soh = health_df['SOH'].values
y_rul = health_df['RUL'].values
y_bct = health_df['BCt'].values

# 70/30 split as requested
X_train, X_test, y_soh_train, y_soh_test = train_test_split(X, y_soh, test_size=0.3, random_state=42)
_, _, y_rul_train, y_rul_test = train_test_split(X, y_rul, test_size=0.3, random_state=42)
_, _, y_bct_train, y_bct_test = train_test_split(X, y_bct, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} (70/30 split)")
print(f"  Features ({len(feature_cols)}): {feature_cols}")

def create_all_models():
    """Create fresh model instances for fair comparison"""
    return {
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
    }

def train_target(target_name, y_train, y_test, y_range_label, Xtr=None, Xte=None):
    """Train all models for one target, return best"""
    if Xtr is None: Xtr = X_train_s
    if Xte is None: Xte = X_test_s
    
    print(f"\n{'─' * 60}")
    print(f"  {target_name}  (range: {y_train.min():.2f} — {y_train.max():.2f})")
    print(f"{'─' * 60}")
    print(f"  {'Model':22s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s} {'CV R²':>8s}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    
    models = create_all_models()
    results = []
    
    for name, model in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        try:
            cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2').mean()
        except:
            cv = 0.0
        
        results.append({'name': name, 'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse, 'cv': cv})
        
        print(f"  {name:22s} {r2:8.4f} {mae:10.4f} {rmse:10.4f} {cv:8.4f}")
    
    # Pick best by R²
    results.sort(key=lambda x: x['r2'], reverse=True)
    best = results[0]
    
    # Verify prediction range
    best_pred = best['model'].predict(Xte)
    print(f"\n  🏆 BEST: {best['name']}")
    print(f"     R² = {best['r2']:.4f}  |  MAE = {best['mae']:.4f}  |  RMSE = {best['rmse']:.4f}")
    print(f"     Prediction range: {best_pred.min():.2f} — {best_pred.max():.2f}")
    print(f"     Actual range:     {y_test.min():.2f} — {y_test.max():.2f}")
    
    return best['name'], best['model'], results

# Train all 3 targets
soh_name, soh_model, soh_results = train_target("SOH (State of Health %)", y_soh_train, y_soh_test, "37-99%")
rul_name, rul_model, rul_results = train_target("RUL (Remaining Useful Life)", y_rul_train, y_rul_test, "0-249 cycles")
bct_name, bct_model, bct_results = train_target("BCT (Battery Capacity)", y_bct_train, y_bct_test, "0.75-1.99")

# ===================================================================
# PHASE 3b: RUL SEQUENCE-BASED PIPELINE
# ===================================================================
print(f"\n\n{'=' * 70}")
print("  PHASE 3b: RUL SEQUENCE-BASED PIPELINE")
print("=" * 70)

# For RUL, use previous N cycles as sequence features
def create_sequence_features(data, window=5):
    """Create sequence features using sliding window of previous cycles"""
    seq_rows = []
    for bid in data['battery_id'].unique():
        bdata = data[data['battery_id'] == bid].sort_values('cycle').reset_index(drop=True)
        for i in range(window, len(bdata)):
            features = {}
            # Current cycle basic features
            for col in ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT']:
                features[f'{col}_now'] = bdata.loc[i, col]
            
            # Rolling stats from previous N cycles  
            for col in ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'SOH']:
                window_data = bdata.loc[i-window:i-1, col].values
                features[f'{col}_mean_{window}'] = np.mean(window_data)
                features[f'{col}_std_{window}'] = np.std(window_data)
                features[f'{col}_trend_{window}'] = window_data[-1] - window_data[0]
            
            # SOH degradation rate
            soh_window = bdata.loc[i-window:i-1, 'SOH'].values
            features['soh_degradation_rate'] = (soh_window[0] - soh_window[-1]) / window
            
            features['RUL'] = bdata.loc[i, 'RUL']
            seq_rows.append(features)
    
    return pd.DataFrame(seq_rows)

print(f"\n  Building sequence features (window=5 previous cycles)...")
seq_df = create_sequence_features(health_df, window=5)
print(f"  Sequence dataset: {seq_df.shape[0]} rows × {seq_df.shape[1]} columns")

seq_feature_cols = [c for c in seq_df.columns if c != 'RUL']
X_seq = seq_df[seq_feature_cols].values
y_seq_rul = seq_df['RUL'].values

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq_rul, test_size=0.3, random_state=42)

seq_scaler = StandardScaler()
X_seq_train_s = seq_scaler.fit_transform(X_seq_train)
X_seq_test_s = seq_scaler.transform(X_seq_test)

print(f"  Train: {X_seq_train.shape[0]} | Test: {X_seq_test.shape[0]}")

seq_rul_name, seq_rul_model, seq_rul_results = train_target(
    "RUL (Sequence-Based)", y_seq_train, y_seq_test, "0-249",
    Xtr=X_seq_train_s, Xte=X_seq_test_s
)

# Compare standard vs sequence RUL
print(f"\n  📊 RUL COMPARISON:")
rul_best = [r for r in rul_results if r['name'] == rul_name][0]
seq_best = [r for r in seq_rul_results if r['name'] == seq_rul_name][0]
print(f"    Standard RUL:  {rul_name:20s}  R²={rul_best['r2']:.4f}  MAE={rul_best['mae']:.4f}")
print(f"    Sequence RUL:  {seq_rul_name:20s}  R²={seq_best['r2']:.4f}  MAE={seq_best['mae']:.4f}")

# Pick the better one
if seq_best['r2'] > rul_best['r2']:
    final_rul_name = f"Seq-{seq_rul_name}"
    final_rul_model = seq_rul_model
    final_rul_r2 = seq_best['r2']
    final_rul_mae = seq_best['mae']
    use_sequence = True
    print(f"    ✅ Sequence-based RUL is better!")
else:
    final_rul_name = rul_name
    final_rul_model = rul_model
    final_rul_r2 = rul_best['r2']
    final_rul_mae = rul_best['mae']
    use_sequence = False
    print(f"    ✅ Standard RUL is better (or equal)!")

# ===================================================================
# PHASE 4: FINAL SUMMARY & SAVE
# ===================================================================
print(f"\n\n{'=' * 70}")
print("  FINAL RESULTS SUMMARY")
print("=" * 70)

soh_best = [r for r in soh_results if r['name'] == soh_name][0]
bct_best = [r for r in bct_results if r['name'] == bct_name][0]

print(f"\n  {'Target':10s} {'Best Model':22s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s}")
print(f"  {'─'*10} {'─'*22} {'─'*8} {'─'*10} {'─'*10}")
print(f"  {'SOH':10s} {soh_name:22s} {soh_best['r2']:8.4f} {soh_best['mae']:10.4f} {soh_best['rmse']:10.4f}")
print(f"  {'RUL':10s} {final_rul_name:22s} {final_rul_r2:8.4f} {final_rul_mae:10.4f}")
print(f"  {'BCT':10s} {bct_name:22s} {bct_best['r2']:8.4f} {bct_best['mae']:10.4f} {bct_best['rmse']:10.4f}")

# Verification
print(f"\n  🔍 SANITY CHECK:")
# New battery
new = scaler.transform(np.array([[1, 1.4, 4.25, 27.0, 2.0, 3.9, 33.0, 0.35, 6.0, 33.0, 7.8, 5.95, 1.31, 1.43, 66.0]]))
soh_p = soh_model.predict(new)[0]
bct_p = bct_model.predict(new)[0]
print(f"  New battery (cycle=1):  SOH={soh_p:.2f}%  BCT={bct_p:.4f}")

old = scaler.transform(np.array([[200, 1.3, 4.1, 29.0, 1.8, 3.0, 36.0, 1.1, 7.0, 36.0, 5.4, 5.33, 1.01, 1.38, 64.8]]))
soh_p2 = soh_model.predict(old)[0]
bct_p2 = bct_model.predict(old)[0]
print(f"  Old battery (cycle=200): SOH={soh_p2:.2f}%  BCT={bct_p2:.4f}")

# ===================================================================
# SAVE MODELS
# ===================================================================
output_dir = r'c:\Users\SriHaran\Documents\Projects\evapp\backend\models_ml'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'soh_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(soh_model, f)
print(f"\n  💾 Saved SOH model ({soh_name})")

with open(os.path.join(output_dir, 'rul_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(final_rul_model, f)
print(f"  💾 Saved RUL model ({final_rul_name})")

with open(os.path.join(output_dir, 'bct_model.pkl'), 'wb') as f:
    pickle.dump(bct_model, f)
print(f"  💾 Saved BCT model ({bct_name})")

with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"  💾 Saved feature scaler (15 features)")

with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
    f.write(','.join(feature_cols))
print(f"  💾 Saved feature names")

# Save health DataFrame for reference
health_df.to_csv(os.path.join(output_dir, 'health_indicators.csv'), index=False)
print(f"  💾 Saved health indicators CSV")

print(f"\n  ✅ ALL DONE! Models saved to {output_dir}")
