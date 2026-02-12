"""
Train ML models for SOH, RUL, and BCT prediction — Fixed version
Each target gets FRESH model instances (not shared).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv(r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Features: cycle, chI, chV, chT, disI, disV, disT
feature_cols = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT']
X = df[feature_cols].values

# Targets
y_soh = df['SOH'].values
y_rul = df['RUL'].values
y_bct = df['BCt'].values

# Same split for all targets
X_train, X_test, y_soh_train, y_soh_test = train_test_split(X, y_soh, test_size=0.2, random_state=42)
_, _, y_rul_train, y_rul_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
_, _, y_bct_train, y_bct_test = train_test_split(X, y_bct, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"Features: {feature_cols}")

output_dir = r'c:\Users\SriHaran\Documents\Projects\evapp\backend\models_ml'
os.makedirs(output_dir, exist_ok=True)

# ============================
# 2. Helper function
# ============================
def create_models():
    """Create FRESH model instances — must be called for each target"""
    return {
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }

def train_and_pick_best(target_name, y_train, y_test):
    """Train all 3 models on one target and return the best"""
    print(f"\n{'='*60}")
    print(f"  TARGET: {target_name}  (range: {y_train.min():.2f} — {y_train.max():.2f})")
    print(f"{'='*60}")
    
    models = create_models()  # Fresh instances!
    best_model = None
    best_name = None
    best_r2 = -999
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
        
        print(f"  {name:20s}  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  CV={cv:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
    
    # Verify the best model predicts in the right range
    test_pred = best_model.predict(X_test_scaled)
    print(f"  ✅ BEST: {best_name} (R²={best_r2:.4f})")
    print(f"     Prediction range: {test_pred.min():.2f} — {test_pred.max():.2f}")
    print(f"     Actual range:     {y_test.min():.2f} — {y_test.max():.2f}")
    
    return best_name, best_model

# ============================
# 3. Train Each Target
# ============================
soh_name, soh_model = train_and_pick_best("SOH", y_soh_train, y_soh_test)
rul_name, rul_model = train_and_pick_best("RUL", y_rul_train, y_rul_test)
bct_name, bct_model = train_and_pick_best("BCT", y_bct_train, y_bct_test)

# ============================
# 4. Verify models aren't mixed up
# ============================
print(f"\n{'='*60}")
print(f"  VERIFICATION: Quick sanity check")
print(f"{'='*60}")

# Test with cycle=1 (new battery should have high SOH, high RUL)
new_battery = scaler.transform(np.array([[1, 1.4, 4.25, 27.0, 2.0, 3.9, 33.0]]))
soh_pred = soh_model.predict(new_battery)[0]
rul_pred = rul_model.predict(new_battery)[0]
bct_pred = bct_model.predict(new_battery)[0]
print(f"  New battery (cycle=1):")
print(f"    SOH = {soh_pred:.2f}%  (should be ~98-99%)")
print(f"    RUL = {int(rul_pred)} cycles  (should be ~240-249)")
print(f"    BCT = {bct_pred:.4f}  (should be ~1.97)")

# Test with cycle=200 (old battery should have low SOH, low RUL)
old_battery = scaler.transform(np.array([[200, 1.3, 4.1, 29.0, 1.8, 3.0, 36.0]]))
soh_pred2 = soh_model.predict(old_battery)[0]
rul_pred2 = rul_model.predict(old_battery)[0]
bct_pred2 = bct_model.predict(old_battery)[0]
print(f"  Old battery (cycle=200):")
print(f"    SOH = {soh_pred2:.2f}%  (should be ~50-60%)")
print(f"    RUL = {int(rul_pred2)} cycles  (should be ~40-50)")
print(f"    BCT = {bct_pred2:.4f}  (should be ~0.9-1.0)")

# ============================
# 5. Save Models
# ============================
with open(os.path.join(output_dir, 'soh_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(soh_model, f)
print(f"\n  Saved SOH model ({soh_name})")

with open(os.path.join(output_dir, 'rul_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(rul_model, f)
print(f"  Saved RUL model ({rul_name})")

with open(os.path.join(output_dir, 'bct_model.pkl'), 'wb') as f:
    pickle.dump(bct_model, f)
print(f"  Saved BCT model ({bct_name})")

with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Saved feature scaler")

print(f"\n  DONE! All models saved to {output_dir}")
