"""
Train ML models for SOH, RUL, and BCT prediction
Compare XGBoost, Random Forest, Gradient Boosting
Pick the best model for each target
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
print(f"Batteries: {df['battery_id'].unique()}")
print(f"Columns: {list(df.columns)}")
print()

# Features: cycle, chI, chV, chT, disI, disV, disT
feature_cols = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT']
X = df[feature_cols].values

# Targets
y_soh = df['SOH'].values
y_rul = df['RUL'].values
y_bct = df['BCt'].values

# Train/test split (same split for all)
X_train, X_test, y_soh_train, y_soh_test = train_test_split(X, y_soh, test_size=0.2, random_state=42)
_, _, y_rul_train, y_rul_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
_, _, y_bct_train, y_bct_test = train_test_split(X, y_bct, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"Features: {feature_cols}")
print()

# ============================
# 2. Define Models
# ============================
models = {
    'XGBoost': XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0
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

# ============================
# 3. Train & Evaluate
# ============================
results = {}

def train_and_evaluate(target_name, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"  TARGET: {target_name}")
    print(f"{'=' * 60}")
    print(f"  Range: {y_train.min():.2f} — {y_train.max():.2f}")
    
    best_model = None
    best_name = None
    best_r2 = -999
    target_results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-val
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        
        target_results[name] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'cv_r2': cv_mean, 'model': model
        }
        
        print(f"\n  {name}:")
        print(f"    MAE:    {mae:.4f}")
        print(f"    RMSE:   {rmse:.4f}")
        print(f"    R²:     {r2:.4f}")
        print(f"    CV R²:  {cv_mean:.4f}  (5-fold)")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
    
    print(f"\n  ✅ BEST for {target_name}: {best_name} (R² = {best_r2:.4f})")
    return best_name, best_model, target_results

# Train for each target
soh_best_name, soh_best_model, soh_results = train_and_evaluate("SOH", y_soh_train, y_soh_test)
rul_best_name, rul_best_model, rul_results = train_and_evaluate("RUL", y_rul_train, y_rul_test)
bct_best_name, bct_best_model, bct_results = train_and_evaluate("BCT", y_bct_train, y_bct_test)

# ============================
# 4. Summary
# ============================
print(f"\n{'=' * 60}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'=' * 60}")
print(f"  SOH:  {soh_best_name:20s}  R² = {soh_results[soh_best_name]['r2']:.4f}  MAE = {soh_results[soh_best_name]['mae']:.4f}")
print(f"  RUL:  {rul_best_name:20s}  R² = {rul_results[rul_best_name]['r2']:.4f}  MAE = {rul_results[rul_best_name]['mae']:.4f}")
print(f"  BCT:  {bct_best_name:20s}  R² = {bct_results[bct_best_name]['r2']:.4f}  MAE = {bct_results[bct_best_name]['mae']:.4f}")

# ============================
# 5. Save Best Models
# ============================
output_dir = r'c:\Users\SriHaran\Documents\Projects\evapp\backend\models_ml'
os.makedirs(output_dir, exist_ok=True)

# Save models
with open(os.path.join(output_dir, 'soh_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(soh_best_model, f)
print(f"\n  Saved SOH model ({soh_best_name})")

with open(os.path.join(output_dir, 'rul_model_xgboost.pkl'), 'wb') as f:
    pickle.dump(rul_best_model, f)
print(f"  Saved RUL model ({rul_best_name})")

with open(os.path.join(output_dir, 'bct_model.pkl'), 'wb') as f:
    pickle.dump(bct_best_model, f)
print(f"  Saved BCT model ({bct_best_name})")

# Save scaler
with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Saved feature scaler")

# Save feature names for reference
with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
    f.write(','.join(feature_cols))
print(f"  Saved feature names: {feature_cols}")

print(f"\n  All models saved to: {output_dir}")
print("  DONE!")
