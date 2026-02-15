"""
Battery Dataset Augmentation Script
====================================
Uses the paper's single exponential degradation model (Patrizi et al. 2024)
to generate synthetic battery data based on real data patterns.

Input:  680 rows, 3 batteries (B5, B6, B7)
Output: ~5000+ rows, 20+ batteries with realistic degradation curves
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

# =================== CONFIGURATION ===================
INPUT_PATH = r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv'
OUTPUT_PATH = r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset.csv'
BACKUP_PATH = r'c:\Users\SriHaran\Documents\Projects\evapp\DATASET\Battery_dataset_original.csv'

NUM_SYNTHETIC_BATTERIES = 20  # Generate 20 new batteries
CYCLES_RANGE = (180, 350)     # Each battery has 180-350 cycles
NOMINAL_CAPACITY = 2.0        # Ah

np.random.seed(42)

# =================== LOAD ORIGINAL DATA ===================
df_orig = pd.read_csv(INPUT_PATH)
print(f"Original dataset: {len(df_orig)} rows, {df_orig['battery_id'].nunique()} batteries")

# Save backup
df_orig.to_csv(BACKUP_PATH, index=False)
print(f"Backup saved to: {BACKUP_PATH}")

# =================== LEARN PATTERNS FROM REAL DATA ===================
# Fit exponential degradation per battery
def exp_model(k, C0, a, b):
    """C_k = C0 + a * exp(b/k)"""
    return C0 + a * np.exp(b / k)

# Extract statistics from each real battery
battery_profiles = {}
for bid in df_orig['battery_id'].unique():
    sub = df_orig[df_orig['battery_id'] == bid].sort_values('cycle')
    
    # Fit exponential to capacity curve
    try:
        popt, _ = curve_fit(exp_model, sub['cycle'].values.astype(float), sub['BCt'].values,
                           p0=[sub['BCt'].min(), 1.0, -50], maxfev=5000)
    except:
        popt = [sub['BCt'].min(), 0.5, -30]
    
    profile = {
        'C0': popt[0], 'a': popt[1], 'b': popt[2],
        'max_cycles': sub['cycle'].max(),
        # Per-column statistics at cycle 1 and at max cycle
        'chI_start': sub['chI'].iloc[0], 'chI_end': sub['chI'].iloc[-1],
        'chI_mean': sub['chI'].mean(), 'chI_std': sub['chI'].std(),
        'chV_start': sub['chV'].iloc[0], 'chV_end': sub['chV'].iloc[-1],
        'chV_mean': sub['chV'].mean(), 'chV_std': sub['chV'].std(),
        'chT_start': sub['chT'].iloc[0], 'chT_end': sub['chT'].iloc[-1],
        'chT_mean': sub['chT'].mean(), 'chT_std': sub['chT'].std(),
        'disI_start': sub['disI'].iloc[0], 'disI_end': sub['disI'].iloc[-1],
        'disI_mean': sub['disI'].mean(), 'disI_std': sub['disI'].std(),
        'disV_start': sub['disV'].iloc[0], 'disV_end': sub['disV'].iloc[-1],
        'disV_mean': sub['disV'].mean(), 'disV_std': sub['disV'].std(),
        'disT_start': sub['disT'].iloc[0], 'disT_end': sub['disT'].iloc[-1],
        'disT_mean': sub['disT'].mean(), 'disT_std': sub['disT'].std(),
        'BCt_start': sub['BCt'].iloc[0], 'BCt_end': sub['BCt'].iloc[-1],
    }
    battery_profiles[bid] = profile
    print(f"  {bid}: C0={popt[0]:.4f}, a={popt[1]:.4f}, b={popt[2]:.2f}, cycles={sub['cycle'].max()}")

# Overall column ranges from real data
col_ranges = {}
for col in ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']:
    col_ranges[col] = {
        'min': df_orig[col].min(),
        'max': df_orig[col].max(),
        'mean': df_orig[col].mean(),
        'std': df_orig[col].std(),
    }

print(f"\nLearned patterns from {len(battery_profiles)} real batteries")

# =================== GENERATE SYNTHETIC BATTERIES ===================
print(f"\n--- Generating {NUM_SYNTHETIC_BATTERIES} synthetic batteries ---")

synthetic_rows = []

for i in range(NUM_SYNTHETIC_BATTERIES):
    bid = f"SYN_{i+1:02d}"
    
    # Pick a random real battery as template
    template_id = np.random.choice(list(battery_profiles.keys()))
    template = battery_profiles[template_id]
    
    # Vary the exponential params slightly
    C0 = template['C0'] + np.random.normal(0, abs(template['C0']) * 0.05)
    a  = template['a']  * np.random.uniform(0.90, 1.10)
    b  = template['b']  * np.random.uniform(0.85, 1.15)
    
    # Random number of cycles
    max_cycle = np.random.randint(CYCLES_RANGE[0], CYCLES_RANGE[1])
    
    # Starting capacity (slightly varied)
    start_cap = np.random.uniform(1.92, 2.00)
    
    # Generate cycle-by-cycle data
    for cycle in range(1, max_cycle + 1):
        frac = cycle / max_cycle  # 0 → 1 (new → old)
        
        # === BCt (capacity) — primary, using exponential model ===
        bct = exp_model(float(cycle), C0, a, b)
        # Add noise
        bct += np.random.normal(0, 0.005)
        # Ensure it starts near start_cap and degrades
        if cycle == 1:
            bct = start_cap
        bct = max(0.5, min(NOMINAL_CAPACITY, bct))
        
        # === Charging Current (chI) — slight increase over life ===
        chI = template['chI_start'] + (template['chI_end'] - template['chI_start']) * frac
        chI += np.random.normal(0, template['chI_std'] * 0.3)
        chI = max(col_ranges['chI']['min'] * 0.9, min(col_ranges['chI']['max'] * 1.1, chI))
        
        # === Charging Voltage (chV) — slight decrease over life ===
        chV = template['chV_start'] + (template['chV_end'] - template['chV_start']) * frac
        chV += np.random.normal(0, template['chV_std'] * 0.3)
        chV = max(col_ranges['chV']['min'] * 0.98, min(col_ranges['chV']['max'] * 1.01, chV))
        
        # === Charging Temp (chT) — slight increase (degradation → more heat) ===
        chT = template['chT_start'] + (template['chT_end'] - template['chT_start']) * frac
        chT += np.random.normal(0, template['chT_std'] * 0.3)
        chT = max(col_ranges['chT']['min'] * 0.95, min(col_ranges['chT']['max'] * 1.05, chT))
        
        # === Discharging Current (disI) — varies over life ===
        disI = template['disI_start'] + (template['disI_end'] - template['disI_start']) * frac
        disI += np.random.normal(0, template['disI_std'] * 0.3)
        disI = max(col_ranges['disI']['min'] * 0.9, min(col_ranges['disI']['max'] * 1.1, disI))
        
        # === Discharging Voltage (disV) — decreases over life (key degradation signal) ===
        disV = template['disV_start'] + (template['disV_end'] - template['disV_start']) * frac
        disV += np.random.normal(0, template['disV_std'] * 0.3)
        disV = max(col_ranges['disV']['min'] * 0.95, min(col_ranges['disV']['max'] * 1.02, disV))
        
        # === Discharging Temp (disT) — increases over life ===
        disT = template['disT_start'] + (template['disT_end'] - template['disT_start']) * frac
        disT += np.random.normal(0, template['disT_std'] * 0.3)
        disT = max(col_ranges['disT']['min'] * 0.95, min(col_ranges['disT']['max'] * 1.05, disT))
        
        # === SOH — Paper Eq. 5: SOH = C_k / C_rated ===
        soh = (bct / NOMINAL_CAPACITY) * 100.0
        
        # === RUL — will be recalculated after all cycles are generated ===
        rul = 0  # placeholder
        
        synthetic_rows.append({
            'battery_id': bid,
            'cycle': cycle,
            'chI': round(chI, 6),
            'chV': round(chV, 6),
            'chT': round(chT, 6),
            'disI': round(disI, 6),
            'disV': round(disV, 6),
            'disT': round(disT, 6),
            'BCt': round(bct, 6),
            'SOH': round(soh, 6),
            'RUL': rul,
        })
    
    print(f"  {bid}: {max_cycle} cycles, template={template_id}, cap={start_cap:.3f}→{bct:.3f} Ah")

df_synth = pd.DataFrame(synthetic_rows)

# =================== CALCULATE RUL FOR SYNTHETIC BATTERIES ===================
print("\n--- Calculating RUL (cycles to SOH < 80%) ---")

for bid in df_synth['battery_id'].unique():
    mask = df_synth['battery_id'] == bid
    sub = df_synth[mask].sort_values('cycle')
    soh_vals = sub['SOH'].values
    cycles = sub['cycle'].values
    
    # Find failure cycle
    fail_mask = soh_vals < 80.0
    if fail_mask.any():
        fail_cycle = cycles[fail_mask][0]
    else:
        # Extrapolate: assume linear decay from last few points
        if len(soh_vals) > 10:
            decay_rate = (soh_vals[-10] - soh_vals[-1]) / 10  # SOH drop per cycle
            if decay_rate > 0:
                cycles_to_fail = (soh_vals[-1] - 80.0) / decay_rate
                fail_cycle = int(cycles[-1] + cycles_to_fail)
            else:
                fail_cycle = int(cycles[-1] + 200)
        else:
            fail_cycle = 500
    
    rul_vals = np.maximum(0, fail_cycle - cycles)
    df_synth.loc[mask, 'RUL'] = rul_vals.astype(int)

# =================== COMBINE ORIGINAL + SYNTHETIC ===================
df_combined = pd.concat([df_orig, df_synth], ignore_index=True)

# Sanity checks
print(f"\n--- Final Dataset ---")
print(f"  Original rows:  {len(df_orig)}")
print(f"  Synthetic rows: {len(df_synth)}")
print(f"  Combined rows:  {len(df_combined)}")
print(f"  Total batteries: {df_combined['battery_id'].nunique()}")
print(f"\n  Per battery:")
for bid in df_combined['battery_id'].unique():
    n = len(df_combined[df_combined['battery_id'] == bid])
    print(f"    {bid}: {n} rows")
print(f"\n  BCt range: {df_combined['BCt'].min():.4f} - {df_combined['BCt'].max():.4f}")
print(f"  SOH range: {df_combined['SOH'].min():.1f}% - {df_combined['SOH'].max():.1f}%")
print(f"  RUL range: {df_combined['RUL'].min()} - {df_combined['RUL'].max()}")

# Save
df_combined.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Augmented dataset saved to: {OUTPUT_PATH}")
print(f"   Backup of original at: {BACKUP_PATH}")
