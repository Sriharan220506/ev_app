"""
EV Battery Analytics - FastAPI Backend with PostgreSQL + XGBoost ML
Deploy to Railway.app for real-time telemetry and ML predictions
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os
import pickle
import numpy as np

from database import get_db, init_db, engine
from models import Base, Vehicle, Telemetry, Prediction, Alert

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EV Battery Analytics API",
    description="Real-time telemetry and XGBoost ML predictions for EV batteries",
    version="3.0.0"
)

# CORS - Allow React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Active Battery Mapping ==============
# Maps Arduino's vehicleId to the battery the user selected in the UI.
# When Arduino sends as "vehicle-001", data is stored under the mapped target.
# e.g. {"vehicle-001": "vehicle-002"} means data goes to Battery 2
active_battery_map = {}

# ============== Load ML Models ==============
# Research paper-based models (Patrizi et al., 2024)
# 12 features: cycle, chI, chV, chT, disI, disV, disT, IR, sqrt_cycle, cycle_sq, cap_residual, temp_diff
# SOH → ExtraTrees (R²=0.9999)
# RUL → RandomForest (R²=0.9995)
# BCT → ExtraTrees (R²=0.9999)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_ml")

soh_model = None
rul_model = None
bct_model = None
feature_scaler = None
exp_params = {"C0": 0.5, "a": 0.5, "b": -30, "nominal_capacity": 2.0, "failure_threshold": 0.8}  # defaults

def single_exp_model(k, C0, a, b):
    """Single exponential degradation: C_k = C0 + a * exp(b/k) [Paper Eq. 4]"""
    return C0 + a * np.exp(b / max(k, 1))

def load_models():
    """Load trained ML models, scaler, and exponential params at startup"""
    global soh_model, rul_model, bct_model, feature_scaler, exp_params
    
    model_files = {
        "soh": "soh_model.pkl",
        "rul": "rul_model.pkl",
        "bct": "bct_model.pkl",
        "scaler": "feature_scaler.pkl",
    }
    
    for key, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if key == "soh":
                soh_model = obj
            elif key == "rul":
                rul_model = obj
            elif key == "bct":
                bct_model = obj
            elif key == "scaler":
                feature_scaler = obj
            print(f"Loaded {key}: {filename}")
        except Exception as e:
            print(f"Warning: {key} model not found: {e}")
    
    # Load exponential degradation params from paper-based training
    exp_path = os.path.join(MODEL_DIR, "exp_params.json")
    try:
        import json
        with open(exp_path, "r") as f:
            exp_params = json.load(f)
        print(f"Loaded exponential params: C0={exp_params['C0']:.4f}, a={exp_params['a']:.4f}, b={exp_params['b']:.2f}")
    except Exception as e:
        print(f"Warning: exp_params not loaded: {e}")

# Load models immediately
load_models()

# ============== Pydantic Models ==============

class TelemetryCreate(BaseModel):
    vehicleId: str
    voltage: float
    current: float
    temperature: float
    internalResistance: float = 0
    soc: float
    power: float = 0
    isCharging: bool = False
    cycleCount: int = 0
    usedCapacity: float = 0
    # Charging values
    chargingVoltage: float = 0
    chargingCurrent: float = 0
    chargingTemp: float = 0
    chargingPower: float = 0
    # Discharging values
    dischargingVoltage: float = 0
    dischargingCurrent: float = 0
    dischargingTemp: float = 0
    dischargingPower: float = 0

class TelemetryResponse(BaseModel):
    id: int
    vehicleId: str
    voltage: float
    current: float
    temperature: float
    internalResistance: float
    soc: float
    power: float
    isCharging: bool
    cycleCount: int
    usedCapacity: float
    chargingVoltage: float
    chargingCurrent: float
    chargingTemp: float
    chargingPower: float
    dischargingVoltage: float
    dischargingCurrent: float
    dischargingTemp: float
    dischargingPower: float
    timestamp: datetime

    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    soh: float
    rul_cycles: int
    rul_months: int
    eul_percentage: float
    confidence: float
    trend: str
    model_type: str
    lastUpdated: str

class HealthResponse(BaseModel):
    overallScore: int
    socHealth: int
    temperatureHealth: int
    resistanceHealth: int
    cycleHealth: int
    efficiencyScore: int
    status: str

# ============== Alert Thresholds ==============
THRESHOLDS = {
    "max_temperature": 45,
    "min_soc": 10,
    "max_resistance": 80,
}

# ============== ML Prediction Functions ==============
#
# RESEARCH PAPER-BASED: Patrizi et al. (2024), Sensors 24, 3382
# Uses single exponential degradation model + 12-feature ML models.
# SOH = C_k / C_rated (Eq. 5)
# RUL = cycles until SOH < 80% (failure threshold)
# Exponential: C_k = C0 + a * exp(b/k) (Eq. 4)
#
# Voltage is STILL the primary signal for real-time health estimation.
# We map voltage to an effective cycle, then compute 12 features.

# Battery voltage thresholds
BATTERY_FULL_VOLTAGE = 3.88   # User specified 100%
BATTERY_DEAD_VOLTAGE = 2.00   # User specified 0%
MAX_CYCLES = 250              # From dataset

# Dataset value ranges (from Battery_dataset.csv) for feature scaling
DATASET_RANGES = {
    "chI":  {"min": 1.0009, "max": 1.7475, "mean": 1.4001},
    "chV":  {"min": 4.0351, "max": 4.3592, "new": 4.35, "old": 4.04},
    "chT":  {"min": 21.60,  "max": 30.91,  "new": 25.0, "old": 30.0},
    "disI": {"min": 1.7024, "max": 2.4112, "mean": 2.0076},
    "disV": {"min": 2.4849, "max": 4.3635, "new": 4.30, "old": 2.50},
    "disT": {"min": 26.85,  "max": 38.39,  "new": 32.0, "old": 38.0},
}

def estimate_effective_cycle(voltage):
    """
    Estimate battery lifecycle position from actual voltage.
    3.88V → cycle 1 (brand new)   |   2.00V → cycle 250 (dead)
    """
    # Clamp voltage to valid range
    v = max(BATTERY_DEAD_VOLTAGE, min(BATTERY_FULL_VOLTAGE, voltage))
    # Linear interpolation
    health_fraction = (v - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)
    effective_cycle = int(1 + (1 - health_fraction) * (MAX_CYCLES - 1))
    return max(1, min(MAX_CYCLES, effective_cycle))

def map_arduino_to_dataset(cycle_count, ch_current, ch_voltage, ch_temp,
                            dis_current, dis_voltage, dis_temp):
    """
    Map Arduino sensor values to dataset-equivalent ranges.
    Scales inputs based on the voltage health fraction (3.88V - 2.0V).
    """
    # Step 1: Determine primary voltage
    primary_voltage = max(ch_voltage, dis_voltage) if dis_voltage > 0.1 else ch_voltage
    if primary_voltage < 1.0:
        primary_voltage = ch_voltage
    
    # Step 2: Estimate effective cycle
    eff_cycle = estimate_effective_cycle(primary_voltage)
    
    if cycle_count > 0:
        mapped_cycle = max(1, min(250, int(0.3 * cycle_count + 0.7 * eff_cycle)))
    else:
        mapped_cycle = eff_cycle
    
    # Step 3: Health fraction
    health = max(0, min(1, (primary_voltage - BATTERY_DEAD_VOLTAGE) / 
                            (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    
    # Step 4: Map features
    mapped_chI = DATASET_RANGES["chI"]["min"] + health * (DATASET_RANGES["chI"]["max"] - DATASET_RANGES["chI"]["min"])
    mapped_chV = DATASET_RANGES["chV"]["old"] + health * (DATASET_RANGES["chV"]["new"] - DATASET_RANGES["chV"]["old"])
    
    mapped_chT = DATASET_RANGES["chT"]["old"] - health * (DATASET_RANGES["chT"]["old"] - DATASET_RANGES["chT"]["new"])
    if 15 <= ch_temp <= 45:
        mapped_chT = np.clip(0.5 * mapped_chT + 0.5 * ch_temp, 
                             DATASET_RANGES["chT"]["min"], DATASET_RANGES["chT"]["max"])
    
    mapped_disI = DATASET_RANGES["disI"]["max"] - health * (DATASET_RANGES["disI"]["max"] - DATASET_RANGES["disI"]["min"])
    
    mapped_disV = np.clip(
        DATASET_RANGES["disV"]["min"] + health * (DATASET_RANGES["disV"]["max"] - DATASET_RANGES["disV"]["min"]),
        DATASET_RANGES["disV"]["min"], DATASET_RANGES["disV"]["max"]
    )
    
    mapped_disT = DATASET_RANGES["disT"]["old"] - health * (DATASET_RANGES["disT"]["old"] - DATASET_RANGES["disT"]["new"])
    if 15 <= dis_temp <= 50:
        mapped_disT = np.clip(0.5 * mapped_disT + 0.5 * dis_temp,
                              DATASET_RANGES["disT"]["min"], DATASET_RANGES["disT"]["max"])
    
    return mapped_cycle, mapped_chI, mapped_chV, mapped_chT, mapped_disI, mapped_disV, mapped_disT

def build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Build the 8-feature array: [cycle, chI, chV, chT, disI, disV, disT, IR]"""
    # Map Arduino values to dataset ranges
    m_cycle, m_chI, m_chV, m_chT, m_disI, m_disV, m_disT = map_arduino_to_dataset(
        cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp
    )
    
    # Compute Internal Resistance (IR) proxy
    ir_val = (m_chV - m_disV) / (m_chI + m_disI) if (m_chI + m_disI) > 0 else 0
    
    # Polynomial features [Paper Eq. 7]
    sqrt_cycle = np.sqrt(m_cycle)
    cycle_sq = m_cycle ** 2
    
    # Exponential model capacity prediction [Paper Eq. 4]
    exp_cap = single_exp_model(float(m_cycle), exp_params['C0'], exp_params['a'], exp_params['b'])
    # Capacity residual: how much actual capacity deviates from exponential prediction
    # For real-time, use health_fraction * nominal_capacity as proxy for actual capacity
    health_frac = max(0, min(1, (max(ch_voltage, dis_voltage) - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    approx_capacity = health_frac * exp_params.get('nominal_capacity', 2.0)
    cap_residual = approx_capacity - exp_cap
    
    # Temperature stress indicator
    temp_diff = m_disT - m_chT
    
    # Coulomb throughput: cumulative charge passed (Cavus et al. 2025)
    coulomb_throughput = m_chI * m_cycle
    
    # 13 features matching training order:
    # cycle, chI, chV, chT, disI, disV, disT, IR, sqrt_cycle, cycle_sq, cap_residual, temp_diff, coulomb_throughput
    raw = np.array([[
        m_cycle, m_chI, m_chV, m_chT, m_disI, m_disV, m_disT,
        ir_val, sqrt_cycle, cycle_sq, cap_residual, temp_diff, coulomb_throughput
    ]])
    
    if feature_scaler is not None:
        return feature_scaler.transform(raw)
    return raw

def predict_soh_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict SOH using model (Ridge/RF)"""
    if soh_model is None:
        return predict_soh_fallback(max(ch_voltage, dis_voltage), cycle, ch_temp)
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        prediction = soh_model.predict(features)[0]
        # Prediction is SOH % (0-100)
        return float(max(0, min(100, prediction)))
    except Exception as e:
        return predict_soh_fallback(max(ch_voltage, dis_voltage), cycle, ch_temp)

def predict_rul_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict RUL (cycles to 80% SOH)"""
    if rul_model is None:
        return predict_rul_fallback(max(ch_voltage, dis_voltage), cycle)
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        prediction = rul_model.predict(features)[0]
        remaining_cycles = int(max(0, prediction))
        months = int(remaining_cycles / 30)
        return {"cycles": remaining_cycles, "months": months}
    except Exception as e:
        return predict_rul_fallback(max(ch_voltage, dis_voltage), cycle)

def predict_bct_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict Battery Capacity (Ah) — based on 1.2Ah nominal"""
    if bct_model is None:
        return predict_bct_fallback(max(ch_voltage, dis_voltage))
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        # Prediction is capacity factor (0-1)
        factor = float(max(0, bct_model.predict(features)[0]))
        return factor * 1.2  # Convert to Ah (1200mAh)
    except Exception as e:
        return predict_bct_fallback(max(ch_voltage, dis_voltage))

# ============== Fallback Predictions (voltage-based) ==============

def predict_soh_fallback(voltage, cycle_count, temperature):
    """Voltage-based SOH fallback — 3.88V=100%, 2.0V=0%"""
    health = max(0, min(1, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    return health * 100.0

def predict_rul_fallback(voltage, cycle_count):
    """Voltage-based RUL fallback"""
    health = max(0, min(1, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    remaining_cycles = int(health * MAX_CYCLES)
    months = int(remaining_cycles / 30)
    return {"cycles": remaining_cycles, "months": months}

def predict_bct_fallback(voltage):
    """Voltage-based BCT fallback (Ah)"""
    health = max(0, min(1, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    return health * 1.2  # 0.0 to 1.2 Ah

def calculate_eul(cycle_count, voltage=None):
    """Calculate Estimated Used Life percentage."""
    if voltage is not None and voltage > 0:
        health = max(0, min(1, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
        return min(100, (1 - health) * 100)
    return min(100, (cycle_count / MAX_CYCLES) * 100)

def calculate_soc_from_voltage(voltage):
    """Calculate SOC from voltage: 3.88V=100%, 2.0V=0%"""
    soc = max(0, min(100, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE) * 100))
    return round(soc, 1)

def detect_battery_condition(voltage, soc, internal_resistance, cycle_count, used_capacity):
    """Detect if the connected battery is NEW or OLD based on voltage-driven SOH"""
    # Primary: use voltage health (3.88V=100%, 2.0V=0%)
    health = max(0, min(1, (voltage - BATTERY_DEAD_VOLTAGE) / (BATTERY_FULL_VOLTAGE - BATTERY_DEAD_VOLTAGE)))
    
    if health >= 0.85:
        return {"condition": "NEW", "score": int(health * 100), "detail": f"Battery is new/healthy ({voltage:.2f}V)"}
    elif health >= 0.60:
        return {"condition": "GOOD", "score": int(health * 100), "detail": f"Battery in good condition ({voltage:.2f}V)"}
    elif health >= 0.35:
        return {"condition": "AGED", "score": int(health * 100), "detail": f"Battery shows wear ({voltage:.2f}V)"}
    else:
        return {"condition": "OLD", "score": int(health * 100), "detail": f"Battery heavily degraded ({voltage:.2f}V)"}

def check_alerts(telemetry: Telemetry, db: Session):
    """Check telemetry against thresholds and create alerts (with deduplication)"""
    alerts_to_create = []
    
    if telemetry.temperature > THRESHOLDS["max_temperature"]:
        alerts_to_create.append({
            "type": "critical",
            "title": "High Temperature Warning",
            "message": f"Battery temperature ({telemetry.temperature:.1f}°C) exceeds safe limit of {THRESHOLDS['max_temperature']}°C",
            "parameter": "temperature",
            "value": telemetry.temperature,
            "threshold": THRESHOLDS["max_temperature"]
        })
    
    if telemetry.soc < THRESHOLDS["min_soc"]:
        alerts_to_create.append({
            "type": "warning",
            "title": "Low Battery",
            "message": f"Battery SOC ({telemetry.soc:.1f}%) is below minimum {THRESHOLDS['min_soc']}%",
            "parameter": "soc",
            "value": telemetry.soc,
            "threshold": THRESHOLDS["min_soc"]
        })
    
    if telemetry.internal_resistance > THRESHOLDS["max_resistance"]:
        alerts_to_create.append({
            "type": "warning",
            "title": "High Internal Resistance",
            "message": f"Internal resistance ({telemetry.internal_resistance:.1f}mΩ) exceeds {THRESHOLDS['max_resistance']}mΩ",
            "parameter": "resistance",
            "value": telemetry.internal_resistance,
            "threshold": THRESHOLDS["max_resistance"]
        })
    
    # Deduplication: don't create same alert type within 5 minutes
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    
    for alert_data in alerts_to_create:
        existing = db.query(Alert).filter(
            Alert.vehicle_id == telemetry.vehicle_id,
            Alert.parameter == alert_data["parameter"],
            Alert.timestamp > five_minutes_ago
        ).first()
        
        if existing is None:
            alert = Alert(vehicle_id=telemetry.vehicle_id, **alert_data)
            db.add(alert)
    
    if alerts_to_create:
        db.commit()

# ============== API Endpoints ==============

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "EV Battery Analytics API",
        "version": "3.0.0",
        "database": "connected",
        "ml_models": {
            "soh": "xgboost" if soh_model else "fallback",
            "rul": "xgboost" if rul_model else "fallback"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": soh_model is not None and rul_model is not None}

@app.on_event("startup")
async def startup():
    init_db()

# ---------- Active Battery Mapping Endpoints ----------

class ActiveBatteryRequest(BaseModel):
    sourceId: str = "vehicle-001"  # Arduino's vehicleId
    targetId: str                   # Battery to store data under

@app.get("/api/active-battery")
def get_active_battery():
    """Get current battery mapping"""
    return {"mapping": active_battery_map, "info": "Arduino data is routed to the mapped target battery"}

@app.post("/api/active-battery")
def set_active_battery(req: ActiveBatteryRequest):
    """Set which battery the Arduino's data should be stored under.
    e.g. sourceId=vehicle-001, targetId=vehicle-002 means Arduino data → Battery 2"""
    active_battery_map[req.sourceId] = req.targetId
    return {
        "status": "ok",
        "message": f"Data from {req.sourceId} will now be stored under {req.targetId}",
        "mapping": active_battery_map
    }

# ---------- Telemetry Endpoints ----------

@app.post("/api/telemetry")
def receive_telemetry(data: TelemetryCreate, db: Session = Depends(get_db)):
    """Receive telemetry from ESP32/Arduino and run ML predictions.
    If active_battery_map has a mapping for this vehicleId, reroute data to the mapped battery."""
    
    # Reroute to active battery if mapping exists
    target_id = active_battery_map.get(data.vehicleId, data.vehicleId)
    
    # Check if vehicle exists, create if not
    vehicle = db.query(Vehicle).filter(Vehicle.id == target_id).first()
    if not vehicle:
        vehicle = Vehicle(id=target_id, name=f"Vehicle {target_id}")
        db.add(vehicle)
    
    # --- Server-side Cycle Counting ---
    # A cycle = full discharge then full charge.
    # Detect transition: previous record was DISCHARGING, now CHARGING.
    prev = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == target_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    if prev:
        server_cycle_count = prev.cycle_count or 0
        # Transition: was discharging (is_charging=False), now charging (is_charging=True)
        if not prev.is_charging and data.isCharging:
            server_cycle_count += 1
    else:
        server_cycle_count = data.cycleCount or 0
    
    # --- Compute SOC from Voltage (server-side) ---
    primary_v = max(data.voltage, data.chargingVoltage, data.dischargingVoltage)
    computed_soc = calculate_soc_from_voltage(primary_v)
    
    # Create telemetry record (use server-computed values, stored under target battery)
    telemetry = Telemetry(
        vehicle_id=target_id,
        voltage=data.voltage,
        current=data.current,
        temperature=data.temperature,
        internal_resistance=data.internalResistance,
        soc=computed_soc,
        power=data.power,
        is_charging=data.isCharging,
        cycle_count=server_cycle_count,
        used_capacity=data.usedCapacity,
        charging_voltage=data.chargingVoltage,
        charging_current=data.chargingCurrent,
        charging_temp=data.chargingTemp,
        charging_power=data.chargingPower,
        discharging_voltage=data.dischargingVoltage,
        discharging_current=data.dischargingCurrent,
        discharging_temp=data.dischargingTemp,
        discharging_power=data.dischargingPower
    )
    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)
    
    # Check for alerts
    check_alerts(telemetry, db)
    
    # ML Predictions using 7 features: [cycle, chI, chV, chT, disI, disV, disT]
    soh = predict_soh_ml(
        data.cycleCount,
        data.chargingCurrent, data.chargingVoltage, data.chargingTemp,
        data.dischargingCurrent, data.dischargingVoltage, data.dischargingTemp
    )
    rul = predict_rul_ml(
        data.cycleCount,
        data.chargingCurrent, data.chargingVoltage, data.chargingTemp,
        data.dischargingCurrent, data.dischargingVoltage, data.dischargingTemp
    )
    bct = predict_bct_ml(
        data.cycleCount,
        data.chargingCurrent, data.chargingVoltage, data.chargingTemp,
        data.dischargingCurrent, data.dischargingVoltage, data.dischargingTemp
    )
    eul = calculate_eul(data.cycleCount, max(data.chargingVoltage, data.dischargingVoltage))
    
    model_type = "ml-trained" if soh_model else "rule-based"
    confidence = 0.95 if soh_model else 0.70
    
    prediction = Prediction(
        vehicle_id=target_id,
        soh=soh,
        rul_cycles=rul["cycles"],
        rul_months=rul["months"],
        eul_percentage=eul,
        trend="declining" if soh < 85 else "stable",
        confidence=confidence
    )
    db.add(prediction)
    db.commit()
    
    return {
        "status": "ok",
        "message": "Telemetry saved",
        "id": telemetry.id,
        "predictions": {
            "soh": round(soh, 2),
            "rul_cycles": rul["cycles"],
            "rul_months": rul["months"],
            "bct": round(bct, 4),
            "model": model_type
        }
    }

@app.get("/api/telemetry/{vehicle_id}")
def get_telemetry(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest telemetry for a vehicle.
    If the vehicle has no data yet (e.g. user just switched batteries),
    fall back to vehicle-001 data (the Arduino's source) so the dashboard shows live readings."""
    
    telemetry = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    # Fallback: if this battery has no data yet, use vehicle-001 (Arduino source)
    fallback_used = False
    if not telemetry and vehicle_id != "vehicle-001":
        telemetry = db.query(Telemetry)\
            .filter(Telemetry.vehicle_id == "vehicle-001")\
            .order_by(desc(Telemetry.timestamp))\
            .first()
        fallback_used = True
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="No telemetry data available")
    
    # Detect battery condition (rule-based)
    battery_condition = detect_battery_condition(
        telemetry.voltage, telemetry.soc,
        telemetry.internal_resistance, telemetry.cycle_count,
        telemetry.used_capacity
    )
    
    # ML BCT prediction
    bct = predict_bct_ml(
        telemetry.cycle_count,
        telemetry.charging_current or 0, telemetry.charging_voltage or 0, telemetry.charging_temp or 0,
        telemetry.discharging_current or 0, telemetry.discharging_voltage or 0, telemetry.discharging_temp or 0
    )
    
    return {
        "vehicleId": vehicle_id,  # Always return the requested battery ID
        "voltage": telemetry.voltage,
        "current": telemetry.current,
        "temperature": telemetry.temperature,
        "internalResistance": telemetry.internal_resistance,
        "soc": telemetry.soc,
        "power": telemetry.power,
        "isCharging": telemetry.is_charging,
        "cycleCount": telemetry.cycle_count,
        "usedCapacity": telemetry.used_capacity,
        "chargingVoltage": telemetry.charging_voltage,
        "chargingCurrent": telemetry.charging_current,
        "chargingTemp": telemetry.charging_temp,
        "chargingPower": telemetry.charging_power,
        "dischargingVoltage": telemetry.discharging_voltage,
        "dischargingCurrent": telemetry.discharging_current,
        "dischargingTemp": telemetry.discharging_temp,
        "dischargingPower": telemetry.discharging_power,
        "batteryCondition": battery_condition,
        "bct": round(bct, 4),
        "timestamp": telemetry.timestamp.isoformat(),
        "fallback": fallback_used  # True if showing vehicle-001 data temporarily
    }

@app.get("/api/telemetry/{vehicle_id}/history")
def get_telemetry_history(vehicle_id: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get telemetry history for analytics"""
    
    records = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .limit(limit)\
        .all()
    
    return [{
        "voltage": r.voltage,
        "current": r.current,
        "temperature": r.temperature,
        "soc": r.soc,
        "power": r.power,
        "internalResistance": r.internal_resistance,
        "usedCapacity": r.used_capacity,
        "cycleCount": r.cycle_count,
        "isCharging": r.is_charging,
        "chargingVoltage": r.charging_voltage,
        "chargingCurrent": r.charging_current,
        "chargingTemp": r.charging_temp,
        "chargingPower": r.charging_power,
        "dischargingVoltage": r.discharging_voltage,
        "dischargingCurrent": r.discharging_current,
        "dischargingTemp": r.discharging_temp,
        "dischargingPower": r.discharging_power,
        "timestamp": r.timestamp.isoformat()
    } for r in reversed(records)]

# ---------- Prediction Endpoints ----------

@app.get("/api/predict/soh/{vehicle_id}")
def get_soh_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest SOH prediction from XGBoost model"""
    
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(desc(Prediction.timestamp))\
        .first()
    
    # Fallback to vehicle-001 if no predictions for this battery yet
    if not prediction and vehicle_id != "vehicle-001":
        prediction = db.query(Prediction)\
            .filter(Prediction.vehicle_id == "vehicle-001")\
            .order_by(desc(Prediction.timestamp))\
            .first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    return {
        "value": prediction.soh,
        "confidence": prediction.confidence,
        "trend": prediction.trend,
        "model": "xgboost" if soh_model else "rule-based",
        "lastUpdated": prediction.timestamp.isoformat()
    }

@app.get("/api/predict/rul/{vehicle_id}")
def get_rul_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest RUL prediction from XGBoost model"""
    
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(desc(Prediction.timestamp))\
        .first()
    
    if not prediction and vehicle_id != "vehicle-001":
        prediction = db.query(Prediction)\
            .filter(Prediction.vehicle_id == "vehicle-001")\
            .order_by(desc(Prediction.timestamp))\
            .first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    return {
        "cycles": prediction.rul_cycles,
        "months": prediction.rul_months,
        "confidence": prediction.confidence,
        "trend": prediction.trend,
        "model": "xgboost" if rul_model else "rule-based",
        "lastUpdated": prediction.timestamp.isoformat()
    }

@app.get("/api/predict/eul/{vehicle_id}")
def get_eul_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest EUL prediction"""
    
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(desc(Prediction.timestamp))\
        .first()
    
    if not prediction and vehicle_id != "vehicle-001":
        prediction = db.query(Prediction)\
            .filter(Prediction.vehicle_id == "vehicle-001")\
            .order_by(desc(Prediction.timestamp))\
            .first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    return {
        "percentage": prediction.eul_percentage,
        "cycles": prediction.rul_cycles,
        "months": prediction.rul_months,
        "lastUpdated": prediction.timestamp.isoformat()
    }

@app.get("/api/predict/all/{vehicle_id}")
def get_all_predictions(vehicle_id: str, db: Session = Depends(get_db)):
    """Get all predictions in one call"""
    
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(desc(Prediction.timestamp))\
        .first()
    
    if not prediction and vehicle_id != "vehicle-001":
        prediction = db.query(Prediction)\
            .filter(Prediction.vehicle_id == "vehicle-001")\
            .order_by(desc(Prediction.timestamp))\
            .first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    return {
        "soh": {
            "value": prediction.soh,
            "confidence": prediction.confidence,
            "trend": prediction.trend,
            "model": "xgboost" if soh_model else "rule-based"
        },
        "rul": {
            "cycles": prediction.rul_cycles,
            "months": prediction.rul_months,
            "confidence": prediction.confidence,
            "model": "xgboost" if rul_model else "rule-based"
        },
        "eul": {
            "percentage": prediction.eul_percentage
        },
        "lastUpdated": prediction.timestamp.isoformat()
    }

# ---------- Health Endpoints ----------

@app.get("/api/health/{vehicle_id}")
def get_health_metrics(vehicle_id: str, db: Session = Depends(get_db)):
    """Get health metrics from latest telemetry + ML predictions"""
    
    telemetry = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    # Fallback to vehicle-001 if no telemetry for this battery yet
    if not telemetry and vehicle_id != "vehicle-001":
        telemetry = db.query(Telemetry)\
            .filter(Telemetry.vehicle_id == "vehicle-001")\
            .order_by(desc(Telemetry.timestamp))\
            .first()
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Get latest ML prediction for SOH-based health (also with fallback)
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(desc(Prediction.timestamp))\
        .first()
    
    if not prediction and vehicle_id != "vehicle-001":
        prediction = db.query(Prediction)\
            .filter(Prediction.vehicle_id == "vehicle-001")\
            .order_by(desc(Prediction.timestamp))\
            .first()
    
    # Calculate health scores
    soc_health = 100 if 20 <= telemetry.soc <= 80 else 80
    temp_health = 100 if 15 <= telemetry.temperature <= 35 else (85 if 10 <= telemetry.temperature <= 45 else 60)
    res_health = 100 if telemetry.internal_resistance < 40 else (80 if telemetry.internal_resistance < 60 else 60)
    cycle_health = int(max(0, 100 - (telemetry.cycle_count * 0.03)))
    
    # Use ML SOH as overall score if available
    if prediction:
        overall = int(prediction.soh)
    else:
        overall = int(soc_health * 0.2 + temp_health * 0.2 + res_health * 0.3 + cycle_health * 0.3)
    
    efficiency = int((soc_health + temp_health + res_health) / 3)
    
    status = "healthy" if overall >= 80 else ("warning" if overall >= 60 else "critical")
    
    return {
        "overallScore": overall,
        "socHealth": soc_health,
        "temperatureHealth": temp_health,
        "resistanceHealth": res_health,
        "cycleHealth": cycle_health,
        "efficiencyScore": efficiency,
        "status": status,
        "mlModel": "xgboost" if soh_model else "rule-based"
    }

# ---------- Alerts Endpoints ----------

@app.get("/api/alerts/{vehicle_id}")
def get_alerts(vehicle_id: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get alerts for a vehicle"""
    
    alerts = db.query(Alert)\
        .filter(Alert.vehicle_id == vehicle_id)\
        .order_by(desc(Alert.timestamp))\
        .limit(limit)\
        .all()
    
    return [{
        "id": a.id,
        "type": a.type,
        "title": a.title,
        "message": a.message,
        "parameter": a.parameter,
        "value": a.value,
        "threshold": a.threshold,
        "read": a.read,
        "timestamp": a.timestamp.isoformat()
    } for a in alerts]

@app.put("/api/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as read"""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.read = True
    db.commit()
    
    return {"status": "ok"}

# ---------- Vehicles Endpoint ----------

@app.get("/api/vehicles")
def list_vehicles(db: Session = Depends(get_db)):
    """List all registered vehicles"""
    
    vehicles = db.query(Vehicle).all()
    
    result = []
    for v in vehicles:
        latest = db.query(Telemetry)\
            .filter(Telemetry.vehicle_id == v.id)\
            .order_by(desc(Telemetry.timestamp))\
            .first()
        
        result.append({
            "id": v.id,
            "name": v.name,
            "batteryType": v.battery_type,
            "lastSeen": latest.timestamp.isoformat() if latest else None,
            "soc": latest.soc if latest else None,
            "isCharging": latest.is_charging if latest else None
        })
    
    return result

# ---------- Reset Endpoint ----------

@app.delete("/api/reset/{vehicle_id}")
def reset_vehicle_data(vehicle_id: str, db: Session = Depends(get_db)):
    """Reset all data for a vehicle — manual restart"""
    
    # Delete in order: alerts, predictions, telemetry
    deleted_alerts = db.query(Alert).filter(Alert.vehicle_id == vehicle_id).delete()
    deleted_predictions = db.query(Prediction).filter(Prediction.vehicle_id == vehicle_id).delete()
    deleted_telemetry = db.query(Telemetry).filter(Telemetry.vehicle_id == vehicle_id).delete()
    
    db.commit()
    
    return {
        "status": "ok",
        "message": f"All data reset for {vehicle_id}",
        "deleted": {
            "telemetry": deleted_telemetry,
            "predictions": deleted_predictions,
            "alerts": deleted_alerts
        }
    }

@app.delete("/api/alerts/{vehicle_id}/clear")
def clear_vehicle_alerts(vehicle_id: str, db: Session = Depends(get_db)):
    """Clear all alerts for a vehicle"""
    
    deleted = db.query(Alert).filter(Alert.vehicle_id == vehicle_id).delete()
    db.commit()
    
    return {"status": "ok", "deleted": deleted}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
