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

# ============== Load ML Models ==============
# Models trained on 7 features: [cycle, chI, chV, chT, disI, disV, disT]
# SOH → GradientBoosting (R²=0.9997)
# RUL → RandomForest (R²=0.9055)
# BCT → GradientBoosting (R²=0.9997)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_ml")

soh_model = None
rul_model = None
bct_model = None
feature_scaler = None

def load_models():
    """Load trained ML models and scaler at startup"""
    global soh_model, rul_model, bct_model, feature_scaler
    
    model_files = {
        "soh": "soh_model_xgboost.pkl",
        "rul": "rul_model_xgboost.pkl",
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
# Features order: [cycle, chI, chV, chT, disI, disV, disT]

def build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Build the 7-feature array matching dataset training order"""
    raw = np.array([[cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp]])
    if feature_scaler is not None:
        return feature_scaler.transform(raw)
    return raw

def predict_soh_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict SOH using GradientBoosting model (R²=0.9997)"""
    if soh_model is None:
        return predict_soh_fallback(ch_voltage, cycle, ch_temp)
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        prediction = soh_model.predict(features)[0]
        return float(max(0, min(100, prediction)))
    except Exception as e:
        print(f"SOH ML error: {e}")
        return predict_soh_fallback(ch_voltage, cycle, ch_temp)

def predict_rul_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict RUL using RandomForest model (R²=0.9055)"""
    if rul_model is None:
        return predict_rul_fallback(100, cycle)
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        prediction = rul_model.predict(features)[0]
        remaining_cycles = int(max(0, prediction))
        months = int(remaining_cycles / 30)
        return {"cycles": remaining_cycles, "months": months}
    except Exception as e:
        print(f"RUL ML error: {e}")
        return predict_rul_fallback(100, cycle)

def predict_bct_ml(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp):
    """Predict Battery Condition (BCT) using GradientBoosting model (R²=0.9997)"""
    if bct_model is None:
        return 1.0  # default mid-range
    try:
        features = build_features(cycle, ch_current, ch_voltage, ch_temp, dis_current, dis_voltage, dis_temp)
        prediction = bct_model.predict(features)[0]
        return float(max(0, prediction))
    except Exception as e:
        print(f"BCT ML error: {e}")
        return 1.0

# ============== Fallback Predictions ==============

def predict_soh_fallback(voltage, cycle_count, temperature):
    """Rule-based SOH fallback"""
    voltage_factor = min(100, max(0, (voltage - 3.0) / 1.2 * 100))
    cycle_factor = max(0, 100 - (cycle_count * 0.2))
    temp_factor = 100 if 15 <= temperature <= 35 else 90
    soh = (voltage_factor * 0.3 + cycle_factor * 0.5 + temp_factor * 0.2)
    return max(0, min(100, soh))

def predict_rul_fallback(soh, cycle_count):
    """Rule-based RUL fallback"""
    max_cycles = 250
    remaining_cycles = int((soh / 100) * max_cycles - cycle_count)
    remaining_cycles = max(0, remaining_cycles)
    months = int(remaining_cycles / 30)
    return {"cycles": remaining_cycles, "months": months}

def calculate_eul(cycle_count):
    """Calculate Estimated Used Life percentage"""
    max_cycles = 250
    return min(100, (cycle_count / max_cycles) * 100)

def detect_battery_condition(voltage, soc, internal_resistance, cycle_count, used_capacity):
    """Detect if the connected battery is NEW or OLD based on sensor values"""
    score = 0
    
    # High voltage = new battery (>= 3.9V for li-ion 3.7V cell)
    if voltage >= 3.9:
        score += 25
    elif voltage >= 3.6:
        score += 15
    else:
        score += 5
    
    # Low internal resistance = new battery (< 30 mΩ)
    if internal_resistance < 30:
        score += 25
    elif internal_resistance < 50:
        score += 15
    else:
        score += 0
    
    # Low cycle count = new battery
    if cycle_count <= 5:
        score += 25
    elif cycle_count <= 50:
        score += 15
    else:
        score += 5
    
    # Low used capacity = new battery
    if used_capacity < 0.5:
        score += 25
    elif used_capacity < 5:
        score += 15
    else:
        score += 5
    
    if score >= 80:
        return {"condition": "NEW", "score": score, "detail": "Battery appears to be new or near-new"}
    elif score >= 50:
        return {"condition": "GOOD", "score": score, "detail": "Battery is in good used condition"}
    elif score >= 30:
        return {"condition": "AGED", "score": score, "detail": "Battery shows significant wear"}
    else:
        return {"condition": "OLD", "score": score, "detail": "Battery is heavily degraded"}

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

# ---------- Telemetry Endpoints ----------

@app.post("/api/telemetry")
def receive_telemetry(data: TelemetryCreate, db: Session = Depends(get_db)):
    """Receive telemetry from ESP32/Arduino and run ML predictions"""
    
    # Check if vehicle exists, create if not
    vehicle = db.query(Vehicle).filter(Vehicle.id == data.vehicleId).first()
    if not vehicle:
        vehicle = Vehicle(id=data.vehicleId, name=f"Vehicle {data.vehicleId}")
        db.add(vehicle)
    
    # Create telemetry record
    telemetry = Telemetry(
        vehicle_id=data.vehicleId,
        voltage=data.voltage,
        current=data.current,
        temperature=data.temperature,
        internal_resistance=data.internalResistance,
        soc=data.soc,
        power=data.power,
        is_charging=data.isCharging,
        cycle_count=data.cycleCount,
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
    eul = calculate_eul(data.cycleCount)
    
    model_type = "ml-trained" if soh_model else "rule-based"
    confidence = 0.95 if soh_model else 0.70
    
    prediction = Prediction(
        vehicle_id=data.vehicleId,
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
    """Get latest telemetry for a vehicle"""
    
    telemetry = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
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
        "vehicleId": telemetry.vehicle_id,
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
        "timestamp": telemetry.timestamp.isoformat()
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
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Get latest ML prediction for SOH-based health
    prediction = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
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
