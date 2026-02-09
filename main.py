"""
EV Battery Analytics - FastAPI Backend with PostgreSQL
Deploy to Railway.app for real-time telemetry and predictions
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os

from database import get_db, init_db, engine
from models import Base, Vehicle, Telemetry, Prediction, Alert

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EV Battery Analytics API",
    description="Real-time telemetry and ML predictions for EV batteries",
    version="2.0.0"
)

# CORS - Allow React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ============== Prediction Functions ==============

def calculate_soh(telemetry: Telemetry) -> float:
    """Calculate State of Health"""
    resistance_factor = max(0, 100 - (telemetry.internal_resistance - 30) * 0.5)
    cycle_factor = max(0, 100 - (telemetry.cycle_count * 0.02))
    temp_factor = 100 if 15 <= telemetry.temperature <= 35 else 90
    
    soh = (resistance_factor * 0.4 + cycle_factor * 0.4 + temp_factor * 0.2)
    return max(0, min(100, soh))

def calculate_rul(soh: float, cycle_count: int) -> dict:
    """Calculate Remaining Useful Life"""
    max_cycles = 2500
    remaining_cycles = int((soh / 100) * max_cycles - cycle_count)
    remaining_cycles = max(0, remaining_cycles)
    months = int(remaining_cycles / 30)
    
    return {"cycles": remaining_cycles, "months": months}

def calculate_eul(cycle_count: int) -> float:
    """Calculate Estimated Used Life percentage"""
    max_cycles = 2500
    return min(100, (cycle_count / max_cycles) * 100)

def check_alerts(telemetry: Telemetry, db: Session):
    """Check telemetry against thresholds and create alerts"""
    alerts_to_create = []
    
    # Temperature alert
    if telemetry.temperature > THRESHOLDS["max_temperature"]:
        alerts_to_create.append({
            "type": "critical",
            "title": "High Temperature Warning",
            "message": f"Battery temperature ({telemetry.temperature}°C) exceeds safe limit",
            "parameter": "temperature",
            "value": telemetry.temperature,
            "threshold": THRESHOLDS["max_temperature"]
        })
    
    # Low SOC alert
    if telemetry.soc < THRESHOLDS["min_soc"]:
        alerts_to_create.append({
            "type": "warning",
            "title": "Low Battery",
            "message": f"Battery SOC ({telemetry.soc}%) is critically low",
            "parameter": "soc",
            "value": telemetry.soc,
            "threshold": THRESHOLDS["min_soc"]
        })
    
    # High resistance alert
    if telemetry.internal_resistance > THRESHOLDS["max_resistance"]:
        alerts_to_create.append({
            "type": "warning",
            "title": "High Internal Resistance",
            "message": f"Internal resistance ({telemetry.internal_resistance}mΩ) indicates degradation",
            "parameter": "resistance",
            "value": telemetry.internal_resistance,
            "threshold": THRESHOLDS["max_resistance"]
        })
    
    # Create alerts in database
    for alert_data in alerts_to_create:
        alert = Alert(vehicle_id=telemetry.vehicle_id, **alert_data)
        db.add(alert)
    
    if alerts_to_create:
        db.commit()

# ============== API Endpoints ==============

@app.get("/")
def root():
    return {"status": "ok", "message": "EV Battery Analytics API", "version": "2.0.0", "database": "connected"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup():
    init_db()

# ---------- Telemetry Endpoints ----------

@app.post("/api/telemetry")
def receive_telemetry(data: TelemetryCreate, db: Session = Depends(get_db)):
    """Receive telemetry from ESP32/Arduino"""
    
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
        cycle_count=data.cycleCount
    )
    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)
    
    # Check for alerts
    check_alerts(telemetry, db)
    
    # Calculate and store predictions
    soh = calculate_soh(telemetry)
    rul = calculate_rul(soh, data.cycleCount)
    eul = calculate_eul(data.cycleCount)
    
    prediction = Prediction(
        vehicle_id=data.vehicleId,
        soh=soh,
        rul_cycles=rul["cycles"],
        rul_months=rul["months"],
        eul_percentage=eul,
        trend="declining" if soh < 85 else "stable"
    )
    db.add(prediction)
    db.commit()
    
    return {"status": "ok", "message": "Telemetry saved", "id": telemetry.id}

@app.get("/api/telemetry/{vehicle_id}")
def get_telemetry(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest telemetry for a vehicle"""
    
    telemetry = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
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
        "timestamp": r.timestamp.isoformat()
    } for r in reversed(records)]

# ---------- Prediction Endpoints ----------

@app.get("/api/predict/soh/{vehicle_id}")
def get_soh_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest SOH prediction"""
    
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
        "lastUpdated": prediction.timestamp.isoformat()
    }

@app.get("/api/predict/rul/{vehicle_id}")
def get_rul_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """Get latest RUL prediction"""
    
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

# ---------- Health Endpoints ----------

@app.get("/api/health/{vehicle_id}")
def get_health_metrics(vehicle_id: str, db: Session = Depends(get_db)):
    """Get health metrics calculated from latest telemetry"""
    
    telemetry = db.query(Telemetry)\
        .filter(Telemetry.vehicle_id == vehicle_id)\
        .order_by(desc(Telemetry.timestamp))\
        .first()
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Calculate health scores
    soc_health = 100 if 20 <= telemetry.soc <= 80 else 80
    temp_health = 100 if 15 <= telemetry.temperature <= 35 else (85 if 10 <= telemetry.temperature <= 45 else 60)
    res_health = 100 if telemetry.internal_resistance < 40 else (80 if telemetry.internal_resistance < 60 else 60)
    cycle_health = int(max(0, 100 - (telemetry.cycle_count * 0.03)))
    
    efficiency = int((soc_health + temp_health + res_health) / 3)
    overall = int(soc_health * 0.2 + temp_health * 0.2 + res_health * 0.3 + cycle_health * 0.3)
    
    status = "healthy" if overall >= 80 else ("warning" if overall >= 60 else "critical")
    
    return {
        "overallScore": overall,
        "socHealth": soc_health,
        "temperatureHealth": temp_health,
        "resistanceHealth": res_health,
        "cycleHealth": cycle_health,
        "efficiencyScore": efficiency,
        "status": status
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
        # Get latest telemetry
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
