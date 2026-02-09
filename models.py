"""
Database Models for EV Battery Analytics
Uses SQLAlchemy with PostgreSQL on Railway
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Vehicle(Base):
    """Vehicle registration table"""
    __tablename__ = "vehicles"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    battery_capacity = Column(Float, default=60.0)  # kWh
    battery_type = Column(String(50), default="li-ion")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Telemetry(Base):
    """Real-time telemetry data from ESP32"""
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    
    # Sensor readings
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    internal_resistance = Column(Float, default=0)
    soc = Column(Float, nullable=False)
    power = Column(Float, default=0)
    is_charging = Column(Boolean, default=False)
    cycle_count = Column(Integer, default=0)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Prediction(Base):
    """ML predictions for battery health"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    
    # Predictions
    soh = Column(Float)  # State of Health %
    rul_cycles = Column(Integer)  # Remaining cycles
    rul_months = Column(Integer)  # Remaining months
    eul_percentage = Column(Float)  # Used life %
    
    # Confidence and trend
    confidence = Column(Float, default=0.85)
    trend = Column(String(20), default="stable")
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    """System alerts and warnings"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    
    type = Column(String(20), nullable=False)  # critical, warning, info
    title = Column(String(200), nullable=False)
    message = Column(Text)
    parameter = Column(String(50))  # temperature, soc, resistance, etc.
    value = Column(Float)
    threshold = Column(Float)
    
    read = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
