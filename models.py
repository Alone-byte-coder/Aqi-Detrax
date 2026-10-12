
from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Text

from database import Base


class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100), nullable=False)
    recorded_at = Column("timestamp", DateTime, nullable=False, index=True)

    pm25 = Column(Float, nullable=False)
    pm10 = Column(Float, nullable=False)

    temperature = Column("temperature_c", Float, nullable=True)
    humidity = Column("humidity_pct", Float, nullable=True)
    pressure = Column("pressure_hpa", Float, nullable=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    gps_fix = Column(Boolean, nullable=True)


class CurrentAQIResult(Base):
    __tablename__ = "current_aqi_results"

    id = Column(Integer, primary_key=True, index=True)
    sensor_data_id = Column(
        Integer, ForeignKey("sensor_data.id"), nullable=False
    )

    final_aqi = Column("aqi", Float, nullable=True)
    category = Column(String(50), nullable=True)
    risk_level = Column(String(30), nullable=True)
    risk_percentage = Column(Float, nullable=True)
    advice = Column(Text, nullable=True)

    calculated_at = Column("processed_at", DateTime, nullable=False)


class PredictedData(Base):
    __tablename__ = "predicted_data"

    id = Column(Integer, primary_key=True, index=True)

    source_sensor_data_id = Column(
        "sensor_data_id",
        Integer,
        ForeignKey("sensor_data.id"),
        nullable=False,
    )

    future_pm25 = Column("predicted_pm25", Float, nullable=True)
    future_pm10 = Column("predicted_pm10", Float, nullable=True)
    forecast_for = Column(DateTime, nullable=True)
    model_status = Column(String(30), nullable=False)
    predicted_at = Column("processed_at", DateTime, nullable=False)


class FutureAQIResult(Base):
    __tablename__ = "future_aqi_results"

    id = Column(Integer, primary_key=True, index=True)

    predicted_data_id = Column(
        Integer, ForeignKey("predicted_data.id"), nullable=False
    )

    final_aqi = Column("future_aqi", Float, nullable=True)
    category = Column(String(50), nullable=True)
    risk_level = Column(String(30), nullable=True)
    risk_percentage = Column(Float, nullable=True)
    precautions = Column(Text, nullable=True)

    calculated_at = Column("processed_at", DateTime, nullable=False)
