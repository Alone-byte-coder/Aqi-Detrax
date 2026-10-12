
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import (
    SensorData,
    CurrentAQIResult,
    PredictedData,
    FutureAQIResult,
)
from schemas import SensorReading
from aqi import calculate_aqi
from forecasting import predict_future

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Real-Time AQI Monitoring API")


def get_risk_percentage(aqi_value):
    # Simple normalized AQI scale; this is not a medical risk probability.
    return round(max(0.0, min(100.0, aqi_value / 500 * 100)), 2)


def get_advice(category):
    advice = {
        "Good": "Air quality is satisfactory. Continue normal activities.",
        "Satisfactory": "Sensitive people should monitor their symptoms.",
        "Moderate": "Reduce prolonged outdoor exertion if sensitive.",
        "Poor": "Limit prolonged outdoor activities.",
        "Very Poor": "Avoid strenuous outdoor activities.",
        "Severe": "Avoid outdoor exertion and follow local health advisories.",
    }
    return advice.get(category, "Monitor air quality and follow local advisories.")


@app.get("/")
def home():
    return {"message": "AQI backend is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/api/sensors/readings")
def receive_sensor_data(
    payload: SensorReading,
    db: Session = Depends(get_db),
):
    now = datetime.utcnow()

    try:
        reading = SensorData(
            device_id=payload.device_id,
            recorded_at=payload.recorded_at or now,
            pm25=payload.pm25,
            pm10=payload.pm10,
            temperature=payload.temperature,
            humidity=payload.humidity,
            pressure=payload.pressure,
            latitude=payload.latitude,
            longitude=payload.longitude,
            gps_fix=payload.gps_fix,
        )

        db.add(reading)
        db.flush()

        current = calculate_aqi(reading.pm25, reading.pm10)
        risk_percentage = get_risk_percentage(current["final_aqi"])

        current_result = CurrentAQIResult(
            sensor_data_id=reading.id,
            final_aqi=current["final_aqi"],
            category=current["category"],
            risk_level=current["category"],
            risk_percentage=risk_percentage,
            advice=get_advice(current["category"]),
            calculated_at=now,
        )

        db.add(current_result)
        db.flush()

        recent = (
            db.query(SensorData)
            .order_by(SensorData.recorded_at.desc(), SensorData.id.desc())
            .limit(4)
            .all()
        )

        forecast_response = {
            "available": False,
            "message": "Waiting for four sensor readings.",
        }

        
        if len(recent) >= 4:
            # The query returns newest first, so reverse it.
            # The model receives readings from oldest to newest.
            readings = list(reversed(recent))

            try:
                future_pm25, future_pm10 = predict_future(readings)
                future = calculate_aqi(future_pm25, future_pm10)
                future_risk = get_risk_percentage(future["final_aqi"])

            except Exception as exc:
                print("FORECAST MODEL ERROR:", repr(exc))
                forecast_response = {
                    "available": False,
                    "message": f"Forecast failed: {type(exc).__name__}: {exc}",
                }

            else:
                # Save prediction only when model execution succeeds.
                predicted = PredictedData(
                    source_sensor_data_id=reading.id,
                    future_pm25=future_pm25,
                    future_pm10=future_pm10,
                    model_status="success",
                    predicted_at=now,
                )

                db.add(predicted)
                db.flush()

                future_result = FutureAQIResult(
                    predicted_data_id=predicted.id,
                    final_aqi=future["final_aqi"],
                    category=future["category"],
                    risk_level=future["category"],
                    risk_percentage=future_risk,
                    precautions=get_advice(future["category"]),
                    calculated_at=now,
                )

                db.add(future_result)
                db.flush()

                forecast_response = {
                    "available": True,
                    "predicted_data_id": predicted.id,
                    "future_pm25": future_pm25,
                    "future_pm10": future_pm10,
                    "future_aqi": future["final_aqi"],
                    "future_category": future["category"],
                    "risk_percentage": future_risk,
                    "horizon_steps": 6,
                    "model_status": "success",
                }

        else:
            print(f"FORECAST WAITING: only {len(recent)} readings available.")


        db.commit()

        return {
            "message": "Sensor data saved successfully",
            "sensor_data_id": reading.id,
            "current_aqi": {
                "final_aqi": current["final_aqi"],
                "category": current["category"],
                "pm25_sub_index": current["pm25_sub_index"],
                "pm10_sub_index": current["pm10_sub_index"],
                "risk_percentage": risk_percentage,
                "risk_level": current["category"],
                "advice": get_advice(current["category"]),
            },
            "forecast": forecast_response,
        }

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not process sensor data: {exc}",
        ) from exc


@app.get("/api/sensors/latest")
def latest_sensor(db: Session = Depends(get_db)):
    reading = (
        db.query(SensorData)
        .order_by(SensorData.recorded_at.desc(), SensorData.id.desc())
        .first()
    )

    if reading is None:
        raise HTTPException(status_code=404, detail="No sensor readings found.")

    return {
        "id": reading.id,
        "device_id": reading.device_id,
        "recorded_at": reading.recorded_at,
        "pm25": reading.pm25,
        "pm10": reading.pm10,
        "temperature": reading.temperature,
        "humidity": reading.humidity,
        "pressure": reading.pressure,
        "latitude": reading.latitude,
        "longitude": reading.longitude,
        "gps_fix": reading.gps_fix,
    }


@app.get("/api/aqi/current")
def latest_aqi(db: Session = Depends(get_db)):
    result = (
        db.query(CurrentAQIResult)
        .order_by(CurrentAQIResult.id.desc())
        .first()
    )

    if result is None:
        raise HTTPException(status_code=404, detail="No AQI results found.")

    return {
        "sensor_data_id": result.sensor_data_id,
        "final_aqi": result.final_aqi,
        "category": result.category,
        "risk_level": result.risk_level,
        "risk_percentage": result.risk_percentage,
        "advice": result.advice,
    }


@app.get("/api/forecast/latest")
def latest_forecast(db: Session = Depends(get_db)):
    result = (
        db.query(FutureAQIResult)
        .order_by(FutureAQIResult.id.desc())
        .first()
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No future AQI results found.",
        )

    predicted = db.query(PredictedData).filter(
        PredictedData.id == result.predicted_data_id
    ).first()

    if predicted is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found.",
        )

    return {
        "predicted_data_id": predicted.id,
        "future_pm25": predicted.future_pm25,
        "future_pm10": predicted.future_pm10,
        "future_aqi": result.final_aqi,
        "future_category": result.category,
        "risk_level": result.risk_level,
        "risk_percentage": result.risk_percentage,
        "precautions": result.precautions,
        "model_status": predicted.model_status,
    }


@app.get("/api/sensors/history")
def sensor_history(
    limit: int = 100,
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 5000))

    readings = (
        db.query(SensorData)
        .order_by(SensorData.recorded_at.desc(), SensorData.id.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": item.id,
            "device_id": item.device_id,
            "recorded_at": item.recorded_at,
            "pm25": item.pm25,
            "pm10": item.pm10,
            "latitude": item.latitude,
            "longitude": item.longitude,
        }
        for item in readings
    ]
