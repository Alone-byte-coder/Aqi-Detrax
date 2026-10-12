
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    device_id: str = "arduino-uno-q"

    pm25: float = Field(ge=0)
    pm10: float = Field(ge=0)

    temperature: Optional[float] = None
    humidity: Optional[float] = Field(default=None, ge=0, le=100)
    pressure: Optional[float] = Field(default=None, ge=0)

    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    gps_fix: Optional[bool] = None

    recorded_at: Optional[datetime] = None
