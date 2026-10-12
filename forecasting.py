
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "pollution_forecasters.pkl"
)

model_bundle = None


def load_models():
    global model_bundle

    if model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model_bundle = joblib.load(MODEL_PATH)

    if not isinstance(model_bundle, dict):
        raise ValueError(
            "Expected a dictionary containing PM2.5 and PM10 models."
        )

    if "PM2.5" not in model_bundle or "PM10" not in model_bundle:
        raise ValueError(
            "The model file must contain 'PM2.5' and 'PM10'."
        )

    return model_bundle


def predict_future(readings):
    """
    readings must contain at least four records,
    ordered from oldest to newest.
    """

    if len(readings) < 4:
        raise ValueError("At least four sensor readings are required.")

    model = load_models()

    current = readings[-1]
    lag1 = readings[-2]
    lag2 = readings[-3]
    lag3 = readings[-4]

    features = np.array([[
        current.pm25,
        current.pm10,
        lag1.pm25,
        lag2.pm25,
        lag3.pm25,
        lag1.pm10,
        lag2.pm10,
        lag3.pm10
    ]])

    future_pm25 = float(
        model["PM2.5"].predict(features)[0]
    )

    future_pm10 = float(
        model["PM10"].predict(features)[0]
    )

    return max(0.0, future_pm25), max(0.0, future_pm10)
