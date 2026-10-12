
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

BASE_DIR = Path(r"D:\project\BPUT")
DATA_PATH = BASE_DIR / "dataset" / "output.csv"
MODEL_PATH = BASE_DIR / "pollution_forecasters.pkl"
OUTPUT_PATH = BASE_DIR / "dataset" / "Final.csv"
RESULTS_PATH = BASE_DIR / "forecast_accuracy_results.csv"

dataset = pd.read_csv(DATA_PATH)

print("Original dataset shape:", dataset.shape)
print("Original columns:", dataset.columns.tolist())

# Remove unnecessary columns
remove = [
    "Benzene", "Toluene", "NO", "NO2", "NH3",
    "CO", "SO2", "O3", "PM2.5_SubIndex",
    "PM10_SubIndex", "Final_AQI", "Result"
]

dataset = dataset.drop(columns=remove, errors="ignore")

# Check required columns
required = ["Date", "Timestamp", "PM2.5", "PM10"]
missing_columns = [
    col for col in required if col not in dataset.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# 2. CORRECT TIMESTAMP HANDLING
# --------------------------------------------------

# In the provided CSV, Date contains the actual
# observation date and time at 15-minute intervals.
dataset["datetime"] = pd.to_datetime(
    dataset["Date"],
    errors="coerce",
    utc=True
)

# Convert pollutant columns to numeric
for pollutant in ["PM2.5", "PM10"]:
    dataset[pollutant] = pd.to_numeric(
        dataset[pollutant],
        errors="coerce"
    )

# Remove invalid timestamps and missing pollutant values
dataset = dataset.dropna(
    subset=["datetime", "PM2.5", "PM10"]
).copy()

# Sort by the actual observation timestamp
dataset = dataset.sort_values(
    "datetime"
).reset_index(drop=True)

# Check duplicate observation timestamps
print("\nDuplicate observation timestamps:",
      dataset["datetime"].duplicated().sum())

print("First observation:", dataset["datetime"].min())
print("Last observation:", dataset["datetime"].max())

# Check actual sampling interval
time_differences = dataset["datetime"].diff().dropna()

print("\nMost common sampling intervals:")
print(time_differences.value_counts().head(10))

print("\nDataset shape after cleaning:", dataset.shape)


# --------------------------------------------------
# 3. CONFIGURE FORECAST HORIZON
# --------------------------------------------------

# HORIZON = 1 means one observed row ahead.
# If readings are 15 minutes apart:
# 1 = 15 minutes, 2 = 30 minutes,
# 4 = 60 minutes, 6 = 90 minutes.

HORIZON = 6
LAGS = [1, 2, 3]

pollutants = ["PM2.5", "PM10"]

# --------------------------------------------------
# 4. CREATE LAG FEATURES AND FUTURE TARGETS
# --------------------------------------------------

lag_features = []

for pollutant in pollutants:

    for lag in LAGS:
        lag_name = f"{pollutant}_lag{lag}"

        dataset[lag_name] = dataset[pollutant].shift(lag)
        lag_features.append(lag_name)

    target_name = f"future_{pollutant}"

    dataset[target_name] = dataset[pollutant].shift(-HORIZON)

features = pollutants + lag_features

target_columns = [
    f"future_{pollutant}" for pollutant in pollutants
]

dataset = dataset.replace(
    [np.inf, -np.inf], np.nan
)

dataset = dataset.dropna(
    subset=features + target_columns
).reset_index(drop=True)

if len(dataset) < 100:
    raise ValueError(
        "Too few usable rows for reliable model evaluation."
    )

X = dataset[features]

print("\nFeatures:", features)
print("Usable rows:", len(dataset))
print("Forecast horizon:", HORIZON, "rows ahead")

# --------------------------------------------------
# 5. CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

split_index = int(len(dataset) * 0.8)

# Keep a gap so training targets do not overlap
# the beginning of the test period.
train_end = split_index - HORIZON

if train_end <= 0 or split_index >= len(dataset):
    raise ValueError("Insufficient data for train/test split.")

x_train = X.iloc[:train_end]
x_test = X.iloc[split_index:]

print("\nTraining samples:", len(x_train))
print("Testing samples:", len(x_test))

# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

models = {}
results = []

for pollutant in pollutants:

    target_col = f"future_{pollutant}"

    y = dataset[target_col]

    y_train = y.iloc[:train_end]
    y_test = y.iloc[split_index:]

    print(f"\nTraining XGBoost for {target_col}...")

    regressor = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(x_train, y_train)

    predictions = regressor.predict(x_test)

    # XGBoost evaluation
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # MAPE: exclude actual values close to zero
    actual = y_test.to_numpy()
    valid = np.abs(actual) > 1e-8

    if valid.any():
        mape = np.mean(
            np.abs(
                (actual[valid] - predictions[valid])
                / actual[valid]
            )
        ) * 100
    else:
        mape = np.nan

    # Baseline: future value equals current value
    baseline = x_test[pollutant].to_numpy()

    baseline_mae = mean_absolute_error(
        y_test, baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline)
    )

    # Store trained model
    models[pollutant] = regressor

    results.append({
        "Pollutant": pollutant,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "MAE_improvement_percent": (
            (baseline_mae - mae) / baseline_mae * 100
            if baseline_mae > 0 else np.nan
        )
    })

    print(f"\n--- {pollutant} results ---")
    print(f"XGBoost MAE:       {mae:.4f}")
    print(f"XGBoost RMSE:      {rmse:.4f}")
    print(f"XGBoost R2:        {r2:.4f}")
    print(f"XGBoost MAPE:      {mape:.2f}%")
    print(f"Baseline MAE:      {baseline_mae:.4f}")
    print(f"Baseline RMSE:     {baseline_rmse:.4f}")

    if mae < baseline_mae:
        print("Result: XGBoost beats the baseline on MAE.")
    else:
        print("Result: XGBoost does not beat the baseline on MAE.")

# --------------------------------------------------
# 7. SAVE MODELS AND RESULTS
# --------------------------------------------------

joblib.dump(models, MODEL_PATH)

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

dataset.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print("\nModels saved to:", MODEL_PATH)
print("Evaluation saved to:", RESULTS_PATH)
print("Processed dataset saved to:", OUTPUT_PATH)
