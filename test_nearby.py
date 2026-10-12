import pandas as pd
from aqi_agent import find_nearby_features

CSV_FILE = r"D:\project\BPUT\dataset\location.csv"

df = pd.read_csv(CSV_FILE)

print("CSV columns:", df.columns.tolist())

# Adjust these names if your CSV uses different column names.
latitude_column = next(
    (c for c in ["latitude", "Latitude", "lat", "LAT"]
     if c in df.columns),
    None
)

longitude_column = next(
    (c for c in ["longitude", "Longitude", "lon", "lng", "LON"]
     if c in df.columns),
    None
)

if not latitude_column or not longitude_column:
    raise ValueError(
        "Latitude/longitude columns were not found. "
        "Check the printed CSV columns."
    )

row = df.dropna(subset=[latitude_column, longitude_column]).iloc[-1]

location = {
    "latitude": float(row[latitude_column]),
    "longitude": float(row[longitude_column])
}

features = find_nearby_features(location)

print(f"\nNearby mapped features found: {len(features)}")

for feature in features[:15]:
    print(
        f"Type: {feature['category']} | "
        f"Distance: {feature['distance_m']} m | "
        f"Details: {feature['details']}"
    )

if not features:
    print("No nearby features found within the search radius.")