import json
import os
from pathlib import Path

import requests
from shapely.geometry import Point, shape
from shapely.ops import transform
from pyproj import CRS, Transformer
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SEARCH_RADIUS = float(os.getenv("NEARBY_RADIUS_METERS", "3000"))

MAP_FILES = {
    "road": os.getenv(
        "ROADS_GEOJSON",
        str(BASE_DIR / "maps" / "roads.geojson")
    ),
    "traffic": os.getenv(
        "TRAFFIC_GEOJSON",
        str(BASE_DIR / "maps" / "traffic.geojson")
    ),
    "poi": os.getenv(
        "POIS_GEOJSON",
        str(BASE_DIR / "maps" / "pois.geojson")
    ),
    "landuse": os.getenv(
        "LANDUSE_GEOJSON",
        str(BASE_DIR / "maps" / "Landuse.geojson")
    ),
}


def resolve_map_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


def find_nearby_features(location):
    """Find nearby features in local GeoJSON files."""

    latitude = float(location["latitude"])
    longitude = float(location["longitude"])

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True
    )

    sensor_point = transform(
        transformer.transform,
        Point(longitude, latitude)
    )

    search_area = sensor_point.buffer(SEARCH_RADIUS)
    nearby = []

    useful_keys = (
        "name", "type", "category", "highway",
        "landuse", "industrial", "man_made",
        "amenity", "natural", "shop", "office"
    )

    for category, configured_path in MAP_FILES.items():
        file_path = resolve_map_path(configured_path)

        if not file_path.exists():
            print(f"Map layer missing: {file_path}")
            continue

        try:
            with file_path.open("r", encoding="utf-8") as file:
                geojson = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            print(f"Could not read {file_path.name}: {error}")
            continue

        if geojson.get("type") != "FeatureCollection":
            print(f"Invalid GeoJSON FeatureCollection: {file_path.name}")
            continue

        for feature in geojson.get("features", []):
            geometry_data = feature.get("geometry")

            if not geometry_data:
                continue

            try:
                geometry = shape(geometry_data)

                if geometry.is_empty:
                    continue

                projected = transform(
                    transformer.transform,
                    geometry
                )

                if not projected.intersects(search_area):
                    continue

                distance = projected.distance(sensor_point)
                properties = feature.get("properties") or {}

                details = {
                    key: properties[key]
                    for key in useful_keys
                    if properties.get(key) not in (None, "")
                }

                nearby.append({
                    "category": category,
                    "distance_m": round(distance, 1),
                    "details": details
                })

            except (ValueError, TypeError, KeyError) as error:
                print(f"Skipping invalid feature: {error}")
                continue

    nearby.sort(key=lambda item: item["distance_m"])

    return nearby[:50]


def collect_evidence():
    """Fetch current readings, AQI, forecast and nearby map context."""

    base_url = os.getenv(
        "AQI_API_BASE_URL",
        "http://127.0.0.1:8080"
    ).rstrip("/")

    def fetch(endpoint):
        try:
            response = requests.get(
                f"{base_url}{endpoint}",
                timeout=5
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return data if isinstance(data, dict) else None

        except (requests.RequestException, ValueError) as error:
            print(f"Could not fetch {endpoint}: {error}")
            return None

    sensor = fetch("/api/sensors/latest")
    current = fetch("/api/aqi/current")
    forecast = fetch("/api/forecast/latest")

    # Current measured data
    current_data = None

    if sensor:
        current_data = {
            "PM2.5": sensor.get("pm25"),
            "PM10": sensor.get("pm10"),
            "temperature": sensor.get("temperature"),
            "humidity": sensor.get("humidity"),
            "timestamp": sensor.get("recorded_at"),
        }

        if current:
            current_data.update({
                "AQI": current.get("final_aqi"),
                "category": current.get("category"),
                "advice": current.get("advice"),
            })

    # Future predicted data
    future_data = None

    if forecast:
        future_data = {
            "predicted_PM2.5": forecast.get("future_pm25"),
            "predicted_PM10": forecast.get("future_pm10"),
            "future_AQI": forecast.get("future_aqi"),
            "category": forecast.get("future_category"),
            "precautions": forecast.get("precautions"),
            "model_status": forecast.get("model_status"),
        }

    # Validate sensor GPS internally
    location = None

    if sensor:
        latitude = sensor.get("latitude")
        longitude = sensor.get("longitude")

        try:
            if latitude is not None and longitude is not None:
                latitude = float(latitude)
                longitude = float(longitude)

                if (
                    -90 <= latitude <= 90
                    and -180 <= longitude <= 180
                ):
                    location = {
                        "latitude": latitude,
                        "longitude": longitude
                    }

        except (TypeError, ValueError):
            location = None

    features = (
        find_nearby_features(location)
        if location
        else []
    )

    # Keep only readable, deduplicated map context.
    # Do not send GPS coordinates or exact distances to the LLM.
    safe_features = []
    seen_features = set()

    for feature in features:
        category = str(feature.get("category", "unknown"))
        details = feature.get("details") or {}

        name = (
            details.get("name")
            or details.get("highway")
            or details.get("landuse")
            or details.get("amenity")
            or details.get("shop")
            or details.get("industrial")
            or details.get("natural")
            or details.get("type")
            or category
        )

        name = str(name).strip()
        key = (category.lower(), name.lower())

        if key in seen_features:
            continue

        seen_features.add(key)

        safe_features.append({
            "category": category,
            "name": name
        })

        if len(safe_features) >= 10:
            break

    return {
        "current_data": current_data,
        "predicted_data": future_data,
        "future_data": future_data,
        "nearby_features": safe_features,
        "gps_status": "available" if location else "unavailable",
    }


def generate_report(
    question="Analyze the current air quality and future precautions."
):
    """Generate an evidence-based report using the local Ollama model."""

    evidence = collect_evidence()

    model_name = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:0.5b"
    )

    ollama_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    system_prompt = """
You are an evidence-based air-quality advice assistant.
Use only the supplied sensor measurements, AQI results,
forecast results and nearby map features.

Return exactly these five sections:

1. CURRENT AIR QUALITY
Report measured PM2.5, PM10, current AQI and category
only when supplied. If unavailable, say so.

2. IMMEDIATE ADVICE
Give exactly two distinct actions based on current measurements.
Do not repeat any action in section 5.

3. LOCATION CONTEXT
Mention relevant nearby mapped features when available.
Explain that proximity does not prove a pollution source.
If no location or features are available, state this clearly.

4. FUTURE AIR QUALITY
Report predicted PM2.5, PM10, future AQI and category
only when supplied. Clearly label predictions as forecasts.
If unavailable, state: Forecast unavailable.

5. FUTURE PRECAUTIONS
Give exactly two different actions based on the predicted
air quality. Do not copy or paraphrase section 2.
For a Moderate forecast, suggest proportionate precautions,
such as monitoring AQI before outdoor activities and reducing
prolonged strenuous outdoor exercise if sensitive.
Do not exaggerate the risk or invent future conditions.
Provide two future-focused precautions distinct from section 2.
For example, suggest checking local AQI updates before outdoor activities or planning indoor exercise during peak traffic hours.

Rules:
- Never invent measurements, AQI values, causes or predictions.
- Do not confuse current AQI with future AQI.
- Do not repeat recommendations.
- Nearby map features are context, not proof of causation.
- Never reveal GPS coordinates or exact map distances.
- Do not claim a nearby feature caused pollution without evidence.
- If a value is missing, state that it is unavailable.
- Keep the report concise and under 200 words.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            "User question: {question}\n\nEvidence:\n{evidence}"
        )
    ])

    model = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.1,
        num_predict=600,
    )

    chain = prompt | model | StrOutputParser()

    report = chain.invoke({
        "question": question,
        "evidence": json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False
        )
    })

    return {
        "status": "ok",
        "report": report,
        "gps_status": evidence["gps_status"],
        "map_features_found": len(evidence["nearby_features"])
    }


if __name__ == "__main__":
    result = generate_report()

    print(result["report"])
    print("\nGPS status:", result["gps_status"])
    print("Nearby map features:", result["map_features_found"])