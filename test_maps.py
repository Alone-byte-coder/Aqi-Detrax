from pathlib import Path
import json

folder = Path(__file__).resolve().parent / "maps"

for name in [
    "roads.geojson",
    "traffic.geojson",
    "pois.geojson",
    "landuse.geojson",
]:
    path = folder / name

    if not path.exists():
        print(f"MISSING: {name}")
        continue

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        count = len(data.get("features", []))
        print(f"OK: {name} — {count} features")
    except Exception as error:
        print(f"ERROR: {name} — {error}")