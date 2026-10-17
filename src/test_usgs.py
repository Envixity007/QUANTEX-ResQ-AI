import requests

URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

try:
    response = requests.get(URL, timeout=20)
    response.raise_for_status()

    data = response.json()
    earthquakes = data["features"]

    print(f"Earthquakes found: {len(earthquakes)}")
    print("\nLatest earthquake records:\n")

    for quake in earthquakes[:10]:
        props = quake["properties"]
        coordinates = quake["geometry"]["coordinates"]

        print(f"Place: {props.get('place')}")
        print(f"Magnitude: {props.get('mag')}")
        print(f"Time: {props.get('time')}")
        print(f"Coordinates: {coordinates[1]}, {coordinates[0]}")
        print("-" * 40)

except requests.RequestException as error:
    print(f"USGS request failed: {error}")