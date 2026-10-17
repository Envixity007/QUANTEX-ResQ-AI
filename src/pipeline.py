"""Minimal data-ingestion pipeline for ResQ AI.

This starter fetches USGS earthquake events and optionally cleans local EM-DAT
and HumAID CSV exports. It deliberately does not invent labels or predictions.
"""
from pathlib import Path
import logging

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_usgs() -> pd.DataFrame:
    """Fetch the USGS all-day GeoJSON feed and normalize core fields."""
    response = requests.get(USGS_URL, timeout=30)
    response.raise_for_status()
    features = response.json().get("features", [])
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
        rows.append({
            "source": "USGS",
            "event_id": feature.get("id"),
            "title": props.get("title"),
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "time_utc": pd.to_datetime(props.get("time"), unit="ms", utc=True, errors="coerce"),
            "longitude": coords[0] if len(coords) > 0 else None,
            "latitude": coords[1] if len(coords) > 1 else None,
            "depth_km": coords[2] if len(coords) > 2 else None,
            "url": props.get("url"),
        })
    return pd.DataFrame(rows)


def clean_optional_csv(filename: str, output_name: str) -> None:
    source_path = RAW / filename
    if not source_path.exists():
        logging.info("Optional input not found; skipping %s", source_path)
        return
    df = pd.read_csv(source_path, low_memory=False)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates().dropna(how="all")
    output_path = PROCESSED / output_name
    df.to_csv(output_path, index=False)
    logging.info("Saved %s rows to %s", len(df), output_path)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    clean_optional_csv("emdat_india.csv", "emdat_clean.csv")
    clean_optional_csv("humAID.csv", "humAID_clean.csv")
    try:
        usgs = fetch_usgs()
        output_path = PROCESSED / "usgs_earthquakes.csv"
        usgs.to_csv(output_path, index=False)
        logging.info("Saved %s USGS events to %s", len(usgs), output_path)
    except (requests.RequestException, ValueError) as exc:
        logging.error("Could not fetch USGS feed: %s", exc)
        logging.info("You can rerun the pipeline when internet access is available.")


if __name__ == "__main__":
    main()
