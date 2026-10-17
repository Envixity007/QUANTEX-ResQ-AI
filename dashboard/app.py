import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np

st.set_page_config(
    page_title="ResQ AI | Disaster Intelligence",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 ResQ AI")
st.caption("AI-Powered Disaster Intelligence & Response Platform")

USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
    "summary/all_day.geojson"
)

@st.cache_data(ttl=300)
def get_earthquakes():
    response = requests.get(USGS_URL, timeout=20)
    response.raise_for_status()

    records = []

    for item in response.json()["features"]:
        properties = item["properties"]
        coordinates = item["geometry"]["coordinates"]

        records.append({
            "Place": properties.get("place") or "Unknown",
            "Magnitude": properties.get("mag"),
            "Time (UTC)": (
                datetime.fromtimestamp(
                    properties["time"] / 1000,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M")
                if properties.get("time")
                else "Unknown"
            ),
            "Latitude": coordinates[1],
            "Longitude": coordinates[0],
            "Depth (km)": coordinates[2],
            "Alert": properties.get("alert") or "Not provided",
            "Status": properties.get("status") or "Unknown",
            "URL": properties.get("url") or "",
        })

    return pd.DataFrame(records)


@st.cache_data
def get_historical_data():
    path = Path("data/processed/ibm final data.xlsx")

    if not path.exists():
        return pd.DataFrame()

    return pd.read_excel(path)


# Sidebar
st.sidebar.header("Dashboard Controls")

if st.sidebar.button("🔄 Refresh earthquake data"):
    get_earthquakes.clear()

st.sidebar.caption("Live earthquake feed: USGS")
st.sidebar.caption("Refresh interval: 5 minutes")

# Live data
st.header("🌍 Live Earthquake Monitoring")

try:
    earthquakes = get_earthquakes()

    if earthquakes.empty:
        st.warning("The USGS feed returned no earthquake records.")
    else:
        significant = earthquakes[
            earthquakes["Magnitude"].fillna(0) >= 4.0
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Earthquakes Today", len(earthquakes))
        col2.metric("Magnitude ≥ 4", len(significant))
        col3.metric(
            "Largest Magnitude",
            f"{earthquakes['Magnitude'].max():.1f}",
        )
        col4.metric(
            "Strongest Event",
            (
                significant["Place"].iloc[0]
                if not significant.empty
                else "No M4+ events"
            ),
        )

        st.subheader("Earthquake Locations")

        map_data = earthquakes.dropna(
            subset=["Latitude", "Longitude"]
        )

        st.map(
            map_data.rename(
                columns={"Latitude": "lat", "Longitude": "lon"}
            )[["lat", "lon"]],
            color="#ef4444",
            size=12,
        )

        st.subheader("Recent Earthquake Events")

        min_magnitude = st.slider(
            "Minimum magnitude",
            min_value=0.0,
            max_value=7.0,
            value=0.0,
            step=0.5,
        )

        filtered = earthquakes[
            earthquakes["Magnitude"].fillna(-1) >= min_magnitude
        ].sort_values("Magnitude", ascending=False)

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("USGS Event"),
            },
        )

except requests.RequestException as error:
    st.error(f"Unable to retrieve USGS data: {error}")

# Historical data
st.divider()
st.header("📊 Historical Indian Disasters")

historical = get_historical_data()

if historical.empty:
    st.warning(
        "Historical dataset not found. Place "
        "'ibm final data.xlsx' in data/processed/."
    )
else:
    if "severity" in historical.columns:
        severity_counts = (
            historical["severity"].value_counts().rename_axis("Severity")
            .reset_index(name="Records")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Severity Distribution")
            fig = px.bar(
                severity_counts,
                x="Severity",
                y="Records",
                color="Severity",
                title="Historical Severity Labels",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Disaster Types")
            if "disaster_type" in historical.columns:
                types = (
                    historical["disaster_type"]
                    .value_counts()
                    .head(10)
                    .rename_axis("Disaster Type")
                    .reset_index(name="Records")
                )
                fig = px.bar(
                    types,
                    x="Records",
                    y="Disaster Type",
                    orientation="h",
                    title="Top 10 Disaster Types",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical Records")
    st.dataframe(
        historical.head(100),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "ResQ AI | QUANTEX | Data sources: USGS and supplied historical dataset. "
    "Severity classifications are experimental and must not be used as "
    "the sole basis for emergency decisions."
)


st.divider()
st.header("🤖 AI Disaster Severity Classification")

MODEL_PATH = Path("models/severity_model.joblib")
LABEL_PATH = Path("models/severity_labels.joblib")

if historical.empty:
    st.warning("Historical dataset is unavailable.")
elif not MODEL_PATH.exists() or not LABEL_PATH.exists():
    st.warning("Trained severity model not found in the models folder.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_PATH)

        # Prepare features exactly as in train_model.py
        excluded_columns = [
            "severity",
            "event_id",
            "event_name",
            "deaths",
            "injured",
            "total_affected",
            "damage_usd_thousand",
            "injured_missing",
            "total_affected_missing",
        ]

        X = historical.drop(
            columns=excluded_columns,
            errors="ignore",
        ).copy()

        # Match the date transformation used during training
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col]).map(
                    lambda value: (
                        value.timestamp()
                        if pd.notna(value)
                        else np.nan
                    )
                )

        # Run model predictions
        predicted_encoded = model.predict(X)
        predicted_labels = label_encoder.inverse_transform(
            predicted_encoded.astype(int)
        )

        probabilities = model.predict_proba(X)

        results = historical.copy()
        results["Predicted Severity"] = predicted_labels
        results["Prediction Confidence"] = probabilities.max(axis=1)

        st.info(
            "These are model predictions for historical records, "
            "not predictions of future disasters. Confidence is the "
            "model's highest class probability, not a guarantee of correctness."
        )

        # Summary metrics
        metric1, metric2, metric3 = st.columns(3)

        counts = pd.Series(predicted_labels).value_counts()

        metric1.metric("Predicted High", int(counts.get("High", 0)))
        metric2.metric("Predicted Medium", int(counts.get("Medium", 0)))
        metric3.metric("Predicted Low", int(counts.get("Low", 0)))

        # Filter predictions
        selected_severity = st.multiselect(
            "Filter predicted severity",
            options=list(label_encoder.classes_),
            default=list(label_encoder.classes_),
        )

        filtered_results = results[
            results["Predicted Severity"].isin(selected_severity)
        ]

        display_columns = [
            column for column in [
                "event_id",
                "disaster_type",
                "disaster_subtype",
                "primary_state",
                "start_date",
                "severity",
                "Predicted Severity",
                "Prediction Confidence",
            ]
            if column in filtered_results.columns
        ]

        st.subheader("Historical Disaster Predictions")

        st.dataframe(
            filtered_results[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prediction Confidence": st.column_config.ProgressColumn(
                    "Prediction Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1%%",
                )
            },
        )

        # Compare historical labels with model predictions
        if "severity" in results.columns:
            st.subheader("Actual vs Predicted Severity")

            comparison = pd.crosstab(
                results["severity"],
                results["Predicted Severity"],
                rownames=["Historical Label"],
                colnames=["Model Prediction"],
            )

            fig = px.imshow(
                comparison,
                text_auto=True,
                aspect="auto",
                title="Historical Labels vs Model Predictions",
                labels={
                    "x": "Predicted Severity",
                    "y": "Historical Label",
                    "color": "Records",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as error:
        st.error(f"Unable to generate severity predictions: {error}")