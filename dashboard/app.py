import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

# PAGE CONFIGURATION

st.set_page_config(
    page_title="ResQ AI | Disaster Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS — DARK PROFESSIONAL UI

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #142238 0%, transparent 45%),
        #080d16;
    color: #e7edf7;
}

[data-testid="stSidebar"] {
    background: #0b111d;
    border-right: 1px solid #202b3b;
}

[data-testid="stSidebar"] * {
    color: #d8e2f0;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: #f4f7ff !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #aab8cc;
}

.hero {
    padding: 30px 34px;
    border-radius: 18px;
    background: linear-gradient(110deg, #15243a, #101a2b 65%, #241b2c);
    border: 1px solid #29384d;
    margin-bottom: 22px;
}

.hero-label {
    color: #ff6978;
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: 38px;
    margin: 8px 0 4px;
}

.hero p {
    margin-bottom: 0;
    font-size: 14px;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #46df9b;
    margin-right: 7px;
}

.section-heading {
    font-size: 19px;
    font-weight: 700;
    color: #f0f5ff;
    margin: 14px 0 4px;
}

.section-subtitle {
    color: #8496b0;
    font-size: 12px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #141f30, #101824);
    border: 1px solid #273448;
    padding: 19px 20px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #a8b8ce;
}

div[data-testid="stMetricValue"] {
    color: #f5f8ff;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293548;
    border-radius: 12px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0d1522;
    padding: 7px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #aab8cc;
}

.stTabs [aria-selected="true"] {
    background: #263b59 !important;
    color: white !important;
}

div.stButton > button {
    border-radius: 9px;
    border: 1px solid #34445b;
    background: #172438;
    color: white;
}

div.stButton > button:hover {
    border-color: #ff6476;
    color: #ff8793;
}

hr {
    border-color: #263448;
}

.footer {
    color: #72839d;
    font-size: 11px;
    text-align: center;
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# PATHS


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "ibm final data.xlsx"
MODEL_PATH = ROOT / "models" / "severity_model.joblib"
LABEL_PATH = ROOT / "models" / "severity_labels.joblib"

USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
    "summary/all_day.geojson"
)

# DATA FUNCTIONS

@st.cache_data(ttl=300)
def get_earthquakes():
    response = requests.get(USGS_URL, timeout=20)
    response.raise_for_status()

    records = []

    for item in response.json().get("features", []):
        props = item.get("properties", {})
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        if len(coordinates) < 3:
            continue

        event_time = props.get("time")

        records.append({
            "Magnitude": props.get("mag"),
            "Location": props.get("place") or "Unknown",
            "Time (UTC)": (
                datetime.fromtimestamp(
                    event_time / 1000,
                    tz=timezone.utc,
                ).strftime("%d %b %Y, %H:%M")
                if event_time
                else "Unknown"
            ),
            "Latitude": coordinates[1],
            "Longitude": coordinates[0],
            "Depth (km)": coordinates[2],
            "Status": props.get("status") or "Unknown",
            "USGS Link": props.get("url") or "",
        })

    return pd.DataFrame(records)


@st.cache_data
def get_historical_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()

    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        return None, None

    return joblib.load(MODEL_PATH), joblib.load(LABEL_PATH)


def prepare_features(df):
    excluded = [
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

    X = df.drop(columns=excluded, errors="ignore").copy()

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = pd.to_datetime(X[col]).map(
                lambda value: (
                    value.timestamp()
                    if pd.notna(value)
                    else np.nan
                )
            )

    return X


def style_chart(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#bdc9dc",
            size=11,
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#bdc9dc"),
        ),
    )

    fig.update_xaxes(
        gridcolor="#263448",
        zerolinecolor="#263448",
    )

    fig.update_yaxes(
        gridcolor="#263448",
        zerolinecolor="#263448",
    )

    return fig


# SIDEBAR

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 25px;">
            <div style="font-size:27px;font-weight:800;color:#f4f7ff;">
                Res<span style="color:#ff6576;">Q</span> AI
            </div>
            <div style="color:#8fa3bf;font-size:12px;">
                DISASTER INTELLIGENCE
            </div>
            <div style="color:#647b99;font-size:11px;margin-top:5px;">
                Powered by QUANTEX
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Earthquakes",
            "Historical Data",
            "AI Analysis",
            "About",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("#### DATA SOURCES")

    st.markdown(
        """
        <div style="font-size:12px;line-height:2.5;">
        <span style="color:#46df9b;">●</span> USGS Earthquake Feed<br>
        <span style="color:#46df9b;">●</span> Historical Disaster Dataset<br>
        <span style="color:#46df9b;">●</span> XGBoost Severity Model
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("↻ Refresh live data", use_container_width=True):
        get_earthquakes.clear()
        st.rerun()

    st.caption("Live feed refresh interval: 5 minutes")
    st.caption("IBM Z Datathon 2026")

# LOAD DATA


historical = get_historical_data()

# HERO HEADER

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">QUANTEX • IBM Z DATATHON 2026</div>
        <h1>Res<span style="color:#ff6576;">Q</span> AI</h1>
        <div style="font-size:19px;color:#c5d3e7;font-weight:600;">
            AI-Powered Disaster Intelligence & Response
        </div>
        <p style="margin-top:10px;">
            Real-time monitoring. Data-driven insights. Safer communities.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# OVERVIEW

if page == "Overview":

    st.markdown('<div class="section-heading">Operational Overview</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="section-subtitle">'
        'Live earthquake monitoring and historical disaster intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        earthquakes = get_earthquakes()
        live_error = None
    except requests.RequestException as error:
        earthquakes = pd.DataFrame()
        live_error = str(error)

    if live_error:
        st.warning("USGS live data is temporarily unavailable.")

    earthquake_count = len(earthquakes)

    magnitude_series = (
        pd.to_numeric(earthquakes["Magnitude"], errors="coerce")
        if not earthquakes.empty
        else pd.Series(dtype=float)
    )

    major_count = int((magnitude_series >= 4).sum())
    largest = magnitude_series.max()

    if historical.empty:
        severity_counts = pd.Series(dtype=int)
        high_count = 0
    elif "severity" in historical.columns:
        severity_counts = historical["severity"].value_counts()
        high_count = int(severity_counts.get("High", 0))
    else:
        severity_counts = pd.Series(dtype=int)
        high_count = 0

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("EARTHQUAKES · 24H", f"{earthquake_count:,}")
    k2.metric("MAGNITUDE ≥ 4.0", f"{major_count:,}")
    k3.metric(
        "LARGEST MAGNITUDE",
        f"{largest:.1f}" if pd.notna(largest) else "N/A",
    )
    k4.metric("HISTORICAL HIGH LABELS", f"{high_count:,}")

    st.markdown("")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            '<div class="section-heading">Live Earthquake Activity</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            '<span class="status-dot"></span>'
            'USGS events from the past 24 hours'
            '</div>',
            unsafe_allow_html=True,
        )

        if not earthquakes.empty:
            map_df = earthquakes.dropna(
                subset=["Latitude", "Longitude"]
            ).copy()

            map_df["Magnitude"] = pd.to_numeric(
                map_df["Magnitude"], errors="coerce"
            )

            fig = px.scatter_geo(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Magnitude",
                size=map_df["Magnitude"].fillna(0).clip(lower=0.1),
                hover_name="Location",
                hover_data=["Magnitude", "Depth (km)", "Time (UTC)"],
                color_continuous_scale=["#f7b955", "#ff7b51", "#ff435f"],
                projection="natural earth",
            )

            fig.update_geos(
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#182537",
                showocean=True,
                oceancolor="#0b1421",
                showcountries=True,
                countrycolor="#334257",
                showcoastlines=True,
                coastlinecolor="#334257",
                showframe=False,
            )

            fig.update_layout(
                height=410,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                coloraxis_colorbar=dict(
                    title="Magnitude",
                    tickfont=dict(color="#cbd5e1"),
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No earthquake data available at the moment.")

    with right:
        st.markdown(
            '<div class="section-heading">Recent Earthquake Events</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Latest events reported by USGS'
            '</div>',
            unsafe_allow_html=True,
        )

        if not earthquakes.empty:
            recent = earthquakes.copy()
            recent["Magnitude"] = pd.to_numeric(
                recent["Magnitude"], errors="coerce"
            )

            recent = recent.sort_values(
                "Magnitude", ascending=False, na_position="last"
            ).head(10)

            st.dataframe(
                recent[
                    ["Magnitude", "Location", "Time (UTC)", "Depth (km)"]
                ],
                use_container_width=True,
                hide_index=True,
                height=410,
            )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-heading">Historical Disaster Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Analysis of the supplied historical dataset'
            '</div>',
            unsafe_allow_html=True,
        )

        if not historical.empty and "severity" in historical.columns:

            sev_df = (
                historical["severity"]
                .value_counts()
                .rename_axis("Severity")
                .reset_index(name="Records")
            )

            colors = {
                "High": "#ff596b",
                "Medium": "#ffad55",
                "Low": "#45d6a0",
            }

            fig = px.pie(
                sev_df,
                names="Severity",
                values="Records",
                hole=0.68,
                color="Severity",
                color_discrete_map=colors,
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(line=dict(color="#101824", width=3)),
            )

            fig.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="h", y=-0.05),
                annotations=[
                    dict(
                        text=f"<b>{len(historical):,}</b><br>records",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20, color="#f4f7ff"),
                        showarrow=False,
                    )
                ],
            )

            st.plotly_chart(
                style_chart(fig, 300),
                use_container_width=True,
            )

        else:
            st.info("Historical severity labels are unavailable.")

    with right:
        st.markdown(
            '<div class="section-heading">Top Disaster Types</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Most frequent categories in the historical dataset'
            '</div>',
            unsafe_allow_html=True,
        )

        if not historical.empty and "disaster_type" in historical.columns:
            types = (
                historical["disaster_type"]
                .fillna("Unknown")
                .value_counts()
                .head(8)
                .sort_values()
                .rename_axis("Disaster Type")
                .reset_index(name="Records")
            )

            fig = px.bar(
                types,
                x="Records",
                y="Disaster Type",
                orientation="h",
                color="Records",
                color_continuous_scale=["#385d9d", "#9b7bff", "#ff6576"],
            )

            fig.update_layout(coloraxis_showscale=False)

            st.plotly_chart(
                style_chart(fig, 300),
                use_container_width=True,
            )

        else:
            st.info("Disaster type data is unavailable.")


# LIVE EARTHQUAKES PAGE

elif page == "Live Earthquakes":

    st.header("Live Earthquake Monitoring")
    st.caption("Global earthquake events from the USGS past-day feed.")

    try:
        earthquakes = get_earthquakes()

        if earthquakes.empty:
            st.info("No records returned by USGS.")
        else:
            minimum = st.slider(
                "Minimum magnitude",
                0.0,
                7.0,
                0.0,
                0.5,
            )

            filtered = earthquakes[
                pd.to_numeric(
                    earthquakes["Magnitude"], errors="coerce"
                ).fillna(-1) >= minimum
            ].copy()

            st.metric("Matching Events", len(filtered))

            map_df = filtered.dropna(
                subset=["Latitude", "Longitude"]
            )

            if not map_df.empty:
                st.map(
                    map_df.rename(
                        columns={
                            "Latitude": "lat",
                            "Longitude": "lon",
                        }
                    )[["lat", "lon"]]
                )

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "USGS Link": st.column_config.LinkColumn("USGS Details"),
                },
            )

    except requests.RequestException as error:
        st.error(f"Could not load USGS data: {error}")


# HISTORICAL DATA PAGE

elif page == "Historical Data":

    st.header("Historical Disaster Intelligence")
    st.caption("Explore and filter the supplied historical records.")

    if historical.empty:
        st.error(f"Dataset not found: {DATA_PATH}")
    else:
        filtered = historical.copy()

        if "disaster_type" in filtered.columns:
            types = sorted(
                filtered["disaster_type"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_types = st.multiselect(
                "Disaster type",
                types,
                default=types,
            )

            filtered = filtered[
                filtered["disaster_type"].astype(str).isin(selected_types)
            ]

        if "primary_state" in filtered.columns:
            states = sorted(
                filtered["primary_state"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_states = st.multiselect(
                "State",
                states,
                default=[],
                placeholder="All states",
            )

            if selected_states:
                filtered = filtered[
                    filtered["primary_state"].astype(str).isin(selected_states)
                ]

        if "severity" in filtered.columns:
            severities = sorted(
                filtered["severity"].dropna().astype(str).unique()
            )

            selected_severities = st.multiselect(
                "Severity",
                severities,
                default=severities,
            )

            filtered = filtered[
                filtered["severity"].astype(str).isin(selected_severities)
            ]

        st.metric("Records Matching Filters", len(filtered))

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download filtered CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="resq_filtered_disasters.csv",
            mime="text/csv",
        )


# AI ANALYSIS PAGE

elif page == "AI Analysis":

    st.header("AI Disaster Severity Analysis")
    st.caption("Predictions from the locally trained XGBoost classifier.")

    model, label_encoder = load_model()

    if historical.empty:
        st.warning("Historical dataset is unavailable.")
    elif model is None:
        st.error(
            "Model files are missing. Run `python src/train_model.py` first."
        )
    else:
        st.info(
            "Predictions on these historical records demonstrate model "
            "integration. They are not forecasts of future disasters. "
            "Model confidence is not a guarantee of correctness."
        )

        if st.button("Run severity analysis", type="primary"):

            with st.spinner("Analysing historical disaster records..."):
                try:
                    X = prepare_features(historical)
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)

                    results = historical.copy()
                    results["Predicted Severity"] = (
                        label_encoder.inverse_transform(
                            predictions.astype(int)
                        )
                    )
                    results["Model Confidence"] = probabilities.max(axis=1)

                    counts = results["Predicted Severity"].value_counts()

                    a, b, c = st.columns(3)
                    a.metric("Predicted High", int(counts.get("High", 0)))
                    b.metric("Predicted Medium", int(counts.get("Medium", 0)))
                    c.metric("Predicted Low", int(counts.get("Low", 0)))

                    st.subheader("Predicted Severity Distribution")

                    distribution = (
                        results["Predicted Severity"]
                        .value_counts()
                        .rename_axis("Severity")
                        .reset_index(name="Records")
                    )

                    fig = px.bar(
                        distribution,
                        x="Severity",
                        y="Records",
                        color="Severity",
                        color_discrete_map={
                            "High": "#ff596b",
                            "Medium": "#ffad55",
                            "Low": "#45d6a0",
                        },
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True,
                    )

                    st.subheader("Prediction Explorer")

                    selected = st.multiselect(
                        "Filter predicted severity",
                        list(label_encoder.classes_),
                        default=list(label_encoder.classes_),
                    )

                    results = results[
                        results["Predicted Severity"].isin(selected)
                    ]

                    columns = [
                        c for c in [
                            "event_id",
                            "disaster_type",
                            "disaster_subtype",
                            "primary_state",
                            "start_date",
                            "severity",
                            "Predicted Severity",
                            "Model Confidence",
                        ]
                        if c in results.columns
                    ]

                    st.dataframe(
                        results[columns],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Model Confidence": st.column_config.ProgressColumn(
                                "Model Confidence",
                                min_value=0.0,
                                max_value=1.0,
                                format="%.1%%",
                            )
                        },
                    )

                    if "severity" in results.columns:
                        st.subheader("Historical Labels vs Predictions")

                        matrix = pd.crosstab(
                            results["severity"],
                            results["Predicted Severity"],
                        )

                        fig = px.imshow(
                            matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Blues",
                            labels={
                                "x": "Predicted",
                                "y": "Historical Label",
                                "color": "Records",
                            },
                        )

                        st.plotly_chart(
                            style_chart(fig),
                            use_container_width=True,
                        )

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

# ABOUT PAGE

elif page == "About":

    st.header("About ResQ AI")

    st.markdown("""
    **ResQ AI** is a disaster intelligence prototype developed by **QUANTEX**
    for the IBM Z Datathon 2026.

    ### Objective

    Bring historical disaster information, live earthquake observations,
    and experimental machine-learning analysis into one interface.

    ### Technology

    - **Python** — application logic
    - **Streamlit** — interactive dashboard
    - **Pandas** — data processing
    - **Plotly** — visual analytics
    - **XGBoost** — experimental severity classification
    - **USGS** — live global earthquake feed

    ### Current capabilities

    - Global earthquake monitoring
    - Historical disaster exploration
    - Interactive charts and filters
    - Historical severity model integration

    ### Important limitations

    - Earthquake events come from the global USGS past-day feed.
    - Historical dataset statistics describe the supplied dataset only.
    - Severity predictions are experimental and are not emergency alerts.
    - IBM Granite and humanitarian-message classification are not represented
      as integrated until their actual implementations are connected.
    """)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ResQ AI · QUANTEX · IBM Z Datathon 2026<br>
        Built for disaster awareness and data-driven response.
        Experimental predictions must not replace official emergency guidance.
    </div>
    """,
    unsafe_allow_html=True,
)