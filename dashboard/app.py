import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn

# PAGE CONFIGURATION

st.set_page_config(
    page_title="ResQ AI | Disaster Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS — JET BLACK / CERULEAN / NEON PINK

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #050505;
  --paper: #050505;
  --panel: #101014;
  --line: #24242B;
  --muted: #A8A8B3;
  --signal: #007BA7;
  --teal: #007BA7;
  --gold: #FF10F0;
}

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}
.stApp {
  background: var(--paper);
  color: var(--ink);
}
[data-testid="stHeader"] {
  background: rgba(16,23,19,.96);
}
[data-testid="stSidebar"] {
  background: #09090D;
  border-right: 1px solid #24242B;
}
[data-testid="stSidebar"] * { color: #F2F2F7; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #A8A8B3;
}
[data-testid="stSidebar"] hr { border-color: #24242B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding: 7px 10px;
  border-radius: 5px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #14141B;
}
.block-container {
  max-width: 1500px;
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}
h1, h2, h3, h4 {
  color: var(--ink) !important;
  letter-spacing: -0.045em;
}
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 750 !important; }
p, label { color: #B8B8C2; }
a { color: var(--teal) !important; }

.hero {
  background: var(--ink);
  border-radius: 4px;
  padding: 28px 32px 30px;
  margin: 6px 0 24px;
  border-left: 5px solid var(--signal);
  position: relative;
}
.hero-label {
  color: #FF10F0;
  text-transform: uppercase;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 2.2px;
  font-weight: 500;
}
.hero h1 {
  color: #F5F5FA !important;
  font-size: clamp(34px, 4vw, 52px);
  line-height: 1;
  margin: 14px 0 8px;
  letter-spacing: -0.065em;
}
.hero p {
  color: #B8B8C2;
  max-width: 760px;
  line-height: 1.65;
  font-size: 13px;
}
.status-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #007BA7;
  margin-right: 7px;
}
.section-heading {
  font-size: 21px;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -.045em;
  margin: 20px 0 3px;
}
.section-subtitle {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: .35px;
  margin-bottom: 17px;
}
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 17px 18px;
  box-shadow: none;
}
div[data-testid="stMetricLabel"] {
  color: #A8A8B3;
  font-family: 'DM Mono', monospace;
  text-transform: uppercase;
  letter-spacing: .7px;
  font-size: 10px;
}
div[data-testid="stMetricValue"] {
  color: var(--ink);
  font-weight: 800;
  letter-spacing: -.06em;
}
div[data-testid="stMetricDelta"] { font-size: 11px; }
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
  border: 1px solid var(--line);
  border-radius: 4px;
  overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 3px;
  background: transparent;
  border-bottom: 1px solid var(--line);
  padding: 0;
}
.stTabs [data-baseweb="tab"] {
  color: #A8A8B3;
  background: transparent;
  border-radius: 0;
  padding: 10px 14px;
  font-size: 12px;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--signal);
  background: transparent !important;
  font-weight: 800;
}
div.stButton > button, div[data-testid="stDownloadButton"] button {
  border-radius: 3px;
  border: 1px solid var(--ink);
  background: var(--ink);
  color: #F5F5FA;
  font-weight: 700;
  transition: transform .12s ease, background .12s ease;
}
div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover {
  background: var(--signal);
  border-color: var(--signal);
  color: white;
  transform: translateY(-1px);
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: #101014;
  border-color: #34343D;
  border-radius: 3px;
}
[data-testid="stAlert"] { border-radius: 3px; }
hr { border-color: var(--line); }
.footer {
  color: #A8A8B3;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  text-align: left;
  padding-top: 24px;
  border-top: 1px solid var(--line);
  margin-top: 30px;
}
@media (max-width: 700px) {
  .hero { padding: 22px 20px; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# STRICT THREE-COLOUR THEME — jet black / cerulean blue / neon pink
st.markdown("""
<style>
:root {
  --ink: #F2F2F7; --paper: #050505; --panel: #101014; --line: #24242B;
  --muted: #A8A8B3; --signal: #007BA7; --teal: #007BA7; --gold: #FF10F0;
}
.stApp, [data-testid="stAppViewContainer"] { background: #050505 !important; color: #F2F2F7 !important; }
[data-testid="stHeader"] { background: rgba(17,21,19,.96) !important; }
[data-testid="stSidebar"] { background: #101014 !important; border-right: 1px solid #24242B !important; }
[data-testid="stSidebar"] * { color: #F2F2F7 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #A8A8B3 !important; }
[data-testid="stSidebar"] hr { border-color: #24242B !important; }
h1,h2,h3,h4,h5,h6,.section-heading { color: #F2F2F7 !important; }
p, label, [data-testid="stMarkdownContainer"] { color: #C4C4CE; }
a { color: #FF10F0 !important; }
.hero { background: #101014 !important; border-left-color: #007BA7 !important; }
.hero h1 { color: #F5F5FA !important; }
.hero p { color: #B8B8C2 !important; }
.section-subtitle, .footer { color: #A8A8B3 !important; border-color: #24242B !important; }
div[data-testid="stMetric"] { background: #101014 !important; border: 1px solid #24242B !important; border-radius: 2px !important; }
div[data-testid="stMetricLabel"] { color: #B8B8C2 !important; }
div[data-testid="stMetricValue"] { color: #F5F5FA !important; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border-color: #24242B !important; }
.stTabs [data-baseweb="tab-list"] { border-color: #24242B !important; }
.stTabs [data-baseweb="tab"] { color: #A8A8B3 !important; }
.stTabs [aria-selected="true"] { color: #F5F5FA !important; border-bottom-color: #007BA7 !important; }
div.stButton > button, div[data-testid="stDownloadButton"] button { background: #007BA7 !important; border: 1px solid #007BA7 !important; color: #050505 !important; border-radius: 2px !important; }
div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover { background: #25A9D6 !important; border-color: #25A9D6 !important; }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { background: #101014 !important; color: #F2F2F7 !important; border-color: #34343D !important; border-radius: 2px !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background: #101014 !important; color: #F2F2F7 !important; }
[data-testid="stAlert"] { background: #101014 !important; border-radius: 2px !important; }
hr { border-color: #24242B !important; }
[data-testid="stExpander"] { border: 1px solid #24242B !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { color: #F2F2F7 !important; }
[data-testid="stCaptionContainer"] { color: #A8A8B3 !important; }
</style>
""", unsafe_allow_html=True)

# Use a dark plotting palette for Plotly Express figures.
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#007BA7", "#FF10F0"]

# PATHS


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "ibm final data.xlsx"
MODEL_PATH = ROOT / "models" / "severity_model.joblib"
LABEL_PATH = ROOT / "models" / "severity_labels.joblib"

# HumAID NLP model artifacts
NLP_MODEL_PATH = ROOT / "models" / "humaid_text_model.pt"
NLP_VECTORIZER_PATH = ROOT / "models" / "humaid_tfidf.joblib"
NLP_LABELS_PATH = ROOT / "models" / "humaid_labels.joblib"

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



# HUM AID NLP CLASSIFIER

class DisasterTextClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_nlp_model():
    """Load the trained HumAID classifier and preprocessing artifacts once."""
    required = [NLP_MODEL_PATH, NLP_VECTORIZER_PATH, NLP_LABELS_PATH]
    if not all(path.exists() for path in required):
        return None, None, None

    checkpoint = torch.load(
        NLP_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    vectorizer = joblib.load(NLP_VECTORIZER_PATH)
    encoder = joblib.load(NLP_LABELS_PATH)

    model = DisasterTextClassifier(
        checkpoint["input_size"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vectorizer, encoder


def predict_disaster_tweet(tweet, model, vectorizer, encoder):
    """Return predicted HumAID category, probability, and all class scores."""
    features = vectorizer.transform([tweet]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=1)[0]

    index = int(torch.argmax(probabilities).item())
    label = encoder.inverse_transform([index])[0]
    confidence = float(probabilities[index].item())
    return label, confidence, probabilities.cpu().numpy()


def style_chart(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#B8B8C2",
            size=11,
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B8B8C2"),
        ),
    )

    fig.update_xaxes(
        gridcolor="#24242B",
        zerolinecolor="#24242B",
    )

    fig.update_yaxes(
        gridcolor="#24242B",
        zerolinecolor="#24242B",
    )

    return fig


# SIDEBAR

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 25px;">
            <div style="font-size:27px;font-weight:800;color:#F5F5FA;">
                Res<span style="color:#FF10F0;">Q</span> AI
            </div>
            <div style="color:#007BA7;font-size:12px;">
                FIELD INTELLIGENCE / 01
            </div>
            <div style="color:#007BA7;font-size:11px;margin-top:5px;">
                A QUANTEX field system
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
        <span style="color:#007BA7;">●</span> USGS Earthquake Feed<br>
        <span style="color:#007BA7;">●</span> Historical Disaster Dataset<br>
        <span style="color:#007BA7;">●</span> XGBoost Severity Model<br>\n        <span style="color:#007BA7;">●</span> HumAID NLP Classifier
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
        <div class="hero-label">QUANTEX  /  IBM Z DATATHON 2026  /  FIELD SYSTEM</div>
        <h1>Res<span style="color:#FF10F0;">Q</span> AI</h1>
        <div style="font-size:19px;color:#B8B8C2;font-weight:600;">
            AI-Powered Disaster Intelligence & Response
        </div>
        <p style="margin-top:10px;">
            A field intelligence workspace for monitoring seismic activity, examining historical events, and triaging humanitarian signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# OVERVIEW

if page == "Overview":

    st.markdown('<div class="section-heading">Situation brief</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="section-subtitle">'
        'Current USGS feed alongside the historical dataset and model outputs'
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
                color_continuous_scale=["#050505", "#007BA7", "#FF10F0"],
                projection="natural earth",
            )

            fig.update_geos(
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#101014",
                showocean=True,
                oceancolor="#050505",
                showcountries=True,
                countrycolor="#24242B",
                showcoastlines=True,
                coastlinecolor="#24242B",
                showframe=False,
            )

            fig.update_layout(
                height=410,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#D6D6DF"),
                coloraxis_colorbar=dict(
                    title="Magnitude",
                    tickfont=dict(color="#D6D6DF"),
                ),
            )

            fig.update_layout(template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="#050505", font_color="#F2F2F7")
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
                "High": "#FF10F0",
                "Medium": "#007BA7",
                "Low": "#777784",
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
                marker=dict(line=dict(color="#24242B", width=3)),
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
                        font=dict(size=20, color="#F5F5FA"),
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
                color_continuous_scale=["#050505", "#007BA7", "#FF10F0"],
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
                            "High": "#FF10F0",
                            "Medium": "#007BA7",
                            "Low": "#777784",
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
                            color_continuous_scale=[[0, "#050505"], [0.55, "#007BA7"], [1, "#FF10F0"]],
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


# HUMANITARIAN TWEET NLP SECTION

if page == "AI Analysis":
    st.divider()
    st.header("Humanitarian Tweet Intelligence")
    st.caption(
        "Experimental PyTorch classifier trained on the HumAID disaster-tweet dataset."
    )

    nlp_model, nlp_vectorizer, nlp_encoder = load_nlp_model()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("HumAID categories", "10")
    metric_col2.metric("Held-out test accuracy", "73.94%")
    metric_col3.metric("Test samples", "15,160")

    st.warning(
        "Experimental baseline: validation loss increased during training, "
        "indicating overfitting. The missing_or_found_people category had "
        "zero recall on the test set. Verify outputs manually; do not use "
        "this model as the sole basis for emergency decisions."
    )

    if nlp_model is None:
        st.info(
            "NLP model files were not found. Run `python src/train_nlp.py` "
            "and ensure these files exist in the project's models folder: "
            "`humaid_text_model.pt`, `humaid_tfidf.joblib`, and "
            "`humaid_labels.joblib`."
        )
    else:
        with st.form("humaid_tweet_form"):
            tweet_text = st.text_area(
                "Disaster-related tweet",
                placeholder=(
                    "Example: People are trapped after the earthquake. "
                    "We urgently need rescue teams."
                ),
                height=120,
            )
            classify_clicked = st.form_submit_button(
                "Classify Tweet",
                type="primary",
                use_container_width=True,
            )

        if classify_clicked:
            if not tweet_text.strip():
                st.warning("Enter a tweet before running classification.")
            else:
                try:
                    category, confidence, probabilities = predict_disaster_tweet(
                        tweet_text.strip(),
                        nlp_model,
                        nlp_vectorizer,
                        nlp_encoder,
                    )

                    result_col1, result_col2 = st.columns([2, 1])
                    result_col1.metric("Predicted category", category)
                    result_col2.metric("Model probability", f"{confidence:.1%}")

                    st.caption(
                        "Probability is the model's score, not a guarantee "
                        "that the category is correct."
                    )

                    if category == "missing_or_found_people":
                        st.error(
                            "High caution: this category had zero recall in "
                            "the reported test results. Do not rely on this "
                            "prediction without human review."
                        )

                    probability_df = pd.DataFrame({
                        "Category": nlp_encoder.classes_,
                        "Probability": probabilities,
                    }).sort_values("Probability", ascending=False)

                    st.subheader("Category scores")
                    fig = px.bar(
                        probability_df.sort_values("Probability").tail(10),
                        x="Probability",
                        y="Category",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale=["#050505", "#007BA7", "#FF10F0"],
                    )
                    fig.update_layout(
                        coloraxis_showscale=False,
                        xaxis_tickformat=".0%",
                    )
                    st.plotly_chart(
                        style_chart(fig, 360),
                        use_container_width=True,
                    )
                except Exception as error:
                    st.error(f"Tweet classification failed: {error}")


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
    - **PyTorch + TF-IDF** — experimental HumAID tweet classification
    - **USGS** — live global earthquake feed

    ### Current capabilities

    - Global earthquake monitoring
    - Historical disaster exploration
    - Interactive charts and filters
    - Historical severity model integration
    - Humanitarian tweet category prediction (when model artifacts are present)

    ### Important limitations

    - Earthquake events come from the global USGS past-day feed.
    - Historical dataset statistics describe the supplied dataset only.
    - Severity predictions are experimental and are not emergency alerts.
    - The HumAID NLP classifier is an experimental baseline and overfits.
    - The missing_or_found_people category had zero recall on the reported test set.
    - IBM Granite is not integrated in this dashboard.
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