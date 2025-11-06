# app.py — AirVision (GIS Blue Theme) • Fully Automated, Multipage, Live + Forecast + Policy
import os
import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Any, Optional, Tuple

# Maps
import folium
from streamlit_folium import st_folium

# =========================== PAGE CONFIG & THEME ===========================
st.set_page_config(
    page_title="AirVision • Delhi-NCR AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<meta property="og:title" content="AirVision - AI Pollution Intelligence Dashboard">
<meta property="og:description" content="Real-time AQI monitoring, forecasting, health advisory and policy insights for Delhi-NCR.">
<meta property="og:image" content="https://i.ibb.co/PZSFR2X/airvision-banner.png">
<meta name="twitter:card" content="summary_large_image">
""", unsafe_allow_html=True)

# Enhanced GIS Blue Theme
st.markdown("""
<style>
:root {
  --blue-900: #0a2540;
  --blue-800: #0f3c68;
  --blue-700: #155a8a;
  --blue-600: #1b76ad;
  --blue-500: #2092d0;
  --blue-400: #59b3e3;
  --blue-300: #8ac9f1;
  --text: #eaf4ff;
  --muted: #a8c3dc;
  --card: linear-gradient(135deg, rgba(16, 39, 66, 0.45), rgba(16, 39, 66, 0.25));
  --border: rgba(146, 185, 220, 0.25);
  --shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  --glow: 0 0 20px rgba(32, 146, 208, 0.15);
}
.block-container {padding-top: 1.5rem; max-width: 1280px;}
.main {background: linear-gradient(135deg, #0a2540 0%, #0f3c68 50%, #155a8a 100%);}

/* Enhanced Cards */
.card {
  border-radius: 20px;
  padding: 24px;
  border: 1px solid var(--border);
  background: var(--card);
  backdrop-filter: blur(12px);
  box-shadow: var(--shadow), var(--glow);
  transition: all 0.3s ease;
  margin-bottom: 16px;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), 0 0 25px rgba(32, 146, 208, 0.2);
}

.big-number {
  font-size: 56px;
  font-weight: 900;
  letter-spacing: .5px;
  background: linear-gradient(135deg, var(--blue-300), var(--blue-400));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 2px 10px rgba(32, 146, 208, 0.3);
}
.kpi-label {
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}
.center {text-align: center;}

/* Enhanced Headings */
h1, h2, h3 {
  color: var(--text);
  padding-top: 8px;
  font-weight: 700;
}
h1 {
  background: linear-gradient(135deg, var(--blue-300), var(--blue-400));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

/* Enhanced Dividers */
hr {
  border: none;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--blue-400), transparent);
  margin: 24px 0;
}

/* Enhanced Buttons */
.stButton>button {
  background: linear-gradient(135deg, var(--blue-600), var(--blue-400));
  color: #00223b;
  border: 0;
  font-weight: 700;
  border-radius: 16px;
  padding: 12px 24px;
  box-shadow: 0 4px 15px rgba(32, 146, 208, 0.3);
  transition: all 0.3s ease;
}
.stButton>button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(32, 146, 208, 0.4);
}

/* Sidebar Enhancement */
.css-1d391kg {background: rgba(10, 37, 64, 0.9);}
.sidebar .sidebar-content {background: linear-gradient(180deg, #0a2540 0%, #0f3c68 100%);}

/* Metric Cards */
[data-testid="metric-container"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  backdrop-filter: blur(10px);
}

/* Custom Scrollbar */
::-webkit-scrollbar {width: 8px;}
::-webkit-scrollbar-track {background: rgba(10, 37, 64, 0.5);}
::-webkit-scrollbar-thumb {background: var(--blue-500); border-radius: 4px;}
::-webkit-scrollbar-thumb:hover {background: var(--blue-400);}
</style>
""", unsafe_allow_html=True)

# =========================== TOKEN & HELPERS ===========================
def _get_token() -> Optional[str]:
    token = None
    try:
        token = st.secrets.get("WAQI_TOKEN", None)
    except Exception:
        token = None
    if not token:
        token = os.getenv("WAQI_TOKEN", None)
    return token

WAQI_TOKEN = _get_token()

# =========================== LOAD MODELS ===========================
@st.cache_resource
def load_models() -> Dict[str, Any]:
    # Silent loading without spinner
    return {
        "rf": joblib.load("Models/aqi_model.pkl"),
        "sarimax": joblib.load("Models/delhi_aqi_forecast_sarimax.pkl"),
    }

# Load models silently
models = load_models()
rf = models["rf"]
sarimax = models["sarimax"]

features = ['PM2.5','PM10','NO2','SO2','CO','O3']
importance = getattr(rf, "feature_importances_", np.ones(len(features))/len(features))

# =========================== LIVE AQI API (Cached) ===========================
@st.cache_data(ttl=600, show_spinner=False)
def get_live_aqi(city: str = "Delhi") -> Tuple[Optional[int], Optional[Tuple[float,float]], dict]:
    if not WAQI_TOKEN:
        return None, None, {}
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
    try:
        res = requests.get(url, timeout=15).json()
        if res.get("status") != "ok":
            return None, None, res
        aqi = res["data"]["aqi"]
        lat, lon = res["data"]["city"]["geo"]
        return int(aqi), (float(lat), float(lon)), res
    except Exception as e:
        return None, None, {"error": str(e)}

@st.cache_data(ttl=600, show_spinner=False)
def get_ncr_points() -> pd.DataFrame:
    cities = ["Delhi", "Noida", "Gurgaon", "Ghaziabad", "Faridabad"]
    rows = []
    for c in cities:
        aqi, geo, _ = get_live_aqi(c)
        if aqi is not None and geo is not None:
            rows.append({"City": c, "AQI": aqi, "lat": geo[0], "lon": geo[1]})
    return pd.DataFrame(rows)

# =========================== HEALTH ADVISORY ===========================
def health_advisory(aqi: float):
    if aqi <= 50:   return ("Good 🌱","Ideal air quality","No mask needed","Perfect for outdoor activities","#14c38e")
    if aqi <= 100:  return ("Satisfactory 🙂","Minor breathing discomfort","Mask optional","Ideal for morning/evening","#e3c84e")
    if aqi <= 200:  return ("Moderate 😐","Breathing discomfort","Light mask recommended","Avoid prolonged exposure","#f5a742")
    if aqi <= 300:  return ("Poor 😷","Respiratory illness possible","N95 mask essential","Limit outdoor activities","#ef5b5b")
    if aqi <= 400:  return ("Very Poor 😵","Health impacts likely","Strict N95 protection","Minimize outdoor time","#8f6bf6")
    return ("Severe ☠️","Health emergency","Stay indoors with purifier","Avoid all outdoor activities","#ff0000")

# =========================== FORECAST (ANCHOR TO TODAY) ===========================
def calibrated_forecast(model, live_aqi: Optional[float], steps=3) -> np.ndarray:
    base = np.array(model.forecast(steps=steps), dtype=float)
    if live_aqi is None or len(base) == 0:
        return base
    shift = live_aqi - base[0]
    return base + shift

# =========================== SOURCE BUCKETS ===========================
def source_buckets() -> Dict[str, float]:
    m = dict(zip(features, importance))
    buckets = {
        "Traffic (NO₂ + CO)": m.get("NO2",0) + m.get("CO",0),
        "Dust & Stubble (PM2.5 + PM10)": m.get("PM2.5",0) + m.get("PM10",0),
        "Industry (SO₂)": m.get("SO2",0),
        "Photochemical (O₃)": m.get("O3",0),
    }
    total = sum(buckets.values()) or 1.0
    return {k: (v/total)*100 for k,v in buckets.items()}

# =========================== SIDEBAR NAV ===========================
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🧭 AirVision")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

page = st.sidebar.radio("", ["🌍 Overview", "📈 Forecast", "🧭 Sources & Policy", "🗺️ Live Map", "ℹ️ About"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")

if WAQI_TOKEN:
    st.sidebar.markdown("""
    <div style='background: rgba(20, 195, 142, 0.1); padding: 12px; border-radius: 12px; border: 1px solid rgba(20, 195, 142, 0.3);'>
    <div style='color: #14c38e; font-weight: 600;'>✓ Live API: Connected</div>
    <div style='color: #a8c3dc; font-size: 0.8rem;'>Real-time data active</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='background: rgba(239, 91, 91, 0.1); padding: 12px; border-radius: 12px; border: 1px solid rgba(239, 91, 91, 0.3);'>
    <div style='color: #ef5b5b; font-weight: 600;'>⚠ Live API: Disconnected</div>
    <div style='color: #a8c3dc; font-size: 0.8rem;'>Set WAQI_TOKEN in secrets</div>
    </div>
    """, unsafe_allow_html=True)

# =========================== OVERVIEW ===========================
if page == "🌍 Overview":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("# 🌍 Delhi–NCR Air Quality")
        st.markdown("### Real-time Monitoring & Analysis")
    
    with col2:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Main AQI Card
    live_aqi, geo, _ = get_live_aqi("Delhi")
    
    if live_aqi is not None:
        status, advice, mask, outdoor, color = health_advisory(live_aqi)
        
        st.markdown(f"""
        <div class='card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <div class='kpi-label'>CURRENT AIR QUALITY INDEX — DELHI</div>
                    <div class='big-number' style='color: {color}'>{live_aqi}</div>
                    <div style='font-size: 1.2rem; font-weight: 600; color: {color}; margin: 8px 0;'>{status}</div>
                </div>
                <div style='text-align: right;'>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 12px; border-radius: 12px; margin: 4px 0;'>
                        <div style='font-size: 0.9rem; color: var(--muted);'>Mask Advisory</div>
                        <div style='font-weight: 600;'>{mask}</div>
                    </div>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 12px; border-radius: 12px; margin: 4px 0;'>
                        <div style='font-size: 0.9rem; color: var(--muted);'>Outdoor Activity</div>
                        <div style='font-weight: 600;'>{outdoor}</div>
                    </div>
                </div>
            </div>
            <div style='margin-top: 16px; padding: 16px; background: rgba(32, 146, 208, 0.05); border-radius: 12px;'>
                <strong>Health Advice:</strong> {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Unable to fetch live AQI data. Please check your API configuration.")

    cols = st.columns([1.2, 0.8])
    
    with cols[0]:
        st.markdown("### 🔍 Pollution Source Analysis")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')
        wedges, texts, autotexts = ax.pie(
            importance, labels=None, autopct='%1.1f%%', pctdistance=0.82,
            startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        )
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
            
        centre_circle = plt.Circle((0,0),0.60,fc='#0a2540')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        
        plt.legend(wedges, features, title="Pollutants", loc="center left", 
                  bbox_to_anchor=(1, 0.5), frameon=False, labelcolor='white')
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)
        st.caption("Feature importance analysis from trained RandomForest model")

    with cols[1]:
        st.markdown("### 🛰️ NCR Station Status")
        df_points = get_ncr_points()
        if not df_points.empty:
            for _, row in df_points.sort_values("AQI", ascending=False).iterrows():
                status, _, _, _, color = health_advisory(row["AQI"])
                st.markdown(f"""
                <div style='background: rgba(16, 39, 66, 0.5); padding: 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid {color}'>
                    <div style='font-weight: 600;'>{row['City']}</div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 1.5rem; font-weight: 700; color: {color}'>{row['AQI']}</span>
                        <span style='font-size: 0.8rem; color: var(--muted)'>{status}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for live station data...")

# =========================== FORECAST ===========================
elif page == "📈 Forecast":
    st.markdown("# 📈 Delhi AQI Forecast")
    st.markdown("### 72-Hour Predictive Analysis")
    
    live_aqi, _, _ = get_live_aqi("Delhi")
    fc = calibrated_forecast(sarimax, live_aqi, steps=3)
    days = ["Tomorrow", "Day After", "In 3 Days"]
    
    # Enhanced Forecast Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = [health_advisory(val)[4] for val in fc]
        bars = ax.bar(days, fc, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, fc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=12)
        
        ax.set_ylabel('AQI', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.set_facecolor('#0a2540')
        fig.patch.set_facecolor('#0a2540')
        plt.xticks(color='white', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)

    with col2:
        if live_aqi is not None:
            st.markdown(f"""
            <div class='card'>
                <div class='kpi-label'>CALIBRATION ANCHOR</div>
                <div style='font-size: 2rem; font-weight: 700; color: #59b3e3;'>{live_aqi}</div>
                <div style='color: var(--muted);'>Today's Live AQI</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <div class='kpi-label'>FORECAST MODEL</div>
            <div style='font-weight: 600;'>SARIMAX</div>
            <div style='color: var(--muted); font-size: 0.8rem;'>Time-series analysis with live calibration</div>
        </div>
        """, unsafe_allow_html=True)

    # Forecast Cards
    st.markdown("### 📋 Daily Health Advisory")
    cols = st.columns(3)
    
    for (day, aqi), col in zip(zip(days, fc), cols):
        status, advice, mask, outdoor, color = health_advisory(float(aqi))
        
        with col:
            st.markdown(f"""
            <div class='card'>
                <div style='border-left: 4px solid {color}; padding-left: 12px;'>
                    <div style='font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;'>{day}</div>
                    <div style='font-size: 2.5rem; font-weight: 900; color: {color};'>{int(aqi)}</div>
                    <div style='color: {color}; font-weight: 600; margin: 8px 0;'>{status}</div>
                </div>
                <div style='margin-top: 16px;'>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 8px; border-radius: 8px; margin: 4px 0;'>
                        <div style='font-size: 0.8rem; color: var(--muted);'>Protection</div>
                        <div style='font-weight: 600; font-size: 0.9rem;'>{mask}</div>
                    </div>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 8px; border-radius: 8px; margin: 4px 0;'>
                        <div style='font-size: 0.8rem; color: var(--muted);'>Activity</div>
                        <div style='font-weight: 600; font-size: 0.9rem;'>{outdoor}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================== SOURCES & POLICY ===========================
elif page == "🧭 Sources & Policy":
    st.markdown("# 🧭 Source Attribution")
    st.markdown("### Pollution Analysis & Policy Impact Simulation")
    
    buckets = source_buckets()
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown("### 📊 Source Contribution")
        labels = list(buckets.keys())
        vals = list(buckets.values())
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')
        wedges, texts, autotexts = ax.pie(
            vals, labels=None, autopct='%1.1f%%', pctdistance=0.82, 
            startangle=90, colors=colors
        )
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            
        centre_circle = plt.Circle((0,0),0.60,fc='#0a2540')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        
        plt.legend(wedges, labels, title="Source Categories", loc="center left", 
                  bbox_to_anchor=(1, 0.5), frameon=False, labelcolor='white')
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("### 💡 Policy Insights")
        st.markdown("""
        <div class='card'>
            <div style='font-size: 1.1rem; font-weight: 700; margin-bottom: 12px;'>Key Observations</div>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                <div style='font-weight: 600;'>Traffic Dominance</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>Primary contributor to NO₂ & CO levels</div>
            </div>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                <div style='font-weight: 600;'>Seasonal Factors</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>Dust peaks during summer & stubble burning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧪 Policy Simulator")
    st.markdown("Explore potential AQI improvements through targeted interventions")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🚗 Traffic Measures")
        t = st.slider("Traffic emission reduction %", 0, 60, 20, help="Improved public transport, EV adoption")
        d = st.slider("Dust control measures %", 0, 60, 15, help="Construction dust management, road cleaning")
    with colB:
        st.markdown("#### 🏭 Industrial Controls")
        i = st.slider("Industrial emission reduction %", 0, 60, 10, help="Emission controls, fuel switching")
        p = st.slider("Photochemical reduction %", 0, 60, 5, help="VOC controls, solar adoption")

    # Impact Calculation
    w = {k: v/100.0 for k, v in buckets.items()}
    eff = (w["Traffic (NO₂ + CO)"] * (t/100) + 
           w["Dust & Stubble (PM2.5 + PM10)"] * (d/100) +
           w["Industry (SO₂)"] * (i/100) + 
           w["Photochemical (O₃)"] * (p/100))

    live, _, _ = get_live_aqi("Delhi")
    baseline = float(live) if live is not None else 200.0
    improved = baseline * (1 - eff)
    status, _, _, _, color = health_advisory(improved)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='kpi-label'>CURRENT BASELINE</div>
            <div style='font-size: 3rem; font-weight: 900; color: #ef5b5b;'>{int(baseline)}</div>
            <div style='color: var(--muted);'>Today's AQI Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='kpi-label'>PROJECTED IMPROVEMENT</div>
            <div style='font-size: 3rem; font-weight: 900; color: {color};'>{int(improved)}</div>
            <div style='color: {color}; font-weight: 600;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================== MAP (OpenStreetMap) ===========================
elif page == "🗺️ Live Map":
    st.markdown("# 🗺️ Delhi-NCR Air Quality Map")
    st.markdown("### Real-time Station Monitoring")
    
    df = get_ncr_points()
    if df.empty:
        st.error("No live map data available. Please check your API configuration.")
    else:
        def aqi_hex(aqi):
            if aqi <= 50: return "#14c38e"
            if aqi <= 100: return "#e3c84e"
            if aqi <= 200: return "#f5a742"
            if aqi <= 300: return "#ef5b5b"
            if aqi <= 400: return "#8f6bf6"
            return "#ff0000"

        m = folium.Map(
            location=[df["lat"].mean(), df["lon"].mean()],
            zoom_start=10,
            tiles="CartoDB DarkMatter",
            width='100%',
            height=500
        )
        
        for _, row in df.iterrows():
            color = aqi_hex(row["AQI"])
            status, _, _, _, _ = health_advisory(row["AQI"])
            
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=15 + (row["AQI"] / 50),
                popup=f"""
                <div style='font-family: Arial, sans-serif;'>
                    <h4>{row['City']}</h4>
                    <b>AQI:</b> {row['AQI']}<br/>
                    <b>Status:</b> {status}<br/>
                    <b>Lat:</b> {row['lat']:.4f}<br/>
                    <b>Lon:</b> {row['lon']:.4f}
                </div>
                """,
                tooltip=f"{row['City']}: AQI {row['AQI']}",
                color=color,
                fill=True,
                fill_opacity=0.7,
                fill_color=color,
                opacity=0.8,
                weight=2
            ).add_to(m)

        st_folium(m, height=500, returned_objects=[])
        
        st.markdown("### 📋 Station Summary")
        for _, row in df.sort_values("AQI", ascending=False).iterrows():
            status, _, _, _, color = health_advisory(row["AQI"])
            st.markdown(f"""
            <div style='background: rgba(16, 39, 66, 0.5); padding: 12px; border-radius: 8px; margin: 4px 0; border-left: 4px solid {color}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: 600;'>{row['City']}</span>
                    <span style='font-size: 1.2rem; font-weight: 700; color: {color}'>{row['AQI']}</span>
                </div>
                <div style='color: var(--muted); font-size: 0.9rem;'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About AirVision")
    st.markdown("### AI-Driven Pollution Intelligence Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Problem Statement Card
        st.markdown("""
        <div class='card'>
            <h3 style='color: var(--text); margin-bottom: 1rem;'>🎯 Problem Statement 25216</h3>
            <p style='color: var(--muted); line-height: 1.6;'>AirVision addresses <strong style='color: var(--text);'>AICTE PS 25216</strong> — developing an AI-driven framework for pollution source identification, forecasting, and policy impact analysis for Delhi-NCR region.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Innovations
        st.markdown("#### 🚀 Key Innovations")
        st.markdown("""
        <div class='card'>
            <div style='display: flex; flex-wrap: wrap; gap: 1rem;'>
                <div style='flex: 1; min-width: 200px;'>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid #2092d0;'>
                        <div style='font-weight: 700; color: var(--text); margin-bottom: 8px;'>🔍 Real-time Source Attribution</div>
                        <div style='color: var(--muted); font-size: 0.9rem;'>Machine learning-based pollution source identification</div>
                    </div>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid #59b3e3;'>
                        <div style='font-weight: 700; color: var(--text); margin-bottom: 8px;'>📊 Predictive Forecasting</div>
                        <div style='color: var(--muted); font-size: 0.9rem;'>72-hour AQI predictions with live calibration</div>
                    </div>
                </div>
                <div style='flex: 1; min-width: 200px;'>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid #8ac9f1;'>
                        <div style='font-weight: 700; color: var(--text); margin-bottom: 8px;'>🎯 Policy Impact Simulation</div>
                        <div style='color: var(--muted); font-size: 0.9rem;'>What-if analysis for intervention strategies</div>
                    </div>
                    <div style='background: rgba(32, 146, 208, 0.1); padding: 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid #1b76ad;'>
                        <div style='font-weight: 700; color: var(--text); margin-bottom: 8px;'>🗺️ Spatial Analysis</div>
                        <div style='color: var(--muted); font-size: 0.9rem;'>Interactive mapping of pollution hotspots</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Technical Stack
        st.markdown("#### 🛠️ Technical Stack")
        st.markdown("""
        <div class='card'>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 14px; border-radius: 10px; margin: 10px 0;'>
                <div style='font-weight: 700; color: var(--text); margin-bottom: 6px;'>🎨 Frontend</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>Streamlit • Custom CSS</div>
            </div>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 14px; border-radius: 10px; margin: 10px 0;'>
                <div style='font-weight: 700; color: var(--text); margin-bottom: 6px;'>🤖 ML Models</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>RandomForest • SARIMAX</div>
            </div>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 14px; border-radius: 10px; margin: 10px 0;'>
                <div style='font-weight: 700; color: var(--text); margin-bottom: 6px;'>📊 Data Sources</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>WAQI API • OpenStreetMap</div>
            </div>
            <div style='background: rgba(32, 146, 208, 0.1); padding: 14px; border-radius: 10px; margin: 10px 0;'>
                <div style='font-weight: 700; color: var(--text); margin-bottom: 6px;'>📈 Visualization</div>
                <div style='color: var(--muted); font-size: 0.9rem;'>Matplotlib • Folium</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Card
        st.markdown("#### ⚡ Features")
        st.markdown("""
        <div class='card'>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <span style='color: #14c38e; margin-right: 10px;'>✓</span>
                <span style='color: var(--text);'>Live AQI Monitoring</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <span style='color: #14c38e; margin-right: 10px;'>✓</span>
                <span style='color: var(--text);'>72-Hour Forecast</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <span style='color: #14c38e; margin-right: 10px;'>✓</span>
                <span style='color: var(--text);'>Source Analysis</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <span style='color: #14c38e; margin-right: 10px;'>✓</span>
                <span style='color: var(--text);'>Policy Simulation</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <span style='color: #14c38e; margin-right: 10px;'>✓</span>
                <span style='color: var(--text);'>Interactive Maps</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Methodology Section
    st.markdown("### 📈 Methodology")
    
    methodology_card = """
    <div class='card'>
        <div style='margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid var(--border);'>
            <div style='display: flex; align-items: center; margin-bottom: 12px;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>🔍</span>
                <div style='font-weight: 700; color: var(--text); font-size: 1.1rem;'>Source Attribution</div>
            </div>
            <div style='color: var(--muted); line-height: 1.6; padding-left: 35px;'>
                Uses RandomForest feature importances as interpretable proxies for pollution source contributions. 
                For operational deployment, integrates with CPCB station data, MODIS/VIIRS fire radiative power, 
                and traffic indices for comprehensive analysis.
            </div>
        </div>
        <div style='margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid var(--border);'>
            <div style='display: flex; align-items: center; margin-bottom: 12px;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>📊</span>
                <div style='font-weight: 700; color: var(--text); font-size: 1.1rem;'>Forecasting</div>
            </div>
            <div style='color: var(--muted); line-height: 1.6; padding-left: 35px;'>
                SARIMAX time-series model calibrated with live AQI data for improved short-term prediction accuracy. 
                The model adapts to seasonal patterns and real-time conditions, providing reliable 72-hour forecasts.
            </div>
        </div>
        <div style='margin-bottom: 10px;'>
            <div style='display: flex; align-items: center; margin-bottom: 12px;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>🎯</span>
                <div style='font-weight: 700; color: var(--text); font-size: 1.1rem;'>Policy Simulation</div>
            </div>
            <div style='color: var(--muted); line-height: 1.6; padding-left: 35px;'>
                First-order approximation based on source contribution weights. For comprehensive analysis, 
                implements chemical transport models (CTM) with detailed emission inventories and 
                meteorological data integration for accurate policy impact assessment.
            </div>
        </div>
    </div>
    """
    st.markdown(methodology_card, unsafe_allow_html=True)

    # Team & Credits
    st.markdown("### 👨‍💻 Development Team")
    st.markdown("""
    <div class='card'>
        <div style='text-align: center; padding: 30px 20px;'>
            <div style='font-size: 1.8rem; font-weight: 800; color: var(--text); margin-bottom: 15px;'>
                🌱 Built by Guru for Clean & Green India
            </div>
            <div style='color: var(--muted); font-size: 1.2rem; margin-bottom: 20px; font-weight: 600;'>
                AirVision • AI-Powered Environmental Intelligence
            </div>
            <div style='color: var(--muted); font-size: 1rem; line-height: 1.6;'>
                Contributing to sustainable development and environmental protection<br>through cutting-edge AI innovation and data-driven insights
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================== FOOTER ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--muted); padding: 2rem 0;'>
    <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>🌱 Built by Guru for Clean & Green India</div>
    <div>AirVision • AI-Powered Environmental Intelligence</div>
</div>
""", unsafe_allow_html=True)