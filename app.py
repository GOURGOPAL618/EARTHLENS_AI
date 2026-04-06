"""
EARTHLENS AI — Main Application (Enterprise Mission Control)
==============================================================
Author : Gouragopal Mohapatra
Purpose: Streamlit UI — Premium Satellite Intelligence Platform
"""

import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from earthlens_config.settings import SETTINGS
from earthlens_core.analysis_engine.preprocessing import preprocess_band
from earthlens_core.analysis_engine.ndvi import run_ndvi_pipeline
from earthlens_core.analysis_engine.water_detection import run_water_pipeline
from earthlens_core.visualization_hub.map_view import ndvi_map, water_map
from earthlens_core.visualization_hub.plots import (
    ndvi_histogram,
    ndvi_coverage_pie,
    ndvi_stats_bar,
    water_coverage_bar,
    ndvi_dashboard,
)
from streamlit_folium import st_folium
from loguru import logger

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarthLens AI | Mission Control Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================================
# ENTERPRISE-GRADE CSS - NASA/SpaceX Mission Control Style
# =================================================================================
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* ========== GLOBAL DARK THEME ========== */
    .stApp {
        background: linear-gradient(135deg, #070B14 0%, #0A0F1A 50%, #070B14 100%);
    }
    
    /* ========== MISSION CONTROL HEADER ========== */
    .mission-header {
        background: linear-gradient(135deg, 
            rgba(8, 20, 35, 0.95) 0%,
            rgba(12, 28, 45, 0.95) 50%,
            rgba(8, 20, 35, 0.95) 100%);
        backdrop-filter: blur(16px);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 255, 255, 0.25);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,255,255,0.1);
        position: relative;
    }
    
    .mission-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 10%;
        width: 80%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFFF, #00FF88, #00FFFF, transparent);
    }
    
    .mission-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #00FFFF 40%, #00FF88 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .mission-status {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(0,0,0,0.5);
        padding: 8px 18px;
        border-radius: 40px;
        border: 1px solid rgba(0,255,255,0.3);
    }
    
    .live-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #00FF88;
    }
    
    .pulse {
        width: 10px;
        height: 10px;
        background: #00FF88;
        border-radius: 50%;
        box-shadow: 0 0 10px #00FF88;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
    }
    
    .mission-subtitle {
        font-family: 'Inter', sans-serif;
        color: #88AACC;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    
    .tech-badges {
        display: flex;
        gap: 12px;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }
    
    .tech-badge {
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(0,255,136,0.05));
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 30px;
        padding: 5px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: #00FFFF;
        letter-spacing: 0.5px;
    }
    
    /* ========== ENTERPRISE METRIC CARDS ========== */
    .dashboard-card {
        background: linear-gradient(135deg, 
            rgba(15, 25, 35, 0.85) 0%,
            rgba(10, 18, 28, 0.85) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,255,255,0.08), transparent);
        transition: left 0.5s;
    }
    
    .dashboard-card:hover::before {
        left: 100%;
    }
    
    .dashboard-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,255,255,0.4);
        box-shadow: 0 12px 28px rgba(0,255,255,0.15);
    }
    
    .card-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .card-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #88AACC;
        margin-bottom: 0.3rem;
    }
    
    .card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF, #00FFFF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    /* ========== ENTERPRISE SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(5, 12, 20, 0.98) 0%,
            rgba(3, 8, 15, 0.98) 100%);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
        box-shadow: 5px 0 30px rgba(0,0,0,0.3);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid rgba(0,255,255,0.15);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-header h2 {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00FFFF, #00FF88);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
    }
    
    .sidebar-header p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #556677;
        margin-top: 0.3rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00FFFF;
        margin: 1rem 0 0.8rem 0;
        padding-left: 0.5rem;
        border-left: 3px solid #00FFFF;
    }
    
    /* Sidebar Inputs */
    .stSelectbox label, .stRadio label, .stDateInput label, .stNumberInput label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        color: #88AACC !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: rgba(10, 20, 30, 0.8);
        border-color: rgba(0, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    .stRadio > div {
        background: rgba(10, 20, 30, 0.6);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 12px;
        padding: 0.8rem;
    }
    
    /* Enterprise Run Button */
    .stButton button {
        background: linear-gradient(135deg, #00AA88, #0066AA);
        border: none;
        border-radius: 40px;
        padding: 0.8rem;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: white;
        transition: all 0.3s ease;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,170,136,0.4);
    }
    
    /* Sidebar Footer */
    .sidebar-footer {
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid rgba(0,255,255,0.15);
        text-align: center;
    }
    
    .developer-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #00FFFF;
        margin: 0.3rem 0;
    }
    
    .copyright-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        color: #556677;
    }
    
    /* ========== ENTERPRISE FOOTER ========== */
    .enterprise-footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, 
            rgba(5, 12, 20, 0.95) 0%,
            rgba(3, 8, 15, 0.95) 100%);
        border-radius: 20px;
        border-top: 1px solid rgba(0,255,255,0.2);
        text-align: center;
        position: relative;
    }
    
    .enterprise-footer::before {
        content: '';
        position: absolute;
        top: -1px;
        left: 5%;
        width: 90%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00FFFF, #00FF88, #00FFFF, transparent);
    }
    
    .footer-brand {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF, #00FFFF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .footer-developer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #00FF88;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .footer-tech {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .footer-tech span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6688AA;
    }
    
    .footer-copyright {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        color: #445566;
        margin-top: 0.8rem;
    }
    
    /* ========== DIVIDERS ========== */
    .glow-divider {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,255,255,0.3), rgba(0,255,136,0.3), rgba(0,255,255,0.3), transparent);
    }
    
    /* ========== SECTION TITLES ========== */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF, #00FFFF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* ========== DATAFRAME ========== */
    .stDataFrame {
        background: rgba(10, 20, 30, 0.5);
        border-radius: 14px;
        border: 1px solid rgba(0, 255, 255, 0.1);
    }
    
    /* ========== ALERTS ========== */
    .stAlert {
        background: rgba(0, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* ========== METRICS ========== */
    [data-testid="stMetric"] {
        background: rgba(15, 25, 35, 0.6);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(0, 255, 255, 0.1);
    }
    
    [data-testid="stMetric"] label {
        color: #88AACC !important;
        font-weight: 600;
    }
    
    [data-testid="stMetric"] value {
        color: #00FFFF !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── MISSION CONTROL HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div class="mission-header">
    <div class="mission-title">
        <span>🛰️ EARTHLENS AI</span>
        <div class="mission-status">
            <div class="live-badge">
                <div class="pulse"></div>
                <span>LIVE MISSION</span>
            </div>
        </div>
    </div>
    <div class="mission-subtitle">
        Satellite Intelligence Platform | Advanced Remote Sensing & Geospatial Analytics
    </div>
    <div class="tech-badges">
        <span class="tech-badge">⚡ REAL-TIME PROCESSING</span>
        <span class="tech-badge">🛰️ MULTI-SPECTRAL IMAGERY</span>
        <span class="tech-badge">🧠 DEEP LEARNING MODELS</span>
        <span class="tech-badge">📡 10M RESOLUTION</span>
        <span class="tech-badge">🌍 GEE INTEGRATION</span>
        <span class="tech-badge">🎯 AI-POWERED ANALYTICS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = Path("earthlens_data/raw_imagery")
PROCESSED_DIR = Path("earthlens_data/processed_insights")

BAND_PATHS = {
    "B02": RAW_DIR / "sentinel2_delhi_B02.tif",
    "B03": RAW_DIR / "sentinel2_delhi_B03.tif",
    "B04": RAW_DIR / "sentinel2_delhi_B04.tif",
    "B08": RAW_DIR / "sentinel2_delhi_B08.tif",
    "B11": RAW_DIR / "sentinel2_delhi_B11.tif",
    "B12": RAW_DIR / "sentinel2_delhi_B12.tif",
}

# ── ENTERPRISE SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🎮 MISSION CONTROL</h2>
        <p>Analysis Configuration Terminal</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 ANALYSIS PROTOCOL")
    analysis = st.selectbox(
        "Select Analysis Type",
        options=[
            "🌿 NDVI — Vegetation Analysis",
            "💧 Water Detection",
            "🏙️ Land Use Classification",
            "🔄 Change Detection",
            "🔥 Burn Area Detection",
            "🏗️ Urban Expansion",
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("### 🛰️ DATA SOURCE")
    data_source = st.radio(
        "Select Source",
        options=[
            "📁 Synthetic (Local)",
            "🌍 GEE — Sentinel-2",
            "🌍 GEE — Landsat",
            "🛰️ Sentinel-2 (Live)",
            "🌍 Landsat (Live)",
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("### 📅 TEMPORAL RANGE")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("FROM", value=pd.Timestamp("2024-01-01"), label_visibility="collapsed")
    with col2:
        date_to = st.date_input("TO", value=pd.Timestamp("2024-03-01"), label_visibility="collapsed")
    
    st.markdown("### 📍 AOI BOUNDING BOX")
    col_a, col_b = st.columns(2)
    with col_a:
        west = st.number_input("WEST", value=77.05, step=0.01, format="%.2f", label_visibility="collapsed")
        south = st.number_input("SOUTH", value=28.40, step=0.01, format="%.2f", label_visibility="collapsed")
    with col_b:
        east = st.number_input("EAST", value=77.35, step=0.01, format="%.2f", label_visibility="collapsed")
        north = st.number_input("NORTH", value=28.75, step=0.01, format="%.2f", label_visibility="collapsed")
    
    bbox_coords = (west, south, east, north)
    
    st.markdown("### 📡 DATASET STATUS")
    if "Synthetic" in data_source:
        st.success("🟢 Sentinel-2 Synthetic • 6 Bands Ready")
    elif "GEE" in data_source:
        st.success("🟢 Google Earth Engine • Connected")
    else:
        st.warning("🟡 Live API • Credentials Required")
    
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀 EXECUTE MISSION", use_container_width=True)
    
    # Sidebar Footer
    st.markdown("""
    <div class="sidebar-footer">
        <div class="developer-name">GOURGOPAL MOHAPATRA</div>
        <div class="copyright-text">© 2026 EarthLens AI</div>
        <div class="copyright-text" style="font-size: 0.55rem;">Satellite Intelligence Division</div>
    </div>
    """, unsafe_allow_html=True)

# ── ENTERPRISE DASHBOARD CARDS ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-icon">🛰️</div>
        <div class="card-label">PRIMARY SOURCE</div>
        <div class="card-value">Sentinel-2</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-icon">📐</div>
        <div class="card-label">SPATIAL RESOLUTION</div>
        <div class="card-value">10m / pixel</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-icon">📍</div>
        <div class="card-label">CURRENT AOI</div>
        <div class="card-value">Delhi, India</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-icon">🎨</div>
        <div class="card-label">SPECTRAL BANDS</div>
        <div class="card-value">6 Active</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────────
if "ndvi_result" not in st.session_state:
    st.session_state.ndvi_result = None
if "water_result" not in st.session_state:
    st.session_state.water_result = None
if "landuse_result" not in st.session_state:
    st.session_state.landuse_result = None
if "change_result" not in st.session_state:
    st.session_state.change_result = None
if "burn_result" not in st.session_state:
    st.session_state.burn_result = None
if "urban_result" not in st.session_state:
    st.session_state.urban_result = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# ── Data Loader (UNCHANGED) ────────────────────────────────────────────────────
def load_bands_by_source(data_source, bbox_coords, date_from, date_to):
    from earthlens_core.data_pipeline.sentinel_api import fetch_sentinel2_bands
    from earthlens_core.data_pipeline.landsat_api import run_landsat_pipeline

    date_from_str = str(date_from)
    date_to_str = str(date_to)

    if "Sentinel-2 (Live)" in data_source:
        try:
            bands = fetch_sentinel2_bands(
                bbox_coords=bbox_coords,
                date_from=date_from_str,
                date_to=date_to_str,
                output_dir=PROCESSED_DIR,
            )
            bands.pop("meta", None)
            st.success("✅ Live Sentinel-2 data acquired!")
            return bands
        except Exception as e:
            st.warning(f"⚠️ Live API failed: {e}\n\nUsing synthetic data...")
            return _load_synthetic()

    elif "Landsat (Live)" in data_source:
        try:
            band_paths = run_landsat_pipeline(
                bbox_coords=bbox_coords,
                date_from=date_from_str,
                date_to=date_to_str,
                output_dir=RAW_DIR,
            )
            bands = {}
            for b in ["B02", "B03", "B04", "B08", "B11", "B12"]:
                if b in band_paths:
                    with rasterio.open(band_paths[b]) as src:
                        bands[b] = src.read(1).astype(np.float32)
            st.success("✅ Live Landsat data acquired!")
            return bands
        except Exception as e:
            st.warning(f"⚠️ Live API failed: {e}\n\nUsing synthetic data...")
            return _load_synthetic()

    elif "GEE — Sentinel-2" in data_source:
        st.info("🌍 Accessing GEE Sentinel-2 archive...")
        try:
            from earthlens_core.data_pipeline.gee_api import run_gee_pipeline
            bands = run_gee_pipeline(
                bbox_coords=bbox_coords,
                date_from=date_from_str,
                date_to=date_to_str,
                satellite="sentinel2",
            )
            bands.pop("meta", None)
            bands.pop("source", None)
            st.success("✅ GEE Sentinel-2 data acquired!")
            return bands
        except Exception as e:
            st.warning(f"⚠️ GEE failed: {e}\n\nUsing synthetic data...")
            return _load_synthetic()

    elif "GEE — Landsat" in data_source:
        st.info("🌍 Accessing GEE Landsat archive...")
        try:
            from earthlens_core.data_pipeline.gee_api import run_gee_pipeline
            bands = run_gee_pipeline(
                bbox_coords=bbox_coords,
                date_from=date_from_str,
                date_to=date_to_str,
                satellite="landsat",
            )
            bands.pop("meta", None)
            bands.pop("source", None)
            st.success("✅ GEE Landsat data acquired!")
            return bands
        except Exception as e:
            st.warning(f"⚠️ GEE failed: {e}\n\nUsing synthetic data...")
            return _load_synthetic()
    else:
        return _load_synthetic()


def _load_synthetic():
    bands = {}
    for b in ["B02", "B03", "B04", "B08", "B11", "B12"]:
        path = BAND_PATHS[b]
        with rasterio.open(path) as src:
            bands[b] = src.read(1).astype(np.float32)
    return bands


# ── Run Button Logic (UNCHANGED) ───────────────────────────────────────────────
if run_btn:
    st.session_state.last_analysis = analysis
    st.session_state.ndvi_result = None
    st.session_state.water_result = None
    st.session_state.landuse_result = None
    st.session_state.change_result = None
    st.session_state.burn_result = None
    st.session_state.urban_result = None

    with st.spinner("📡 Acquiring satellite data..."):
        bands = load_bands_by_source(data_source, bbox_coords, date_from, date_to)
        st.session_state.bands = bands

    if "NDVI" in analysis:
        with st.spinner("🌿 Processing NDVI algorithm..."):
            from earthlens_core.analysis_engine.ndvi import calculate_ndvi, ndvi_stats, classify_ndvi
            red = bands["B04"].astype(np.float32)
            nir = bands["B08"].astype(np.float32)
            ndvi = calculate_ndvi(red, nir)
            stats = ndvi_stats(ndvi)
            classified = classify_ndvi(ndvi)
            with rasterio.open(BAND_PATHS["B04"]) as src:
                meta = src.meta.copy()
            st.session_state.ndvi_result = {
                "ndvi": ndvi,
                "classified": classified,
                "stats": stats,
                "meta": meta,
                "saved_to": None,
            }

    elif "Water" in analysis:
        with st.spinner("💧 Detecting water bodies..."):
            from earthlens_core.analysis_engine.water_detection import (
                calculate_ndwi,
                calculate_mndwi,
                create_water_mask,
                water_stats,
            )
            green = bands["B03"].astype(np.float32)
            nir = bands["B08"].astype(np.float32)
            swir1 = bands["B11"].astype(np.float32)
            ndwi = calculate_ndwi(green, nir)
            mndwi = calculate_mndwi(green, swir1)
            water_mask = create_water_mask(ndwi, mndwi)
            stats = water_stats(ndwi, water_mask)
            with rasterio.open(BAND_PATHS["B03"]) as src:
                meta = src.meta.copy()
            st.session_state.water_result = {
                "ndwi": ndwi,
                "mndwi": mndwi,
                "water_mask": water_mask,
                "stats": stats,
                "meta": meta,
                "saved_to": None,
            }

    elif "Land Use" in analysis:
        with st.spinner("🏙️ Running land use classification..."):
            from earthlens_core.intelligence_models.classifier import (
                run_classification_pipeline,
                LAND_CLASSES,
                LAND_COLORS,
            )
            st.session_state.landuse_result = run_classification_pipeline(
                bands=bands, output_dir=PROCESSED_DIR,
            )

    elif "Change" in analysis:
        with st.spinner("🔄 Detecting temporal changes..."):
            from earthlens_core.analysis_engine.change_detection import run_change_pipeline
            st.session_state.change_result = run_change_pipeline(
                bands_t1=bands,
                change_type="deforestation",
                method="ndvi",
                output_dir=PROCESSED_DIR,
            )

    elif "Burn" in analysis:
        with st.spinner("🔥 Analyzing burn scars..."):
            from earthlens_core.analysis_engine.burn_area import run_burn_pipeline
            st.session_state.burn_result = run_burn_pipeline(
                bands_prefire=bands, output_dir=PROCESSED_DIR,
            )

    elif "Urban" in analysis:
        with st.spinner("🏗️ Measuring urban expansion..."):
            from earthlens_core.analysis_engine.urban_expansion import run_urban_pipeline
            st.session_state.urban_result = run_urban_pipeline(
                bands_t1=bands, output_dir=PROCESSED_DIR,
            )


# ── Display Results (COMPLETE - ALL ORIGINAL CODE PRESERVED) ───────────────────
if st.session_state.last_analysis and "NDVI" in st.session_state.last_analysis:
    result = st.session_state.ndvi_result
    if result:
        ndvi = result["ndvi"]
        stats = result["stats"]
        meta = result["meta"]

        st.markdown('<div class="section-title">🌿 NDVI Analysis</div>', unsafe_allow_html=True)
        st.success("✅ NDVI Analysis Complete!")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("NDVI Min", f"{stats['min']:.3f}")
        s2.metric("NDVI Max", f"{stats['max']:.3f}")
        s3.metric("NDVI Mean", f"{stats['mean']:.3f}")
        s4.metric("NDVI Std", f"{stats['std']:.3f}")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])
        with col_map:
            st.markdown("### 🗺️ NDVI Map")
            m = ndvi_map(ndvi, meta)
            st_folium(m, width=600, height=400, key="ndvi_map")

        with col_chart:
            st.markdown("### 📊 Coverage")
            st.plotly_chart(ndvi_coverage_pie(stats), use_container_width=True)

        st.divider()
        st.markdown("### 📈 NDVI Dashboard")
        st.plotly_chart(ndvi_dashboard(ndvi, stats), use_container_width=True)

        st.markdown("### 📋 Class Coverage")
        coverage = stats["coverage"]
        st.dataframe(
            {
                "Land Cover Class": list(coverage.keys()),
                "Coverage (%)": list(coverage.values()),
            },
            use_container_width=True,
        )

elif st.session_state.last_analysis and "Water" in st.session_state.last_analysis:
    result = st.session_state.get("water_result", None)
    if result:
        water_mask = result["water_mask"]
        stats = result["stats"]
        meta = result["meta"]

        st.markdown('<div class="section-title">💧 Water Detection</div>', unsafe_allow_html=True)
        st.success("✅ Water Detection Complete!")

        w1, w2, w3 = st.columns(3)
        w1.metric("Water Coverage", f"{stats['water_coverage']}%")
        w2.metric("Water Pixels", f"{stats['water_pixels']:,}")
        w3.metric("Total Pixels", f"{stats['total_pixels']:,}")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])
        with col_map:
            st.markdown("### 🗺️ Water Map")
            m = water_map(water_mask, meta)
            st_folium(m, width=600, height=400, key="water_map")

        with col_chart:
            st.markdown("### 📊 Coverage")
            st.plotly_chart(water_coverage_bar(stats), use_container_width=True)
    else:
        st.info("👈 Execute mission to initialize")

elif st.session_state.last_analysis and "Land Use" in st.session_state.last_analysis:
    from earthlens_core.intelligence_models.classifier import LAND_CLASSES, LAND_COLORS

    result = st.session_state.get("landuse_result", None)

    if result:
        classified = result["classified"]
        stats = result["stats"]
        report = result["report"]

        st.markdown('<div class="section-title">🏙️ Land Use Classification</div>', unsafe_allow_html=True)
        st.success("✅ Classification Complete!")

        cols = st.columns(len(LAND_CLASSES))
        for idx, (cls_id, cls_name) in enumerate(LAND_CLASSES.items()):
            with cols[idx]:
                pct = stats.get(cls_name, {}).get("coverage", 0)
                px = stats.get(cls_name, {}).get("pixels", 0)
                st.metric(cls_name, f"{pct}%", f"{px:,} px")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])

        with col_map:
            st.markdown("### 🗺️ Classification Map")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            class_colors = list(LAND_COLORS.values())
            cmap_custom = mcolors.ListedColormap(class_colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#070B14')
            ax.set_facecolor('#070B14')
            im = ax.imshow(classified, cmap=cmap_custom, norm=norm)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046)
            cbar.ax.set_yticklabels(list(LAND_CLASSES.values()), color='#88AACC', fontsize=8)
            st.pyplot(fig)

        with col_chart:
            st.markdown("### 📊 Coverage")
            import plotly.graph_objects as go

            labels = list(stats.keys())
            values = [stats[k]["coverage"] for k in labels]
            colors = list(LAND_COLORS.values())

            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors,
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        if report:
            st.markdown("### 📋 Classification Report")
            import pandas as pd

            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
    else:
        st.info("👈 Execute mission to initialize")

elif st.session_state.last_analysis and "Change" in st.session_state.last_analysis:
    from earthlens_core.analysis_engine.change_detection import CHANGE_CLASSES, CHANGE_COLORS

    result = st.session_state.get("change_result", None)
    if result:
        st.markdown('<div class="section-title">🔄 Change Detection</div>', unsafe_allow_html=True)
        st.success("✅ Change Detection Complete!")

        stats = result["stats"]
        change_mask = result["change_mask"]
        delta = result["delta"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Change Type", result["change_type"].title())
        c2.metric("Method", result["method"].upper())
        c3.metric("Changed Pixels", f"{int((change_mask > 0).sum()):,}")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])

        with col_map:
            st.markdown("### 🗺️ Change Map")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            colors = list(CHANGE_COLORS.values())
            cmap = mcolors.ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#070B14')

            axes[0].set_facecolor('#0A0F1A')
            im0 = axes[0].imshow(delta, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0].set_title('NDVI Difference', color='#88AACC', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            axes[1].set_facecolor('#0A0F1A')
            im1 = axes[1].imshow(change_mask, cmap=cmap, norm=norm)
            axes[1].set_title('Change Classification', color='#88AACC', fontsize=10)
            axes[1].axis('off')
            cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2, 3, 4, 5])
            cbar.ax.set_yticklabels(list(CHANGE_CLASSES.values()), color='#88AACC', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        with col_chart:
            st.markdown("### 📊 Coverage")
            import plotly.graph_objects as go

            labels = list(stats.keys())
            values = [stats[k]["coverage"] for k in labels]
            colors_pie = list(CHANGE_COLORS.values())
            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors_pie,
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 📋 Change Statistics")
        import pandas as pd

        rows = []
        for cls, data in stats.items():
            rows.append({"Class": cls, "Pixels": f"{data['pixels']:,}", "Coverage %": data["coverage"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif st.session_state.last_analysis and "Burn" in st.session_state.last_analysis:
    from earthlens_core.analysis_engine.burn_area import BURN_CLASSES, BURN_COLORS

    result = st.session_state.get("burn_result", None)
    if result:
        st.markdown('<div class="section-title">🔥 Burn Area Detection</div>', unsafe_allow_html=True)
        st.success("✅ Burn Detection Complete!")

        stats = result["stats"]
        classified = result["classified"]
        dnbr = result["dnbr"]

        b1, b2, b3 = st.columns(3)
        b1.metric("Total Burned", f"{stats['total_burned_coverage']}%")
        b2.metric("dNBR Mean", f"{stats['dnbr_mean']:.3f}")
        b3.metric("Burned Pixels", f"{stats['total_burned_pixels']:,}")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])

        with col_map:
            st.markdown("### 🗺️ Burn Severity Map")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            colors = list(BURN_COLORS.values())
            cmap = mcolors.ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#070B14')

            axes[0].set_facecolor('#0A0F1A')
            im0 = axes[0].imshow(dnbr, cmap='RdYlGn_r', vmin=-0.5, vmax=1.0)
            axes[0].set_title('dNBR', color='#88AACC', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            axes[1].set_facecolor('#0A0F1A')
            im1 = axes[1].imshow(classified, cmap=cmap, norm=norm)
            axes[1].set_title('Burn Severity', color='#88AACC', fontsize=10)
            axes[1].axis('off')
            cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(list(BURN_CLASSES.values()), color='#88AACC', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        with col_chart:
            st.markdown("### 📊 Severity Coverage")
            import plotly.graph_objects as go

            coverage = stats["coverage"]
            fig = go.Figure(
                go.Pie(
                    labels=list(coverage.keys()),
                    values=[v["coverage"] for v in coverage.values()],
                    marker_colors=list(BURN_COLORS.values()),
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 📋 Burn Statistics")
        import pandas as pd

        rows = []
        for cls, data in stats["coverage"].items():
            rows.append({"Severity Class": cls, "Pixels": f"{data['pixels']:,}", "Coverage %": data["coverage"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif st.session_state.last_analysis and "Urban" in st.session_state.last_analysis:
    from earthlens_core.analysis_engine.urban_expansion import URBAN_CLASSES, URBAN_COLORS

    result = st.session_state.get("urban_result", None)
    if result:
        st.markdown('<div class="section-title">🏗️ Urban Expansion</div>', unsafe_allow_html=True)
        st.success("✅ Urban Expansion Analysis Complete!")

        stats = result["stats"]
        change_mask = result["change_mask"]
        ndbi_t1 = result["ndbi_t1"]
        ndbi_t2 = result["ndbi_t2"]

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Urban T1", f"{stats['urban_t1_coverage']}%")
        u2.metric("Urban T2", f"{stats['urban_t2_coverage']}%")
        u3.metric("Expansion", f"{stats['expansion_pct']}%")
        u4.metric("New Urban px", f"{stats['expansion_pixels']:,}")

        st.divider()

        col_map, col_chart = st.columns([1.5, 1])

        with col_map:
            st.markdown("### 🗺️ Urban Expansion Map")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            colors = list(URBAN_COLORS.values())
            cmap = mcolors.ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#070B14')

            axes[0].set_facecolor('#0A0F1A')
            im0 = axes[0].imshow(np.stack([ndbi_t1, ndbi_t2], axis=0).mean(axis=0), cmap='YlOrRd', vmin=-0.5, vmax=0.5)
            axes[0].set_title('NDBI Map', color='#88AACC', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            axes[1].set_facecolor('#0A0F1A')
            im1 = axes[1].imshow(change_mask, cmap=cmap, norm=norm)
            axes[1].set_title('Urban Change', color='#88AACC', fontsize=10)
            axes[1].axis('off')
            cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(list(URBAN_CLASSES.values()), color='#88AACC', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        with col_chart:
            st.markdown("### 📊 Urban Coverage")
            import plotly.graph_objects as go

            coverage = stats["coverage"]
            fig = go.Figure(
                go.Pie(
                    labels=list(coverage.keys()),
                    values=[v["coverage"] for v in coverage.values()],
                    marker_colors=list(URBAN_COLORS.values()),
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 📋 Expansion Statistics")
        import pandas as pd

        rows = []
        for cls, data in stats["coverage"].items():
            rows.append({"Change Class": cls, "Pixels": f"{data['pixels']:,}", "Coverage %": data["coverage"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem 2rem; background: rgba(0, 212, 255, 0.03); border-radius: 20px; border: 1px solid rgba(0, 212, 255, 0.1);">
        <h2 style="font-family: 'Inter', sans-serif; font-size: 1.5rem; color: #00FFFF; margin-bottom: 1rem;">🎯 MISSION READY</h2>
        <p style="font-family: 'Inter', sans-serif; color: #88AACC;">Select an analysis protocol from Mission Control and execute.</p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 2rem; flex-wrap: wrap;">
            <div class="tech-badge">🌿 NDVI READY</div>
            <div class="tech-badge">💧 WATER DETECTION</div>
            <div class="tech-badge">🔄 CHANGE DETECTION</div>
            <div class="tech-badge">🏙️ LAND USE</div>
            <div class="tech-badge">🔥 BURN AREA</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── ENTERPRISE FOOTER ──────────────────────────────────────────────────────────
st.markdown("""
<div class="enterprise-footer">
    <div class="footer-brand">🛰️ EARTHLENS AI</div>
    <div class="footer-tech">
        <span>⚡ REAL-TIME PROCESSING</span>
        <span>🛰️ MULTI-SPECTRAL IMAGERY</span>
        <span>🧠 DEEP LEARNING MODELS</span>
        <span>📡 10M RESOLUTION</span>
        <span>🌍 GEE INTEGRATION</span>
    </div>
    <div class="footer-developer">Developed by <strong style="color: #00FF88;">GOURGOPAL MOHAPATRA</strong></div>
    <div class="footer-copyright">© 2026 EarthLens AI | Satellite Intelligence Division | All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)