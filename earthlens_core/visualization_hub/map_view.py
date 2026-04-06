"""
EARTHLENS AI — Map View Module
================================
Author : Gouragopal Mohapatra
Purpose: Generate interactive maps from analysis results
Library: Folium + Streamlit-Folium
"""

import numpy as np
import io
import base64
from PIL import Image
import folium
import folium.plugins
from pathlib import Path
from loguru import logger


# ── Colormaps ──────────────────────────────────────────────────────────────────
NDVI_COLORMAP = {
    "Water / Snow"       : "#2166ac",  # Blue
    "Bare Soil / Urban"  : "#d4a56a",  # Brown
    "Sparse Vegetation"  : "#d9ef8b",  # Light Green
    "Moderate Vegetation": "#66bd63",  # Green
    "Dense Vegetation"   : "#1a7837",  # Dark Green
}

WATER_COLORMAP = {
    "No Water" : "#f7f7f7",  # White
    "Water"    : "#2166ac",  # Blue
}


# ── Base Map ───────────────────────────────────────────────────────────────────
def create_base_map(lat: float = 28.57,
                    lon: float = 77.20,
                    zoom: int  = 10) -> folium.Map:
    m = folium.Map(
        location     = [lat, lon],
        zoom_start   = zoom,
        tiles        = "CartoDB dark_matter",
        prefer_canvas= True,
    )

    folium.TileLayer("OpenStreetMap",    name="Street Map").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)

    # Fullscreen button
    folium.plugins.Fullscreen().add_to(m)

    logger.info(f"Base map created | center=({lat}, {lon}) zoom={zoom}")
    return m

# ── NDVI Map ───────────────────────────────────────────────────────────────────
def ndvi_map(ndvi_array  : np.ndarray,
             meta        : dict,
             lat         : float = 28.57,
             lon         : float = 77.20) -> folium.Map:

    m = create_base_map(lat, lon)

    transform = meta.get("transform")
    if transform:
        west  = transform.c
        north = transform.f
        east  = west  + transform.a * ndvi_array.shape[1]
        south = north + transform.e * ndvi_array.shape[0]
        bounds = [[south, west], [north, east]]
    else:
        bounds = [[28.40, 77.05], [28.75, 77.35]]

    # RGBA array → PIL Image → base64
    colored = colorize_ndvi(ndvi_array)
    img     = Image.fromarray(colored, mode="RGBA")
    buf     = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_url = f"data:image/png;base64,{encoded}"

    folium.raster_layers.ImageOverlay(
        image   = img_url,
        bounds  = bounds,
        opacity = 0.7,
        name    = "NDVI Layer",
        show    = True,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    _add_ndvi_legend(m)

    logger.success("NDVI map generated")
    return m


# ── Water Map ──────────────────────────────────────────────────────────────────
def water_map(water_mask : np.ndarray,
              meta       : dict,
              lat        : float = 28.57,
              lon        : float = 77.20) -> folium.Map:

    m = create_base_map(lat, lon)

    transform = meta.get("transform")
    if transform:
        west  = transform.c
        north = transform.f
        east  = west  + transform.a * water_mask.shape[1]
        south = north + transform.e * water_mask.shape[0]
        bounds = [[south, west], [north, east]]
    else:
        bounds = [[28.40, 77.05], [28.75, 77.35]]

    # RGBA array → PIL Image → base64
    colored = colorize_water(water_mask)
    img     = Image.fromarray(colored, mode="RGBA")
    buf     = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_url = f"data:image/png;base64,{encoded}"

    folium.raster_layers.ImageOverlay(
        image   = img_url,
        bounds  = bounds,
        opacity = 0.7,
        name    = "Water Detection Layer",
        show    = True,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    _add_water_legend(m)

    logger.success("Water map generated")
    return m


# ── Colorize NDVI array → RGBA ─────────────────────────────────────────────────
def colorize_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Convert NDVI float array to RGBA image array.
    """
    h, w   = ndvi.shape
    rgba   = np.zeros((h, w, 4), dtype=np.uint8)

    # Color mapping
    colors = [
        ((-1.0, 0.0),  (33,  102, 172, 180)),   # Blue   — Water
        ((0.0,  0.2),  (212, 165, 106, 180)),   # Brown  — Bare/Urban
        ((0.2,  0.4),  (217, 239, 139, 180)),   # Lt Green — Sparse
        ((0.4,  0.6),  (102, 189,  99, 180)),   # Green  — Moderate
        ((0.6,  1.0),  (26,  120,  55, 180)),   # Dk Green — Dense
    ]

    for (low, high), color in colors:
        mask = (ndvi >= low) & (ndvi < high)
        rgba[mask] = color

    return rgba


# ── Colorize Water Mask → RGBA ─────────────────────────────────────────────────
def colorize_water(water_mask: np.ndarray) -> np.ndarray:
    """
    Convert water mask (0/1) to RGBA image array.
    """
    h, w = water_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Water = Blue, No water = transparent
    rgba[water_mask == 1] = (33, 102, 172, 200)
    rgba[water_mask == 0] = (247, 247, 247, 50)

    return rgba


# ── NDVI Legend ────────────────────────────────────────────────────────────────
def _add_ndvi_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(0,0,0,0.8); padding: 12px; border-radius: 8px;
                color: white; font-family: monospace; font-size: 12px;">
        <b>🌿 NDVI Legend</b><br>
        <span style="color:#2166ac">█</span> Water / Snow &nbsp;&nbsp; (-1.0 → 0.0)<br>
        <span style="color:#d4a56a">█</span> Bare / Urban &nbsp;&nbsp; (0.0 → 0.2)<br>
        <span style="color:#d9ef8b">█</span> Sparse Veg &nbsp;&nbsp;&nbsp; (0.2 → 0.4)<br>
        <span style="color:#66bd63">█</span> Moderate Veg &nbsp; (0.4 → 0.6)<br>
        <span style="color:#1a7837">█</span> Dense Veg &nbsp;&nbsp;&nbsp;&nbsp; (0.6 → 1.0)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ── Water Legend ───────────────────────────────────────────────────────────────
def _add_water_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(0,0,0,0.8); padding: 12px; border-radius: 8px;
                color: white; font-family: monospace; font-size: 12px;">
        <b>💧 Water Detection</b><br>
        <span style="color:#2166ac">█</span> Water Body<br>
        <span style="color:#f7f7f7">█</span> No Water<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ── Save Map ───────────────────────────────────────────────────────────────────
def save_map(m: folium.Map,
             output_path: str | Path) -> None:
    """
    Save folium map as HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.success(f"Map saved: {output_path}")