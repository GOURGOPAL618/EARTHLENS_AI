"""
EARTHLENS AI - Synthetic Sentinel-2 Band Generator
Generates 6 realistic band .tif files for Delhi region
No download needed - uses rasterio + numpy (already installed)
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

# ── Output folder ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "earthlens_data" / "raw_imagery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Delhi bounding box (lon/lat) ───────────────────────────────────────────────
WEST, SOUTH, EAST, NORTH = 77.05, 28.40, 77.35, 28.75

# ── Image size ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 256, 256

# ── CRS ────────────────────────────────────────────────────────────────────────
CRS_WGS84 = CRS.from_epsg(4326)
TRANSFORM = from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)

np.random.seed(42)


def smooth(arr, sigma=12):
    """Simple box-blur for realistic spatial patterns."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr.astype(np.float32), sigma=sigma)


def make_base_layers():
    """Generate realistic terrain masks for Delhi."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = WIDTH // 2, HEIGHT // 2

    # Urban core (central blob)
    urban = np.exp(-((x - cx)**2 + (y - cy)**2) / (60**2))

    # Vegetation patches (Yamuna floodplain + parks)
    veg1 = np.exp(-((x - 80)**2 + (y - 100)**2) / (30**2))
    veg2 = np.exp(-((x - 180)**2 + (y - 160)**2) / (25**2))
    vegetation = np.clip(veg1 + veg2, 0, 1)

    # Water body (Yamuna river — diagonal strip)
    river_dist = np.abs((x - y * 0.6) - 30)
    water = np.clip(1 - river_dist / 12, 0, 1)

    return urban, vegetation, water


def generate_band(band_name, urban, vegetation, water):
    """
    Simulate reflectance values per band based on land cover.
    Values are uint16 scaled reflectance (0–10000 range, like Sentinel-2 L2A).
    """
    noise = smooth(np.random.rand(HEIGHT, WIDTH), sigma=8)

    if band_name == "B02":   # Blue — high urban, low veg/water
        data = (urban * 1800 + vegetation * 500 + water * 800
                + noise * 400 + 600)

    elif band_name == "B03": # Green — moderate everywhere
        data = (urban * 1600 + vegetation * 1000 + water * 900
                + noise * 400 + 600)

    elif band_name == "B04": # Red — high urban, LOW vegetation (key for NDVI)
        data = (urban * 1700 + vegetation * 400 + water * 600
                + noise * 350 + 500)

    elif band_name == "B08": # NIR — HIGH vegetation, low urban/water (key for NDVI)
        data = (urban * 1200 + vegetation * 5500 + water * 400
                + noise * 500 + 800)

    elif band_name == "B11": # SWIR-1 — high urban/bare soil
        data = (urban * 2200 + vegetation * 700 + water * 300
                + noise * 500 + 400)

    elif band_name == "B12": # SWIR-2 — high urban/bare soil
        data = (urban * 2000 + vegetation * 600 + water * 200
                + noise * 450 + 300)

    return np.clip(data, 0, 10000).astype(np.uint16)


def save_band(band_name, data):
    filepath = OUTPUT_DIR / f"sentinel2_delhi_{band_name}.tif"
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype=np.uint16,
        crs=CRS_WGS84,
        transform=TRANSFORM,
    ) as dst:
        dst.write(data, 1)
    size_kb = filepath.stat().st_size // 1024
    return filepath, size_kb


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  EARTHLENS AI - SYNTHETIC BAND GENERATOR")
print("=" * 60)
print(f"  Location  : Delhi (28.40°N–28.75°N, 77.05°E–77.35°E)")
print(f"  Grid size : {WIDTH} x {HEIGHT} pixels")
print(f"  Output    : {OUTPUT_DIR}")
print()

urban, vegetation, water = make_base_layers()

BANDS = {
    "B02": "Blue    (490 nm)",
    "B03": "Green   (560 nm)",
    "B04": "Red     (665 nm)  ← NDVI uses this",
    "B08": "NIR     (842 nm)  ← NDVI uses this",
    "B11": "SWIR-1  (1610 nm)",
    "B12": "SWIR-2  (2190 nm)",
}

generated = []
for band_name, desc in BANDS.items():
    data = generate_band(band_name, urban, vegetation, water)
    fp, kb = save_band(band_name, data)
    generated.append((fp.name, kb))
    print(f"  ✅  {fp.name:<35}  {kb:>4} KB   {desc}")

print()
print("=" * 60)
print("  ALL 6 BANDS READY!")
print("=" * 60)
print(f"  Total files : {len(generated)}")
print()
print("  Next step   : ndvi.py banate hain! 🌿")
print("=" * 60)
