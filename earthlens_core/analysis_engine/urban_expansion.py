"""
EARTHLENS AI — Urban Expansion Module
=======================================
Author  : Gouragopal Mohapatra
Purpose : Detect and quantify urban growth over time
Indices : NDBI, UI, NBI, IBI
"""

import numpy as np
from pathlib import Path
from loguru import logger

from earthlens_core.analysis_engine.preprocessing import preprocess_band, save_processed


# ── Urban Classes ──────────────────────────────────────────────────────────────
URBAN_CLASSES = {
    0: "No Change",
    1: "New Urban Area",
    2: "Urban Densification",
    3: "Vegetation to Urban",
    4: "Bare Soil to Urban",
}

URBAN_COLORS = {
    0: "#2d2d2d",   # Dark Grey  — No Change
    1: "#e53935",   # Red        — New Urban
    2: "#ff6f00",   # Orange     — Densification
    3: "#fdd835",   # Yellow     — Veg to Urban
    4: "#8e24aa",   # Purple     — Bare to Urban
}


# ── NDBI — Normalized Difference Built-up Index ────────────────────────────────
def calculate_ndbi(swir1: np.ndarray,
                   nir  : np.ndarray) -> np.ndarray:
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Uses B11 (SWIR1) and B08 (NIR)
    High NDBI → Built-up / Urban areas
    Range: -1.0 to +1.0
    """
    swir1 = swir1.astype(np.float32)
    nir   = nir.astype(np.float32)

    denom = swir1 + nir
    denom[denom == 0] = np.nan

    ndbi = (swir1 - nir) / denom
    ndbi = np.clip(ndbi, -1.0, 1.0)

    logger.info(
        f"NDBI calculated | "
        f"min={np.nanmin(ndbi):.3f} "
        f"max={np.nanmax(ndbi):.3f} "
        f"mean={np.nanmean(ndbi):.3f}"
    )
    return ndbi.astype(np.float32)


# ── UI — Urban Index ───────────────────────────────────────────────────────────
def calculate_ui(swir2: np.ndarray,
                 nir  : np.ndarray) -> np.ndarray:
    """
    UI = (SWIR2 - NIR) / (SWIR2 + NIR)
    Uses B12 (SWIR2) and B08 (NIR)
    Better for dense urban detection.
    """
    swir2 = swir2.astype(np.float32)
    nir   = nir.astype(np.float32)

    denom = swir2 + nir
    denom[denom == 0] = np.nan

    ui = (swir2 - nir) / denom
    ui = np.clip(ui, -1.0, 1.0)

    logger.info(
        f"UI calculated | "
        f"min={np.nanmin(ui):.3f} "
        f"max={np.nanmax(ui):.3f} "
        f"mean={np.nanmean(ui):.3f}"
    )
    return ui.astype(np.float32)


# ── IBI — Index-based Built-up Index ──────────────────────────────────────────
def calculate_ibi(swir1: np.ndarray,
                  nir  : np.ndarray,
                  green: np.ndarray) -> np.ndarray:
    """
    IBI = (NDBI - (NDVI + MNDWI) / 2) / (NDBI + (NDVI + MNDWI) / 2)
    Most accurate built-up index — combines 3 indices.
    """
    swir1 = swir1.astype(np.float32)
    nir   = nir.astype(np.float32)
    green = green.astype(np.float32)

    def safe_div(a, b):
        b = b.copy()
        b[b == 0] = np.nan
        return np.where(np.isnan(b), 0, a / b)

    ndbi  = safe_div(swir1 - nir,   swir1 + nir)
    ndvi  = safe_div(nir   - swir1, nir   + swir1)
    mndwi = safe_div(green - swir1, green + swir1)

    numerator   = ndbi - (ndvi + mndwi) / 2
    denominator = ndbi + (ndvi + mndwi) / 2

    ibi = safe_div(numerator, denominator)
    ibi = np.clip(ibi, -1.0, 1.0)

    logger.info(
        f"IBI calculated | "
        f"min={np.nanmin(ibi):.3f} "
        f"max={np.nanmax(ibi):.3f} "
        f"mean={np.nanmean(ibi):.3f}"
    )
    return ibi.astype(np.float32)


# ── Urban Mask ─────────────────────────────────────────────────────────────────
def create_urban_mask(ndbi     : np.ndarray,
                      threshold: float = 0.1) -> np.ndarray:
    """
    Create binary urban mask from NDBI.
    1 = Urban, 0 = Non-Urban
    """
    urban_mask = (ndbi > threshold).astype(np.uint8)
    urban_pct  = urban_mask.mean() * 100

    logger.info(f"Urban coverage: {urban_pct:.2f}%")
    return urban_mask


# ── Urban Change Detection ─────────────────────────────────────────────────────
def detect_urban_change(bands_t1  : dict,
                        bands_t2  : dict,
                        threshold : float = 0.1) -> tuple:
    """
    Detect urban expansion between two time periods.

    Returns: (urban_t1, urban_t2, change_mask)
    """
    # NDBI for both periods
    ndbi_t1 = calculate_ndbi(bands_t1["B11"], bands_t1["B08"])
    ndbi_t2 = calculate_ndbi(bands_t2["B11"], bands_t2["B08"])

    # NDVI for both periods
    def safe_ndvi(bands):
        b04 = bands["B04"].astype(np.float32)
        b08 = bands["B08"].astype(np.float32)
        denom = b08 + b04
        denom[denom == 0] = np.nan
        return np.where(np.isnan(denom), 0, (b08 - b04) / denom)

    ndvi_t1 = safe_ndvi(bands_t1)
    ndvi_t2 = safe_ndvi(bands_t2)

    # Urban masks
    urban_t1 = create_urban_mask(ndbi_t1, threshold)
    urban_t2 = create_urban_mask(ndbi_t2, threshold)

    # Change classification
    change_mask = np.zeros_like(urban_t1, dtype=np.uint8)

    # New urban — was non-urban, now urban
    new_urban = (urban_t1 == 0) & (urban_t2 == 1)
    change_mask[new_urban] = 1

    # Vegetation to Urban
    veg_to_urban = new_urban & (ndvi_t1 > 0.3)
    change_mask[veg_to_urban] = 3

    # Bare soil to Urban
    bare_to_urban = new_urban & (ndvi_t1 <= 0.3) & (ndvi_t1 >= 0.0)
    change_mask[bare_to_urban] = 4

    # Urban densification — NDBI increased in already urban area
    densification = (urban_t1 == 1) & (urban_t2 == 1) & ((ndbi_t2 - ndbi_t1) > 0.05)
    change_mask[densification] = 2

    logger.success("Urban change detection complete!")
    return urban_t1, urban_t2, change_mask


# ── Urban Stats ────────────────────────────────────────────────────────────────
def urban_stats(urban_t1   : np.ndarray,
                urban_t2   : np.ndarray,
                change_mask: np.ndarray) -> dict:
    """
    Calculate urban expansion statistics.
    """
    total = urban_t1.size

    stats = {
        "urban_t1_pixels"  : int(urban_t1.sum()),
        "urban_t1_coverage": round(float(urban_t1.mean() * 100), 2),
        "urban_t2_pixels"  : int(urban_t2.sum()),
        "urban_t2_coverage": round(float(urban_t2.mean() * 100), 2),
        "expansion_pixels" : int((change_mask > 0).sum()),
        "expansion_pct"    : round(float((change_mask > 0).mean() * 100), 2),
    }

    # Per class coverage
    coverage = {}
    for cls_id, cls_name in URBAN_CLASSES.items():
        count = int((change_mask == cls_id).sum())
        pct   = round(count / total * 100, 2)
        coverage[cls_name] = {
            "pixels"  : count,
            "coverage": pct,
        }

    stats["coverage"] = coverage

    logger.info(
        f"Urban T1={stats['urban_t1_coverage']}% → "
        f"T2={stats['urban_t2_coverage']}% | "
        f"Expansion={stats['expansion_pct']}%"
    )
    return stats


# ── Simulate Urban Growth ──────────────────────────────────────────────────────
def simulate_urban_growth(bands_t1    : dict,
                          growth_rate : float = 0.3) -> dict:
    """
    Simulate urban growth for T2.
    Increases SWIR1/SWIR2, decreases NIR in expansion zones.
    """
    bands_t2 = {k: v.copy().astype(np.float32) for k, v in bands_t1.items()}

    h, w = bands_t2["B08"].shape

    # Expand urban outward from center
    y, x   = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    dist   = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Urban expansion ring (between 40-80px from center)
    expansion_zone = (dist > 40) & (dist < 80)

    bands_t2["B11"][expansion_zone] *= (1 + growth_rate)
    bands_t2["B12"][expansion_zone] *= (1 + growth_rate * 0.8)
    bands_t2["B08"][expansion_zone] *= (1 - growth_rate * 0.6)
    bands_t2["B04"][expansion_zone] *= (1 + growth_rate * 0.3)

    logger.info(f"Simulated urban growth | rate={growth_rate}")
    return bands_t2


# ── Full Urban Expansion Pipeline ──────────────────────────────────────────────
def run_urban_pipeline(bands_t1    : dict,
                       bands_t2    : dict  = None,
                       growth_rate : float = 0.3,
                       threshold   : float = 0.1,
                       output_dir  : Path  = None) -> dict:
    """
    Complete urban expansion pipeline:
    NDBI T1 → NDBI T2 → Change Detection → Stats → Save

    Returns: {
        "urban_t1"   : urban mask T1,
        "urban_t2"   : urban mask T2,
        "ndbi_t1"    : NDBI T1,
        "ndbi_t2"    : NDBI T2,
        "ibi"        : IBI map,
        "change_mask": classified change,
        "stats"      : statistics,
        "saved_to"   : output path
    }
    """
    logger.info("Starting Urban Expansion pipeline...")

    # Simulate T2 if not provided
    if bands_t2 is None:
        logger.warning("T2 bands not provided — simulating urban growth...")
        bands_t2 = simulate_urban_growth(bands_t1, growth_rate)

    # NDBI
    ndbi_t1 = calculate_ndbi(bands_t1["B11"], bands_t1["B08"])
    ndbi_t2 = calculate_ndbi(bands_t2["B11"], bands_t2["B08"])

    # IBI
    ibi = calculate_ibi(
        bands_t1["B11"],
        bands_t1["B08"],
        bands_t1["B03"],
    )

    # Urban change
    urban_t1, urban_t2, change_mask = detect_urban_change(
        bands_t1, bands_t2, threshold
    )

    # Stats
    stats = urban_stats(urban_t1, urban_t2, change_mask)

    result = {
        "urban_t1"   : urban_t1,
        "urban_t2"   : urban_t2,
        "ndbi_t1"    : ndbi_t1,
        "ndbi_t2"    : ndbi_t2,
        "ibi"        : ibi,
        "change_mask": change_mask,
        "stats"      : stats,
        "saved_to"   : None,
    }

    # Save
    if output_dir:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        h, w      = change_mask.shape
        transform = from_bounds(77.05, 28.40, 77.35, 28.75, w, h)

        meta = dict(
            driver="GTiff", height=h, width=w,
            count=1, dtype=np.uint8,
            crs=CRS.from_epsg(4326),
            transform=transform,
        )

        # Save change mask
        change_path = output_dir / "urban_expansion.tif"
        with rasterio.open(change_path, "w", **meta) as dst:
            dst.write(change_mask, 1)

        # Save NDBI T2
        meta.update({"dtype": "float32"})
        ndbi_path = output_dir / "ndbi_t2.tif"
        with rasterio.open(ndbi_path, "w", **meta) as dst:
            dst.write(ndbi_t2, 1)

        result["saved_to"] = str(change_path)
        logger.success(f"Urban expansion saved: {change_path}")

    return result