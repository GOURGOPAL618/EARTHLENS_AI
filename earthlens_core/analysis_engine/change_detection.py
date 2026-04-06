"""
EARTHLENS AI — Change Detection Module
========================================
Author  : Gouragopal Mohapatra
Purpose : Detect land cover changes between two time periods
Method  : Image Differencing + NDVI Change + CVA
"""

import numpy as np
from pathlib import Path
from loguru import logger

from earthlens_core.analysis_engine.preprocessing import preprocess_band, save_processed


# ── Change Classes ─────────────────────────────────────────────────────────────
CHANGE_CLASSES = {
    0: "No Change",
    1: "Vegetation Loss",
    2: "Vegetation Gain",
    3: "Water Increase",
    4: "Water Decrease",
    5: "Urban Expansion",
}

CHANGE_COLORS = {
    0: "#2d2d2d",   # Dark Grey  — No Change
    1: "#e53935",   # Red        — Veg Loss
    2: "#43a047",   # Green      — Veg Gain
    3: "#1e88e5",   # Blue       — Water Increase
    4: "#fb8c00",   # Orange     — Water Decrease
    5: "#8e24aa",   # Purple     — Urban Expansion
}


# ── NDVI Change Detection ──────────────────────────────────────────────────────
def ndvi_change(ndvi_t1: np.ndarray,
                ndvi_t2: np.ndarray,
                threshold: float = 0.15) -> tuple:
    """
    Detect NDVI change between two time periods.

    ndvi_t1 : NDVI at time 1 (earlier)
    ndvi_t2 : NDVI at time 2 (later)
    threshold: minimum change to consider significant

    Returns: (change_array, change_mask)
        change_array : float difference (-2 to +2)
        change_mask  : classified change map
    """
    if ndvi_t1.shape != ndvi_t2.shape:
        raise ValueError(
            f"Shape mismatch: T1={ndvi_t1.shape} vs T2={ndvi_t2.shape}"
        )

    # Raw difference
    delta = ndvi_t2 - ndvi_t1

    # Classify changes
    change_mask = np.zeros_like(delta, dtype=np.uint8)

    # Vegetation loss — NDVI decreased significantly
    change_mask[delta < -threshold] = 1

    # Vegetation gain — NDVI increased significantly
    change_mask[delta >  threshold] = 2

    # No change
    change_mask[np.abs(delta) <= threshold] = 0

    # Stats
    total = delta.size
    for cls_id, cls_name in CHANGE_CLASSES.items():
        count = int((change_mask == cls_id).sum())
        pct   = count / total * 100
        if count > 0:
            logger.info(f"  {cls_name}: {count:,} px ({pct:.1f}%)")

    logger.success("NDVI change detection complete!")
    return delta, change_mask


# ── Band Differencing ──────────────────────────────────────────────────────────
def band_difference(band_t1  : np.ndarray,
                    band_t2  : np.ndarray,
                    threshold: float = 0.1) -> tuple:
    """
    Simple band differencing between two images.
    Returns: (difference, change_mask)
    """
    delta = band_t2 - band_t1
    delta_norm = delta / (np.abs(delta).max() + 1e-8)

    change_mask = np.zeros_like(delta, dtype=np.uint8)
    change_mask[delta_norm >  threshold] = 1   # Increased
    change_mask[delta_norm < -threshold] = 2   # Decreased

    logger.info(f"Band difference | min={delta.min():.3f} max={delta.max():.3f}")
    return delta, change_mask


# ── Change Vector Analysis (CVA) ───────────────────────────────────────────────
def change_vector_analysis(bands_t1: dict,
                           bands_t2: dict,
                           threshold: float = 0.1) -> tuple:
    """
    CVA — Multi-band change detection.
    Calculates magnitude and direction of change across all bands.

    Returns: (magnitude, angle, change_mask)
    """
    # Use NIR (B08) and Red (B04) — most sensitive to vegetation change
    nir_diff = bands_t2["B08"].astype(np.float32) - bands_t1["B08"].astype(np.float32)
    red_diff = bands_t2["B04"].astype(np.float32) - bands_t1["B04"].astype(np.float32)

    # Normalize
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    nir_diff = norm(nir_diff)
    red_diff = norm(red_diff)

    # Magnitude of change
    magnitude = np.sqrt(nir_diff**2 + red_diff**2)

    # Direction of change (angle in degrees)
    angle = np.degrees(np.arctan2(nir_diff, red_diff))

    # Classify by magnitude + angle
    change_mask = np.zeros_like(magnitude, dtype=np.uint8)

    sig = magnitude > threshold  # significant change pixels

    # Vegetation gain — NIR increases, Red decreases (angle ~90-180)
    change_mask[sig & (angle > 45)  & (angle <= 180)] = 2

    # Vegetation loss — NIR decreases (angle ~-90 to -180)
    change_mask[sig & (angle < -45) & (angle >= -180)] = 1

    # Urban expansion — Red increases, NIR stable (angle ~0-45)
    change_mask[sig & (angle >= -45) & (angle <= 45)] = 5

    logger.info(
        f"CVA | magnitude max={magnitude.max():.3f} "
        f"mean={magnitude.mean():.3f}"
    )
    logger.success("Change Vector Analysis complete!")
    return magnitude, angle, change_mask


# ── Change Stats ───────────────────────────────────────────────────────────────
def change_stats(change_mask: np.ndarray) -> dict:
    """
    Calculate per-class change statistics.
    """
    total = change_mask.size
    stats = {}

    for cls_id, cls_name in CHANGE_CLASSES.items():
        count = int((change_mask == cls_id).sum())
        pct   = round(count / total * 100, 2)
        stats[cls_name] = {
            "pixels"  : count,
            "coverage": pct,
        }

    logger.info(f"Change stats: {stats}")
    return stats


# ── Simulate T2 bands ──────────────────────────────────────────────────────────
def simulate_change(bands_t1: dict,
                    change_type: str = "deforestation") -> dict:
    """
    Simulate a second time period by applying changes to T1.
    Used for testing when only one date is available.

    change_type options:
        'deforestation' — reduce NIR in vegetation areas
        'urbanization'  — increase Red/SWIR in some areas
        'flooding'      — increase water signature
    """
    bands_t2 = {k: v.copy().astype(np.float32) for k, v in bands_t1.items()}

    h, w = bands_t2["B04"].shape

    if change_type == "deforestation":
        # Reduce NIR in top-left quadrant (simulate forest clearing)
        bands_t2["B08"][:h//2, :w//2] *= 0.4
        bands_t2["B04"][:h//2, :w//2] *= 1.3
        logger.info("Simulated: Deforestation in top-left quadrant")

    elif change_type == "urbanization":
        # Increase SWIR in bottom-right quadrant
        bands_t2["B11"][h//2:, w//2:] *= 1.6
        bands_t2["B08"][h//2:, w//2:] *= 0.7
        logger.info("Simulated: Urbanization in bottom-right quadrant")

    elif change_type == "flooding":
        # Decrease NIR + increase Green in center strip
        bands_t2["B08"][h//3:2*h//3, :] *= 0.3
        bands_t2["B03"][h//3:2*h//3, :] *= 1.5
        logger.info("Simulated: Flooding in center strip")

    return bands_t2


# ── Full Change Detection Pipeline ─────────────────────────────────────────────
def run_change_pipeline(bands_t1   : dict,
                        bands_t2   : dict = None,
                        change_type: str  = "deforestation",
                        method     : str  = "ndvi",
                        threshold  : float = 0.15,
                        output_dir : Path  = None) -> dict:
    """
    Complete change detection pipeline.

    bands_t1    : bands at time 1
    bands_t2    : bands at time 2 (if None — simulated)
    change_type : simulation type if T2 not available
    method      : 'ndvi' or 'cva'
    threshold   : change sensitivity

    Returns: {
        "delta"      : difference array,
        "change_mask": classified map,
        "magnitude"  : CVA magnitude (if method=cva),
        "stats"      : statistics,
        "saved_to"   : output path
    }
    """
    logger.info(f"Starting Change Detection | method={method}")

    # Compute NDVI for T1
    def safe_ndvi(bands):
        b04 = bands["B04"].astype(np.float32)
        b08 = bands["B08"].astype(np.float32)
        denom = b08 + b04
        denom[denom == 0] = np.nan
        return np.where(np.isnan(denom), 0, (b08 - b04) / denom)

    ndvi_t1 = safe_ndvi(bands_t1)

    # Simulate T2 if not provided
    if bands_t2 is None:
        logger.warning("T2 bands not provided — simulating changes...")
        bands_t2 = simulate_change(bands_t1, change_type)

    ndvi_t2 = safe_ndvi(bands_t2)

    result = {
        "ndvi_t1"    : ndvi_t1,
        "ndvi_t2"    : ndvi_t2,
        "delta"      : None,
        "change_mask": None,
        "magnitude"  : None,
        "stats"      : None,
        "saved_to"   : None,
        "change_type": change_type,
        "method"     : method,
    }

    if method == "ndvi":
        delta, change_mask = ndvi_change(ndvi_t1, ndvi_t2, threshold)
        result["delta"]       = delta
        result["change_mask"] = change_mask

    elif method == "cva":
        magnitude, angle, change_mask = change_vector_analysis(
            bands_t1, bands_t2, threshold
        )
        result["magnitude"]   = magnitude
        result["delta"]       = magnitude
        result["change_mask"] = change_mask

    # Stats
    result["stats"] = change_stats(result["change_mask"])

    # Save
    if output_dir and result["change_mask"] is not None:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"change_{change_type}_{method}.tif"

        h, w = result["change_mask"].shape
        transform = from_bounds(77.05, 28.40, 77.35, 28.75, w, h)

        with rasterio.open(
            output_path, "w",
            driver="GTiff", height=h, width=w,
            count=1, dtype=np.uint8,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(result["change_mask"], 1)

        result["saved_to"] = str(output_path)
        logger.success(f"Change map saved: {output_path}")

    return result