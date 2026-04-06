"""
EARTHLENS AI — Burn Area Detection Module
==========================================
Author  : Gouragopal Mohapatra
Purpose : Detect fire affected / burned areas using NBR and dNBR
Formula :
    NBR  = (NIR - SWIR2) / (NIR + SWIR2)
    dNBR = NBR_prefire - NBR_postfire
"""

import numpy as np
from pathlib import Path
from loguru import logger

from earthlens_core.analysis_engine.preprocessing import preprocess_band, save_processed


# ── Burn Severity Classes (USGS Standard) ─────────────────────────────────────
BURN_CLASSES = {
    0: "Unburned",
    1: "Low Severity",
    2: "Moderate-Low Severity",
    3: "Moderate-High Severity",
    4: "High Severity",
}

BURN_COLORS = {
    0: "#2d6a4f",   # Green      — Unburned
    1: "#ffe169",   # Yellow     — Low
    2: "#f4a261",   # Orange     — Moderate-Low
    3: "#e76f51",   # Dark Orange — Moderate-High
    4: "#9b2226",   # Dark Red   — High Severity
}

# dNBR thresholds (USGS Standard)
DNBR_THRESHOLDS = {
    "Unburned"             : (-np.inf, 0.10),
    "Low Severity"         : (0.10,    0.27),
    "Moderate-Low Severity": (0.27,    0.44),
    "Moderate-High Severity":(0.44,   0.66),
    "High Severity"        : (0.66,    np.inf),
}


# ── NBR Calculation ────────────────────────────────────────────────────────────
def calculate_nbr(nir  : np.ndarray,
                  swir2: np.ndarray) -> np.ndarray:
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Uses B08 (NIR) and B12 (SWIR-2)
    Range: -1.0 to +1.0
    Healthy veg  → high NBR (+)
    Burned area  → low NBR  (-)
    """
    nir   = nir.astype(np.float32)
    swir2 = swir2.astype(np.float32)

    denom = nir + swir2
    denom[denom == 0] = np.nan

    nbr = (nir - swir2) / denom
    nbr = np.clip(nbr, -1.0, 1.0)

    logger.info(
        f"NBR calculated | "
        f"min={np.nanmin(nbr):.3f} "
        f"max={np.nanmax(nbr):.3f} "
        f"mean={np.nanmean(nbr):.3f}"
    )
    return nbr.astype(np.float32)


# ── dNBR Calculation ───────────────────────────────────────────────────────────
def calculate_dnbr(nbr_prefire : np.ndarray,
                   nbr_postfire: np.ndarray) -> np.ndarray:
    """
    dNBR = NBR_prefire - NBR_postfire
    Positive dNBR = burned area
    Negative dNBR = vegetation regrowth
    """
    if nbr_prefire.shape != nbr_postfire.shape:
        raise ValueError(
            f"Shape mismatch: pre={nbr_prefire.shape} post={nbr_postfire.shape}"
        )

    dnbr = nbr_prefire - nbr_postfire
    dnbr = np.clip(dnbr, -2.0, 2.0)

    logger.info(
        f"dNBR calculated | "
        f"min={dnbr.min():.3f} "
        f"max={dnbr.max():.3f} "
        f"mean={dnbr.mean():.3f}"
    )
    return dnbr.astype(np.float32)


# ── Classify Burn Severity ─────────────────────────────────────────────────────
def classify_burn(dnbr: np.ndarray) -> np.ndarray:
    """
    Classify dNBR into burn severity classes.
    Returns integer class map.
    """
    classified = np.zeros_like(dnbr, dtype=np.uint8)

    thresholds = [
        (0, -np.inf, 0.10),   # Unburned
        (1,  0.10,   0.27),   # Low
        (2,  0.27,   0.44),   # Moderate-Low
        (3,  0.44,   0.66),   # Moderate-High
        (4,  0.66,   np.inf), # High
    ]

    for cls_id, low, high in thresholds:
        mask = (dnbr >= low) & (dnbr < high)
        classified[mask] = cls_id

    logger.info("Burn severity classification complete")
    return classified


# ── Burn Stats ─────────────────────────────────────────────────────────────────
def burn_stats(dnbr      : np.ndarray,
               classified: np.ndarray) -> dict:
    """
    Calculate burn area statistics.
    """
    total = classified.size

    stats = {
        "dnbr_min"  : float(dnbr.min()),
        "dnbr_max"  : float(dnbr.max()),
        "dnbr_mean" : float(dnbr.mean()),
    }

    coverage = {}
    for cls_id, cls_name in BURN_CLASSES.items():
        count = int((classified == cls_id).sum())
        pct   = round(count / total * 100, 2)
        coverage[cls_name] = {
            "pixels"  : count,
            "coverage": pct,
        }

    stats["coverage"] = coverage

    # Total burned area (all classes except Unburned)
    burned_px  = int((classified > 0).sum())
    burned_pct = round(burned_px / total * 100, 2)
    stats["total_burned_pixels"]  = burned_px
    stats["total_burned_coverage"]= burned_pct

    logger.info(f"Total burned area: {burned_pct}%")
    return stats


# ── Simulate Post-fire bands ───────────────────────────────────────────────────
def simulate_postfire(bands_prefire: dict,
                      burn_intensity: float = 0.6) -> dict:
    """
    Simulate post-fire bands from pre-fire bands.
    Used for testing when only one date is available.

    burn_intensity: 0.0 to 1.0 (higher = more burned)
    """
    bands_post = {k: v.copy().astype(np.float32) for k, v in bands_prefire.items()}

    h, w = bands_post["B08"].shape

    # Simulate burn in a diagonal patch
    burn_mask = np.zeros((h, w), dtype=bool)
    burn_mask[h//4:3*h//4, w//4:3*w//4] = True

    # Post-fire: NIR drops, SWIR2 increases
    bands_post["B08"][burn_mask] *= (1 - burn_intensity)       # NIR down
    bands_post["B12"][burn_mask] *= (1 + burn_intensity * 1.5) # SWIR2 up
    bands_post["B04"][burn_mask] *= (1 + burn_intensity * 0.5) # Red slightly up

    logger.info(f"Simulated post-fire bands | intensity={burn_intensity}")
    return bands_post


# ── Full Burn Detection Pipeline ───────────────────────────────────────────────
def run_burn_pipeline(bands_prefire  : dict,
                      bands_postfire : dict  = None,
                      burn_intensity : float = 0.6,
                      output_dir     : Path  = None) -> dict:
    """
    Complete burn area detection pipeline:
    NBR pre → NBR post → dNBR → Classify → Stats → Save

    Returns: {
        "nbr_pre"    : pre-fire NBR,
        "nbr_post"   : post-fire NBR,
        "dnbr"       : difference NBR,
        "classified" : burn severity map,
        "stats"      : statistics,
        "saved_to"   : output path
    }
    """
    logger.info("Starting Burn Area Detection pipeline...")

    # Simulate post-fire if not provided
    if bands_postfire is None:
        logger.warning("Post-fire bands not provided — simulating burn...")
        bands_postfire = simulate_postfire(bands_prefire, burn_intensity)

    # Calculate NBR
    nbr_pre  = calculate_nbr(
        bands_prefire["B08"],
        bands_prefire["B12"],
    )
    nbr_post = calculate_nbr(
        bands_postfire["B08"],
        bands_postfire["B12"],
    )

    # dNBR
    dnbr = calculate_dnbr(nbr_pre, nbr_post)

    # Classify
    classified = classify_burn(dnbr)

    # Stats
    stats = burn_stats(dnbr, classified)

    result = {
        "nbr_pre"    : nbr_pre,
        "nbr_post"   : nbr_post,
        "dnbr"       : dnbr,
        "classified" : classified,
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

        h, w      = classified.shape
        transform = from_bounds(77.05, 28.40, 77.35, 28.75, w, h)
        meta = dict(
            driver="GTiff", height=h, width=w,
            count=1, dtype=np.uint8,
            crs=CRS.from_epsg(4326),
            transform=transform,
        )

        # Save classified
        burn_path = output_dir / "burn_severity.tif"
        with rasterio.open(burn_path, "w", **meta) as dst:
            dst.write(classified, 1)

        # Save dNBR
        dnbr_path = output_dir / "dnbr.tif"
        meta.update({"dtype": "float32"})
        with rasterio.open(dnbr_path, "w", **meta) as dst:
            dst.write(dnbr, 1)

        result["saved_to"] = str(burn_path)
        logger.success(f"Burn area saved: {burn_path}")

    return result