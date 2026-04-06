"""
EARTHLENS AI — Preprocessing Module
=====================================
Author : Gouragopal Mohapatra
Purpose: Raw satellite band loading, cleaning, normalization
"""

import numpy as np
import rasterio
from pathlib import Path
from loguru import logger
from earthlens_config.settings import SETTINGS


# ── Load single band ───────────────────────────────────────────────────────────
def load_band(filepath: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a single GeoTIFF band.
    Returns: (array, metadata)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Band not found: {filepath}")

    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        meta["transform"] = src.transform
        meta["crs"]       = src.crs

    logger.info(f"Loaded band: {filepath.name} | shape: {data.shape}")
    return data, meta


# ── Load multiple bands ────────────────────────────────────────────────────────
def load_bands(band_paths: dict) -> tuple[dict, dict]:
    """
    Load multiple bands at once.
    Input : {"B04": "path/to/B04.tif", "B08": "path/to/B08.tif"}
    Returns: ({"B04": array, "B08": array}, metadata)
    """
    arrays = {}
    meta   = {}

    for band_name, path in band_paths.items():
        arrays[band_name], meta = load_band(path)

    logger.info(f"Loaded {len(arrays)} bands: {list(arrays.keys())}")
    return arrays, meta


# ── Normalize band ─────────────────────────────────────────────────────────────
def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to 0.0 - 1.0 range.
    """
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)

    if arr_max - arr_min == 0:
        logger.warning("Band has no variance — returning zeros")
        return np.zeros_like(array)

    normalized = (array - arr_min) / (arr_max - arr_min)
    logger.info(f"Normalized | min={arr_min:.2f} max={arr_max:.2f}")
    return normalized.astype(np.float32)


# ── Mask invalid pixels ────────────────────────────────────────────────────────
def mask_invalid(array: np.ndarray,
                 nodata: float = 0.0) -> np.ndarray:
    """
    Replace nodata / negative values with NaN.
    """
    array = array.astype(np.float32)
    array[array <= nodata] = np.nan
    return array


# ── Clip outliers ──────────────────────────────────────────────────────────────
def clip_percentile(array: np.ndarray,
                    low: float = 2.0,
                    high: float = 98.0) -> np.ndarray:
    """
    Clip array to percentile range — removes extreme outliers.
    """
    p_low  = np.nanpercentile(array, low)
    p_high = np.nanpercentile(array, high)
    clipped = np.clip(array, p_low, p_high)
    logger.info(f"Clipped to [{p_low:.2f}, {p_high:.2f}]")
    return clipped.astype(np.float32)


# ── Full preprocessing pipeline ────────────────────────────────────────────────
def preprocess_band(filepath: str | Path) -> tuple[np.ndarray, dict]:
    """
    Full pipeline:
    load → mask invalid → clip outliers → normalize
    """
    raw, meta = load_band(filepath)
    cleaned   = mask_invalid(raw)
    clipped   = clip_percentile(cleaned)
    normed    = normalize(clipped)

    logger.success(f"Preprocessed: {Path(filepath).name}")
    return normed, meta


# ── Save processed band ────────────────────────────────────────────────────────
def save_processed(array: np.ndarray,
                   meta: dict,
                   output_path: str | Path) -> None:
    """
    Save processed band as GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta.update({
        "dtype"  : "float32",
        "count"  : 1,
        "driver" : "GTiff",
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array, 1)

    logger.success(f"Saved processed band: {output_path.name}")