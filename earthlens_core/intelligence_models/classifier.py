"""
EARTHLENS AI — Land Use Classifier
=====================================
Author  : Gouragopal Mohapatra
Purpose : ML-based land use classification from satellite bands
Model   : Random Forest Classifier
Classes : Water, Urban, Bare Soil, Sparse Veg, Dense Veg
"""

import numpy as np
from pathlib import Path
from loguru import logger

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from earthlens_core.analysis_engine.preprocessing import preprocess_band


# ── Class Labels ───────────────────────────────────────────────────────────────
LAND_CLASSES = {
    0: "Water",
    1: "Urban / Bare Soil",
    2: "Sparse Vegetation",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}

LAND_COLORS = {
    0: "#2166ac",   # Blue   — Water
    1: "#d4a56a",   # Brown  — Urban
    2: "#d9ef8b",   # Lt Green — Sparse
    3: "#66bd63",   # Green  — Moderate
    4: "#1a7837",   # Dk Green — Dense
}


# ── Feature Engineering ────────────────────────────────────────────────────────
def extract_features(bands: dict) -> np.ndarray:
    """
    Extract spectral features from bands.

    Input : {"B02": array, "B03": array, "B04": array,
              "B08": array, "B11": array, "B12": array}
    Output: feature array (H*W, n_features)

    Features:
        Raw bands   : B02, B03, B04, B08, B11, B12
        NDVI        : (B08 - B04) / (B08 + B04)
        NDWI        : (B03 - B08) / (B03 + B08)
        MNDWI       : (B03 - B11) / (B03 + B11)
        NDBI        : (B11 - B08) / (B11 + B08)  ← Built-up index
        EVI         : 2.5 * (B08-B04) / (B08 + 6*B04 - 7.5*B02 + 1)
    """
    b02 = bands["B02"].astype(np.float32)
    b03 = bands["B03"].astype(np.float32)
    b04 = bands["B04"].astype(np.float32)
    b08 = bands["B08"].astype(np.float32)
    b11 = bands["B11"].astype(np.float32)
    b12 = bands["B12"].astype(np.float32)

    def safe_divide(a, b):
        denom = b.copy()
        denom[denom == 0] = np.nan
        return np.where(np.isnan(denom), 0, a / denom)

    # Spectral indices
    ndvi  = safe_divide(b08 - b04, b08 + b04)
    ndwi  = safe_divide(b03 - b08, b03 + b08)
    mndwi = safe_divide(b03 - b11, b03 + b11)
    ndbi  = safe_divide(b11 - b08, b11 + b08)
    evi   = 2.5 * safe_divide(
                b08 - b04,
                b08 + 6*b04 - 7.5*b02 + 1
            )

    # Stack all features
    feature_stack = np.stack([
        b02, b03, b04, b08, b11, b12,
        ndvi, ndwi, mndwi, ndbi, evi,
    ], axis=-1)

    # Reshape to (pixels, features)
    h, w, f = feature_stack.shape
    features = feature_stack.reshape(-1, f)

    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    logger.info(f"Features extracted | shape={features.shape}")
    return features


# ── Generate Training Labels ───────────────────────────────────────────────────
def generate_labels(bands: dict) -> np.ndarray:
    """
    Generate training labels from spectral rules.
    Used when no ground truth is available.

    Rules:
        NDVI  < 0.0              → Water (0)
        NDBI  > 0.1              → Urban (1)
        NDVI  < 0.2              → Bare Soil / Urban (1)
        NDVI  0.2 - 0.4          → Sparse Veg (2)
        NDVI  0.4 - 0.6          → Moderate Veg (3)
        NDVI  > 0.6              → Dense Veg (4)
    """
    b04 = bands["B04"].astype(np.float32)
    b08 = bands["B08"].astype(np.float32)
    b03 = bands["B03"].astype(np.float32)
    b11 = bands["B11"].astype(np.float32)

    def safe_divide(a, b):
        denom = b.copy()
        denom[denom == 0] = np.nan
        return np.where(np.isnan(denom), 0, a / denom)

    ndvi = safe_divide(b08 - b04, b08 + b04)
    ndbi = safe_divide(b11 - b08, b11 + b08)

    labels = np.zeros(ndvi.shape, dtype=np.uint8)

    labels[ndvi >= 0.6]              = 4   # Dense Veg
    labels[(ndvi >= 0.4) & (ndvi < 0.6)] = 3   # Moderate Veg
    labels[(ndvi >= 0.2) & (ndvi < 0.4)] = 2   # Sparse Veg
    labels[(ndvi >= 0.0) & (ndvi < 0.2)] = 1   # Urban/Bare
    labels[ndvi < 0.0]               = 0   # Water

    logger.info("Training labels generated from spectral rules")
    return labels.flatten()


# ── Train Model ────────────────────────────────────────────────────────────────
def train_classifier(bands: dict,
                     labels: np.ndarray = None,
                     save_path: Path = None) -> tuple:
    """
    Train Random Forest classifier.

    Returns: (model, scaler, report)
    """
    logger.info("Training Land Use Classifier...")

    # Extract features
    features = extract_features(bands)

    # Generate labels if not provided
    if labels is None:
        labels = generate_labels(bands)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels,
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 15,
        n_jobs       = -1,
        random_state = 42,
        class_weight = "balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names = list(LAND_CLASSES.values()),
        output_dict  = True,
    )

    logger.success(f"Model trained! Accuracy: {accuracy:.4f}")

    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, save_path)
        logger.success(f"Model saved: {save_path}")

    return model, scaler, report


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(bands     : dict,
            model     : RandomForestClassifier,
            scaler    : StandardScaler) -> np.ndarray:
    """
    Predict land use class for each pixel.
    Returns classified map (H, W) with class indices.
    """
    # Get image shape
    sample_band = list(bands.values())[0]
    if isinstance(sample_band, np.ndarray):
        h, w = sample_band.shape
    else:
        h, w = 256, 256

    features = extract_features(bands)
    features = scaler.transform(features)

    predictions = model.predict(features)
    classified  = predictions.reshape(h, w).astype(np.uint8)

    logger.success("Land use classification complete!")
    return classified


# ── Load Model ─────────────────────────────────────────────────────────────────
def load_classifier(model_path: Path) -> tuple:
    """
    Load saved model and scaler.
    Returns: (model, scaler)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data   = joblib.load(model_path)
    model  = data["model"]
    scaler = data["scaler"]

    logger.success(f"Model loaded: {model_path}")
    return model, scaler


# ── Classification Stats ───────────────────────────────────────────────────────
def classification_stats(classified: np.ndarray) -> dict:
    """
    Calculate per-class coverage statistics.
    """
    total  = classified.size
    stats  = {}

    for class_id, class_name in LAND_CLASSES.items():
        count = int((classified == class_id).sum())
        pct   = round((count / total) * 100, 2)
        stats[class_name] = {
            "pixels"  : count,
            "coverage": pct,
        }

    logger.info(f"Classification stats: {stats}")
    return stats


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def run_classification_pipeline(
    bands      : dict,
    model_path : Path = None,
    output_dir : Path = None,
) -> dict:
    """
    Complete classification pipeline:
    Features → Train → Predict → Stats → Save

    Returns: {
        "classified" : array,
        "stats"      : dict,
        "report"     : classification report,
        "model"      : trained model,
        "saved_to"   : output path
    }
    """
    logger.info("Starting Land Use Classification pipeline...")

    model_path = Path(model_path) if model_path else \
                 Path("earthlens_data/processed_insights/classifier_model.pkl")

    # Load or train model
    if model_path.exists():
        logger.info("Loading existing model...")
        model, scaler = load_classifier(model_path)
        report = {}
    else:
        logger.info("Training new model...")
        model, scaler, report = train_classifier(
            bands     = bands,
            save_path = model_path,
        )

    # Predict
    classified = predict(bands, model, scaler)

    # Stats
    stats = classification_stats(classified)

    result = {
        "classified" : classified,
        "stats"      : stats,
        "report"     : report,
        "model"      : model,
        "saved_to"   : None,
    }

    # Save classified map
    if output_dir:
        import rasterio
        from rasterio.transform import from_bounds

        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "land_use_classified.tif"

        sample = list(bands.values())[0]
        if hasattr(sample, "shape"):
            h, w = sample.shape
        else:
            h, w = 256, 256

        transform = from_bounds(77.05, 28.40, 77.35, 28.75, w, h)

        with rasterio.open(
            output_path, "w",
            driver    = "GTiff",
            height    = h,
            width     = w,
            count     = 1,
            dtype     = np.uint8,
            crs       = "EPSG:4326",
            transform = transform,
        ) as dst:
            dst.write(classified, 1)

        result["saved_to"] = str(output_path)
        logger.success(f"Classified map saved: {output_path}")

    return result