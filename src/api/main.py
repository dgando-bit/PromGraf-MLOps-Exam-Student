import datetime
import io
import logging
import os
import zipfile
from typing import Any, Optional
 
import joblib
import numpy as np
import pandas as pd
import requests

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestRegressor

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---



# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
ALL_FEATS = NUM_FEATS + CAT_FEATS
 
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = os.getenv("DATA_DIR", "/data")
CSV_PATH = os.path.join(DATA_DIR, "hour.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "bike_model.pkl")
REFERENCE_PATH = os.path.join(MODEL_DIR, "reference_data.parquet")

# ---------- Global state ----------
model: Optional[RandomForestRegressor] = None
reference_data: Optional[pd.DataFrame] = None

# ---------
# --- Data Ingestion and Preparation Functions ---
##------------------------------------------------

# ---------- Data helpers ---------- 
def _fetch_data() -> pd.DataFrame:
    """Télécharge et extrait le dataset si nécessaire, retourne le DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        logger.info("Downloading Bike Sharing dataset from UCI repository…")
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extract("hour.csv", DATA_DIR)
        logger.info("Dataset saved to %s", CSV_PATH)
    else:
        logger.info("Dataset already present at %s", CSV_PATH)
    return pd.read_csv(CSV_PATH)
 
 
def _process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre janvier 2011 et prépare les features."""
    df = df[(df["yr"] == 0) & (df["mnth"] == 1)].copy()
    df = df[ALL_FEATS + [TARGET]].reset_index(drop=True)
    # Vérification et conversion des types de données
    for col in CAT_FEATS:
        df[col] = df[col].astype(int)
    for col in NUM_FEATS:
        df[col] = df[col].astype(float)
    df[TARGET] = df[TARGET].astype(float)
    return df

# -------- Model training ----------
def _train_and_predict_reference_model(
    df: pd.DataFrame,
) -> tuple[RandomForestRegressor, np.ndarray]:
    """Entrainer un RandomForestRegressor sur le DataFrame fourni et retourner
    le modèle ajusté ainsi que ses prédictions sur l'échantillon."""
    X = df[ALL_FEATS]
    y = df[TARGET]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    preds = rf.predict(X)
    logger.info(
        "Modele entrainé sur %d échantillons. RMSE en échantillon: %.2f",
        len(y),
        np.sqrt(np.mean((y - preds) ** 2)),
    )
    return rf, preds

# -------- Pipeline d'entraînement complet ----------
def train_and_save() -> None:
    """Pipeline d'entraînement complet: fetch → process → train → persist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    raw = _fetch_data()
    processed = _process_data(raw)
    trained_model, preds = _train_and_predict_reference_model(processed)
 
    # Persistence du modèle
    joblib.dump(trained_model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
 
    # Persistence des données de référence (features + target + prediction) pour la détection de dérive
    ref = processed.copy()
    ref[PREDICTION] = preds
    ref.to_parquet(REFERENCE_PATH, index=False)
    logger.info("Reference data saved to %s", REFERENCE_PATH)


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# ---------------------------------------------------------------------------
# App startup: load model + reference data
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global model, reference_data
 
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Fichier du modèle non trouvé à {MODEL_PATH}. "
            "Exécutez `make train` avant de démarrer l'API."
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé depuis %s", MODEL_PATH)
 
    if os.path.exists(REFERENCE_PATH):
        reference_data = pd.read_parquet(REFERENCE_PATH)
        logger.info(
            "Données de référence chargées (%d lignes, colonnes: %s)",
            len(reference_data),
            list(reference_data.columns),
        )
    else:
        logger.warning(
            "Données de référence non trouvées à %s — la détection de dérive sera indisponible.",
            REFERENCE_PATH,
        )
 
# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "reference_data_loaded": reference_data is not None,
    }
