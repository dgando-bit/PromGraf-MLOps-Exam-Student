import datetime
import io
import logging
import os
import sys
import time
import zipfile
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import requests

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "cnt"
PREDICTION = "prediction"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]
ALL_FEATS = NUM_FEATS + CAT_FEATS
DTEDAY_COL_NAME = "dteday"

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = os.getenv("DATA_DIR", "/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "bike_model.pkl")
REFERENCE_PATH = os.path.join(MODEL_DIR, "reference_data.parquet")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model: Optional[RandomForestRegressor] = None
reference_data: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Evidently DataDefinition — instanciée une seule fois
# ---------------------------------------------------------------------------
DATA_DEFINITION = DataDefinition(
    numerical_columns=NUM_FEATS + [TARGET, PREDICTION],
    categorical_columns=CAT_FEATS,
    regression=[Regression(target=TARGET, prediction=PREDICTION)],
)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Prometheus — registre dédié + métriques
# ---------------------------------------------------------------------------
registry = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests, labelled by endpoint, method and HTTP status code.",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "Duration of API requests in seconds.",
    ["endpoint", "method", "status_code"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

model_rmse_score = Gauge(
    "model_rmse_score",
    "RMSE du modèle sur le dernier batch évalué.",
    registry=registry,
)

model_mae_score = Gauge(
    "model_mae_score",
    "MAE du modèle sur le dernier batch évalué.",
    registry=registry,
)

model_mape_score = Gauge(
    "model_mape_score",
    "MAPE (%) du modèle sur le dernier batch évalué.",
    registry=registry,
)

model_r2_score = Gauge(
    "model_r2_score",
    "R² du modèle sur le dernier batch évalué.",
    registry=registry,
)

# --- Métrique bonus : détection de dérive Evidently ---
#
# Pourquoi evidently_drift_detected ?
# C'est une Gauge binaire (0 = pas de dérive, 1 = dérive détectée) issue du
# rapport Evidently DataDriftPreset. Elle permet de déclencher une alerte
# Grafana/Alertmanager dès qu'une dérive de distribution est détectée sur les
# données courantes par rapport à la référence de janvier 2011. Combinée aux
# Gauges RMSE/MAE/R2, elle fournit le signal le plus direct pour savoir si le
# modèle doit être ré-entraîné (dérive data) ou si ses performances ont
# simplement dégradé (dérive de performance). C'est typiquement la métrique
# que l'on branche sur un alert rule "drift_detected == 1".
evidently_drift_detected = Gauge(
    "evidently_drift_detected",
    "1 si une dérive est détectée par Evidently, 0 sinon.",
    registry=registry,
)


# ---------------------------------------------------------------------------
# Middleware Prometheus
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    status_code = str(response.status_code)
    api_requests_total.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
    api_request_duration_seconds.labels(endpoint=endpoint, method=method, status_code=status_code).observe(duration)
    return response


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_data() -> pd.DataFrame:
    logger.info("Fetching data from UCI archive...")
    try:
        content = requests.get(DATA_URL, verify=False, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(z.open("hour.csv"), header=0, sep=",", parse_dates=[DTEDAY_COL_NAME])
        logger.info("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching data: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error processing data: %s", e)
        sys.exit(1)


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing raw data...")
    raw_data["hr"] = raw_data["hr"].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row[DTEDAY_COL_NAME].date(), datetime.time(row.hr)),
        axis=1,
    )
    raw_data = raw_data.sort_index()
    logger.info("Data processed successfully.")
    return raw_data


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _train_and_predict_reference_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, np.ndarray]:
    X, y = df[ALL_FEATS], df[TARGET]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    preds = rf.predict(X)
    logger.info("Model trained on %d samples. In-sample RMSE: %.2f", len(y), np.sqrt(np.mean((y - preds) ** 2)))
    return rf, preds


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[(df["yr"] == 0) & (df["mnth"] == 1)].copy()
    logger.info("Filtered data to %d samples for training.", len(filtered))
    return filtered


def train_and_save() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    raw = _fetch_data()
    processed = _process_data(raw)
    df = filter_data(processed)
    df = df[ALL_FEATS + [TARGET]].copy()
    for col in CAT_FEATS:
        df[col] = df[col].astype(int)
    for col in NUM_FEATS:
        df[col] = df[col].astype(float)
    df[TARGET] = df[TARGET].astype(float)
    trained_model, preds = _train_and_predict_reference_model(df)
    joblib.dump(trained_model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    ref = df.copy()
    ref[PREDICTION] = preds
    ref.to_parquet(REFERENCE_PATH, index=False)
    logger.info("Reference data saved to %s", REFERENCE_PATH)


# ---------------------------------------------------------------------------
# Evidently metric extraction — API 0.7.x
# ---------------------------------------------------------------------------

def _extract_evidently_metrics(result) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    """Extrait RMSE, MAE, R2 et drift depuis result._metrics (Evidently 0.7.x)."""
    rmse_val: Optional[float] = None
    mae_val: Optional[float] = None
    r2_val: Optional[float] = None
    drift_detected: int = 0

    try:
        for metric_obj in result._metrics.values():
            try:
                params = metric_obj.metric_value_location.metric.params
                metric_type: str = params.get("type", "")
            except Exception:
                metric_type = ""

            display_name: str = getattr(metric_obj, "display_name", "") or ""

            if "RMSE" in metric_type or "RMSE" in display_name:
                val = _get_single_value(metric_obj)
                if val is not None and rmse_val is None:
                    rmse_val = val
            elif "MAE" in metric_type or "AbsError" in metric_type or "MAE" in display_name or "Abs Error" in display_name:
                val = _get_single_value(metric_obj)
                if val is not None and mae_val is None:
                    mae_val = val
            elif "R2" in metric_type or "R2Score" in metric_type or "R2" in display_name or "R²" in display_name:
                val = _get_single_value(metric_obj)
                if val is not None and r2_val is None:
                    r2_val = val
            elif "DriftedFeaturesCount" in metric_type or "DatasetDrift" in metric_type or "Share" in display_name:
                val = _get_single_value(metric_obj)
                if val is not None and val > 0.5:
                    drift_detected = 1

    except Exception as e:
        logger.warning("Evidently _metrics extraction failed: %s", e)

    return rmse_val, mae_val, r2_val, drift_detected


def _get_single_value(metric_obj) -> Optional[float]:
    if hasattr(metric_obj, "value") and isinstance(getattr(metric_obj, "value"), (int, float)):
        return float(metric_obj.value)
    if hasattr(metric_obj, "mean") and hasattr(metric_obj.mean, "value"):
        return float(metric_obj.mean.value)
    return None


# ---------------------------------------------------------------------------
# MAPE — calcul robuste (exclut les y_true == 0 pour éviter division par zéro)
# ---------------------------------------------------------------------------

def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le MAPE en % en excluant les échantillons où y_true == 0.
    sklearn.mean_absolute_percentage_error fait déjà cette exclusion,
    mais on garde une implémentation explicite pour la lisibilité.

    Retourne NaN si tous les y_true sont nuls (cas dégénéré).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning("MAPE undefined: all y_true values are 0.")
        return float("nan")

    mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)
    excluded = (~mask).sum()
    if excluded > 0:
        logger.info("MAPE: excluded %d samples with y_true=0 (out of %d).", excluded, len(y_true))
    return mape


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date (YYYY-MM-DD).")


class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)


class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points with features and true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period")
    model_config = {"arbitrary_types_allowed": True}


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    global model, reference_data
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `make train` first.")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    if os.path.exists(REFERENCE_PATH):
        reference_data = pd.read_parquet(REFERENCE_PATH)
        logger.info("Reference data loaded (%d rows, columns: %s)", len(reference_data), list(reference_data.columns))
    else:
        logger.warning("Reference data not found at %s — drift detection unavailable.", REFERENCE_PATH)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def read_root():
    return {"message": "Bike Sharing Predictor API. Use /predict or /evaluate."}


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None, "reference_data_loaded": reference_data is not None}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

#    import random
#    t = [503, 500, 512, 501, 504, 500]
#    element_aleatoire = random.choice(t)
#    raise HTTPException(status_code=element_aleatoire, detail="Model not loaded. error aléatoire pour test de monitoring.%s", element_aleatoire)

    input_dict = input_data.dict()
    input_dict.pop("dteday")
    features = pd.DataFrame([input_dict])
    try:
        prediction = float(model.predict(features[ALL_FEATS])[0])
        return PredictionOutput(predicted_count=prediction)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    """
    Métriques retournées et exposées via Prometheus :
      - RMSE  → model_rmse_score
      - MAE   → model_mae_score
      - MAPE  → model_mape_score  (exclut les y_true=0)
      - R²    → model_r2_score
      - Drift → evidently_drift_detected (0/1)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if reference_data is None:
        raise HTTPException(status_code=503, detail="Reference data not loaded.")

    try:
        current_df = pd.DataFrame(payload.data)

        # Supprimer dteday — envoyé par run_evaluation.py mais pas une feature ML
        if DTEDAY_COL_NAME in current_df.columns:
            current_df = current_df.drop(columns=[DTEDAY_COL_NAME])

        # Cast dtypes
        for col in CAT_FEATS:
            if col in current_df.columns:
                current_df[col] = current_df[col].astype(int)
        for col in NUM_FEATS:
            if col in current_df.columns:
                current_df[col] = current_df[col].astype(float)
        current_df[TARGET] = current_df[TARGET].astype(float)

        # Inférence
        current_df[PREDICTION] = model.predict(current_df[ALL_FEATS]).astype(float)

        # --- Rapport Evidently ---
        ref_dataset = Dataset.from_pandas(reference_data, data_definition=DATA_DEFINITION)
        cur_dataset = Dataset.from_pandas(current_df, data_definition=DATA_DEFINITION)
        report = Report([RegressionPreset(), DataDriftPreset()])
        result = report.run(cur_dataset, ref_dataset)

        # --- Extraction Evidently ---
        rmse_val, mae_val, r2_val, drift_detected = _extract_evidently_metrics(result)

        # --- Fallback sklearn (source de vérité garantie) ---
        y_true = current_df[TARGET].values
        y_pred = current_df[PREDICTION].values

        if rmse_val is None:
            rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            logger.info("RMSE via sklearn fallback: %.4f", rmse_val)

        if mae_val is None:
            mae_val = float(mean_absolute_error(y_true, y_pred))
            logger.info("MAE via sklearn fallback: %.4f", mae_val)

        if r2_val is None:
            r2_val = float(r2_score(y_true, y_pred))
            logger.info("R2 via sklearn fallback: %.4f", r2_val)

        # MAPE — toujours calculé via sklearn (non fourni par Evidently)
        mape_val = _compute_mape(y_true, y_pred)

        # --- Mise à jour des Gauges Prometheus ---
        model_rmse_score.set(rmse_val)
        model_mae_score.set(mae_val)
        model_r2_score.set(r2_val)
        evidently_drift_detected.set(drift_detected)
        if not np.isnan(mape_val):
            model_mape_score.set(mape_val)

        logger.info(
            "Evaluation '%s' — RMSE=%.3f MAE=%.3f MAPE=%.2f%% R2=%.3f drift=%d items=%d",
            payload.evaluation_period_name,
            rmse_val, mae_val, mape_val, r2_val, drift_detected, len(current_df),
        )

        return EvaluationReportOutput(
            message=f"Evaluation completed for period '{payload.evaluation_period_name}'.",
            rmse=rmse_val,
            mape=mape_val if not np.isnan(mape_val) else None,
            mae=mae_val,
            r2score=r2_val,
            drift_detected=drift_detected,
            evaluated_items=len(current_df),
        )

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

@app.post("/trigger-drift")
async def trigger_drift():
    """
    Simule une dérive en forçant evidently_drift_detected = 1
    et en poussant un RMSE très élevé dans les Gauges Prometheus.
    Utilisé par 'make fire-alert' pour tester l'alerte Grafana.
    Justification : teste l'alerte 'DriftDetected' (evidently_drift_detected == 1)
    et 'RMSETooHigh' (model_rmse_score > 60).
    """
    evidently_drift_detected.set(1)
    model_rmse_score.set(999.0)
    model_mae_score.set(999.0)
    model_r2_score.set(0.0)
    logger.warning("🔥 trigger-drift appelé — métriques forcées pour tester les alertes.")
    return {
        "message": "Drift simulé avec succès.",
        "evidently_drift_detected": 1,
        "model_rmse_score": 999.0,
        "alert_tested": ["DriftDetected", "RMSETooHigh"],
    }