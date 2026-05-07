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
CSV_PATH = os.path.join(DATA_DIR, "hour.csv")
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

# --- Métriques génériques HTTP ---

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests, labelled by endpoint, method and HTTP status code.",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "Duration of API requests in seconds, labelled by endpoint, method and HTTP status code.",
    ["endpoint", "method", "status_code"],
    # Buckets adaptés à une inférence ML légère (ms → quelques secondes)
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

# --- Métriques qualité modèle (mises à jour par /evaluate) ---

model_rmse_score = Gauge(
    "model_rmse_score",
    "Root Mean Squared Error du modèle sur le dernier batch évalué.",
    registry=registry,
)

model_mae_score = Gauge(
    "model_mae_score",
    "Mean Absolute Error du modèle sur le dernier batch évalué.",
    registry=registry,
)

model_r2_score = Gauge(
    "model_r2_score",
    "Coefficient de détermination R² du modèle sur le dernier batch évalué.",
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
    "1 si une dérive de distribution est détectée par Evidently sur le batch courant, 0 sinon.",
    registry=registry,
)


# ---------------------------------------------------------------------------
# Middleware : instrumentation automatique de TOUTES les routes HTTP
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """
    Intercepte chaque requête HTTP pour :
      - incrémenter api_requests_total avec les bons labels
      - mesurer et enregistrer la latence dans api_request_duration_seconds
    Le label status_code est renseigné APRÈS que la réponse est produite,
    ce qui garantit de capturer les codes d'erreur (422, 500…).
    """
    endpoint = request.url.path
    method = request.method
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start
    status_code = str(response.status_code)

    api_requests_total.labels(
        endpoint=endpoint, method=method, status_code=status_code
    ).inc()
    api_request_duration_seconds.labels(
        endpoint=endpoint, method=method, status_code=status_code
    ).observe(duration)

    return response


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    logger.info("Fetching data from UCI archive...")
    try:
        content = requests.get(DATA_URL, verify=False, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(
                z.open("hour.csv"), header=0, sep=",", parse_dates=[DTEDAY_COL_NAME]
            )
        logger.info("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching data: %s. Check URL or network connection.", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error processing fetched data: %s", e)
        sys.exit(1)


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data, setting a DatetimeIndex."""
    logger.info("Processing raw data...")
    raw_data["hr"] = raw_data["hr"].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row[DTEDAY_COL_NAME].date(), datetime.time(row.hr)
        ),
        axis=1,
    )
    raw_data = raw_data.sort_index()
    logger.info("Data processed successfully.")
    return raw_data


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _train_and_predict_reference_model(
    df: pd.DataFrame,
) -> tuple[RandomForestRegressor, np.ndarray]:
    """Train a RandomForestRegressor and return the model + in-sample predictions."""
    X = df[ALL_FEATS]
    y = df[TARGET]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    preds = rf.predict(X)
    logger.info(
        "Model trained on %d samples. In-sample RMSE: %.2f",
        len(y),
        np.sqrt(np.mean((y - preds) ** 2)),
    )
    return rf, preds


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only January 2011 (the reference training period)."""
    filtered = df[(df["yr"] == 0) & (df["mnth"] == 1)].copy()
    logger.info("Filtered data to %d samples for training.", len(filtered))
    return filtered


def train_and_save() -> None:
    """Full training pipeline: fetch → process → filter → train → persist."""
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
    dteday: datetime.date = Field(
        ...,
        example="2011-01-01",
        description="Date of the record in YYYY-MM-DD format.",
    )


class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)


class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of data points, each containing features and the true target ('cnt')."
        ),
    )
    evaluation_period_name: str = Field(
        "unknown_period",
        description="Name of the period being evaluated (e.g., 'week1_february').",
    )
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
# Startup: load model + reference data
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    global model, reference_data

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `make train` before starting the API."
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

    if os.path.exists(REFERENCE_PATH):
        reference_data = pd.read_parquet(REFERENCE_PATH)
        logger.info(
            "Reference data loaded (%d rows, columns: %s)",
            len(reference_data),
            list(reference_data.columns),
        )
    else:
        logger.warning(
            "Reference data not found at %s — drift detection will be unavailable.",
            REFERENCE_PATH,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def read_root():
    return {
        "message": (
            "Welcome to the Bike Sharing Predictor API. "
            "Use /predict to get bike counts or /evaluate to run drift reports."
        )
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "reference_data_loaded": reference_data is not None,
    }


@app.get("/metrics")
async def metrics():
    """
    Expose toutes les métriques Prometheus du registre dédié.
    Le middleware HTTP instrumente automatiquement cet endpoint lui-même,
    donc api_requests_total{endpoint="/metrics"} est aussi suivi.
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    """
    Prédit le nombre de vélos pour un enregistrement horaire.

    Le middleware Prometheus instrumente automatiquement cette route :
      - api_requests_total{endpoint="/predict", method="POST", status_code="200"}
      - api_request_duration_seconds{endpoint="/predict", ...}
    Aucune instrumentation manuelle supplémentaire n'est nécessaire ici.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_dict = input_data.dict()
    input_dict.pop("dteday")  # not a model feature
    features = pd.DataFrame([input_dict])

    try:
        prediction = float(model.predict(features[ALL_FEATS])[0])
        return PredictionOutput(predicted_count=prediction)
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    """
    Exécute un rapport Evidently (RegressionPreset + DataDriftPreset) sur un
    batch de données labellisées, puis met à jour les Gauges Prometheus :
      - model_rmse_score
      - model_mae_score
      - model_r2_score
      - evidently_drift_detected  ← métrique bonus (0/1)

    Le middleware Prometheus instrumente aussi cet endpoint pour
    api_requests_total et api_request_duration_seconds.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if reference_data is None:
        raise HTTPException(
            status_code=503,
            detail="Reference data not loaded — drift detection unavailable.",
        )

    try:
        current_df = pd.DataFrame(payload.data)

        # --- Cast dtypes ---
        for col in CAT_FEATS:
            if col in current_df.columns:
                current_df[col] = current_df[col].astype(int)
        for col in NUM_FEATS:
            if col in current_df.columns:
                current_df[col] = current_df[col].astype(float)
        current_df[TARGET] = current_df[TARGET].astype(float)

        # --- Inférence sur le batch courant ---
        current_df[PREDICTION] = model.predict(current_df[ALL_FEATS]).astype(float)

        # --- Rapport Evidently (nouvelle API 0.7.x) ---
        ref_dataset = Dataset.from_pandas(reference_data, data_definition=DATA_DEFINITION)
        cur_dataset = Dataset.from_pandas(current_df,    data_definition=DATA_DEFINITION)

        report = Report([RegressionPreset(), DataDriftPreset()])
        result = report.run(cur_dataset, ref_dataset)
        result_dict = result.dict()

        # --- Extraction des métriques scalaires depuis result.dict() ---
        metrics_list: list[dict] = result_dict.get("metrics", [])

        rmse_val: Optional[float] = None
        mae_val: Optional[float] = None
        r2_val: Optional[float] = None
        drift_detected: int = 0

        for m in metrics_list:
            metric_id: str = m.get("metric_id", "") or m.get("metric", "")
            res: dict = m.get("result", {}) or {}
            current_res: dict = res.get("current", res)

            if "RegressionQuality" in metric_id or "RegressionPreset" in metric_id:
                rmse_val = rmse_val or current_res.get("rmse")
                mae_val  = mae_val  or current_res.get("mean_abs_error") or current_res.get("mae")
                r2_val   = r2_val   or current_res.get("r2_score") or current_res.get("r2")

            if "DatasetDrift" in metric_id or "DataDrift" in metric_id:
                drift_detected = int(res.get("dataset_drift", False))

        # --- Mise à jour des Gauges Prometheus ---
        if rmse_val is not None:
            model_rmse_score.set(rmse_val)
        if mae_val is not None:
            model_mae_score.set(mae_val)
        if r2_val is not None:
            model_r2_score.set(r2_val)

        # Métrique bonus : 1 si dérive détectée, 0 sinon.
        # Permet de brancher une alerte Grafana sur evidently_drift_detected == 1
        # sans avoir à parser les logs ou le JSON du rapport.
        evidently_drift_detected.set(drift_detected)

        logger.info(
            "Evaluation '%s' — RMSE=%.3f MAE=%.3f R2=%.3f drift=%d items=%d",
            payload.evaluation_period_name,
            rmse_val or 0.0,
            mae_val or 0.0,
            r2_val or 0.0,
            drift_detected,
            len(current_df),
        )

        return EvaluationReportOutput(
            message=f"Evaluation completed for period '{payload.evaluation_period_name}'.",
            rmse=rmse_val,
            mape=None,
            mae=mae_val,
            r2score=r2_val,
            drift_detected=drift_detected,
            evaluated_items=len(current_df),
        )

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")