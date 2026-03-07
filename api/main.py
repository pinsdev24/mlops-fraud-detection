"""CI-7 / CI-8 — FastAPI prediction service with request logging.

Endpoints:
  GET  /health   — health check (used by Docker healthcheck)
  POST /predict  — predict fraud probability for one or more transactions
  GET  /model    — return model metadata (name, version, threshold)
  GET  /docs     — interactive Swagger UI (auto-generated)
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse, ModelInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "MLOps M2 — API de détection de fraude bancaire. "
        "Modèle XGBoost entraîné sur Credit Card Fraud Detection (UCI)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector-xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# CI-8 — prediction logs written to a JSONL file for post-hoc analysis
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG_PATH = LOG_DIR / "predictions.jsonl"

# Global model state
_model = None
_model_version = None
_model_uri = None


@app.on_event("startup")
async def load_model():
    """Load the model from MLflow at startup (CI-7)."""
    global _model, _model_version, _model_uri

    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        # Try to load the latest Production model
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info("Loading model from %s ...", model_uri)
        _model = mlflow.sklearn.load_model(model_uri)
        _model_uri = model_uri
        _model_version = MODEL_STAGE
        logger.info("Model loaded successfully: %s", model_uri)

    except Exception as e:
        # Fallback: load latest version regardless of stage
        logger.warning("Could not load %s stage — trying latest: %s", MODEL_STAGE, e)
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            _model = mlflow.sklearn.load_model(model_uri)
            _model_uri = model_uri
            _model_version = "latest"
            logger.info("Loaded fallback model: %s", model_uri)
        except Exception as e2:
            logger.error("Model loading FAILED: %s", e2)
            _model = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health():
    """Docker healthcheck endpoint."""
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model", response_model=ModelInfo, tags=["ops"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        name=MODEL_NAME,
        version=_model_version or "unknown",
        uri=_model_uri or "",
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: Request, payload: PredictRequest):
    """Predict fraud probability for one or more transactions.

    CI-8: Each request is logged with timestamp, model version, and score.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    try:
        # Convert input to DataFrame with proper column names
        # (required by the ColumnTransformer which selects Amount/Time by name)
        records = [t.dict() for t in payload.transactions]
        features = pd.DataFrame(records)
        probas = _model.predict_proba(features)[:, 1]
        threshold = payload.threshold or DEFAULT_THRESHOLD
        predictions = (probas >= threshold).astype(int).tolist()

        results = []
        for i, (proba, pred) in enumerate(zip(probas.tolist(), predictions)):
            results.append(
                {
                    "transaction_index": i,
                    "fraud_probability": round(proba, 6),
                    "is_fraud": bool(pred),
                    "threshold_used": threshold,
                }
            )

        # CI-8 — Log each prediction for post-hoc analysis
        request_id = str(uuid.uuid4())
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": _model_version,
            "model_uri": _model_uri,
            "n_transactions": len(payload.transactions),
            "threshold": threshold,
            "n_fraud_detected": sum(predictions),
            "scores": probas.tolist(),
        }
        with open(PRED_LOG_PATH, "a") as f:
            import json

            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            "Prediction | request_id=%s | n=%d | fraud_detected=%d | max_score=%.4f",
            request_id,
            len(payload.transactions),
            sum(predictions),
            max(probas),
        )

        return PredictResponse(
            request_id=request_id,
            model_version=_model_version or "unknown",
            results=results,
        )

    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
