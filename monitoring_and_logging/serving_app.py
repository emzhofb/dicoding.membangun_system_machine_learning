from fastapi import FastAPI, Request
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
import mlflow.sklearn
import pandas as pd
import numpy as np
import time
import os
# pyrefly: ignore [missing-import]
from prometheus_client import (
  Counter, Histogram, Gauge, Summary,
  generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

app = FastAPI(title="ML Model Serving API")

# ===== Load Model =====
# Sesuaikan path ini dengan lokasi model MLflow kamu
# Bisa menggunakan run_id dari MLflow
MODEL_PATH = None  # Akan di-set saat startup
model = None

# ===== Prometheus Metrics =====
# Metric 1: Total request counter
REQUEST_COUNT = Counter(
  'model_request_total', 
  'Total jumlah request prediksi',
  ['method', 'endpoint', 'status']
)

# Metric 2: Request latency/duration
REQUEST_LATENCY = Histogram(
  'model_request_duration_seconds',
  'Waktu proses request prediksi dalam detik',
  ['endpoint'],
  buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Metric 3: Prediction distribution
PREDICTION_COUNTER = Counter(
  'model_predictions_total',
  'Distribusi hasil prediksi',
  ['predicted_class']
)

# Metric 4: Model confidence/probability
PREDICTION_CONFIDENCE = Histogram(
  'model_prediction_confidence',
  'Distribusi confidence score prediksi',
  buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metric 5: Active requests gauge
ACTIVE_REQUESTS = Gauge(
  'model_active_requests',
  'Jumlah request yang sedang diproses'
)

# Metric 6: Error counter
ERROR_COUNT = Counter(
  'model_errors_total',
  'Total jumlah error',
  ['error_type']
)

# Metric 7: Feature input summary
FEATURE_VALUES = Summary(
  'model_input_feature_values',
  'Summary statistik input fitur'
)

# Metric 8: Inference time (per-step)
INFERENCE_TIME = Gauge(
  'model_inference_time_seconds',
  'Waktu inferensi terakhir dalam detik'
)

# Metric 9: Total predictions
TOTAL_PREDICTIONS = Counter(
  'model_total_predictions',
  'Total semua prediksi yang telah dilakukan'
)

# Metric 10: Model uptime
MODEL_UPTIME = Gauge(
  'model_uptime_seconds',
  'Uptime model serving dalam detik'
)
START_TIME = time.time()


class PredictionInput(BaseModel):
  """Schema input untuk prediksi. Sesuaikan dengan fitur dataset kamu."""
  features: list  # List of feature values


class PredictionOutput(BaseModel):
  prediction: int
  confidence: float


@app.on_event("startup")
async def load_model():
  """Load model saat aplikasi startup."""
  global model
  # OPSI 1: Load dari MLflow run
  # Ganti <RUN_ID> dengan run_id dari MLflow UI
  # model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

  # OPSI 2: Load dari path lokal (lebih simpel untuk pemula)
  # Cari path model di folder mlruns
  # model = mlflow.sklearn.load_model("mlruns/0/<run_id>/artifacts/model")

  # OPSI 3: Latih ulang model sederhana (paling gampang untuk testing)
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_iris

  iris = load_iris()
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(iris.data, iris.target)
  print("✅ Model loaded successfully!")


@app.get("/")
async def root():
  """Health check endpoint."""
  uptime = time.time() - START_TIME
  MODEL_UPTIME.set(uptime)
  return {"status": "healthy", "uptime_seconds": round(uptime, 2)}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
  """Endpoint untuk melakukan prediksi."""
  ACTIVE_REQUESTS.inc()
  start_time = time.time()

  try:
    # Konversi input
    features = np.array(input_data.features).reshape(1, -1)

    # Log feature values
    for val in input_data.features:
      FEATURE_VALUES.observe(val)

    # Prediksi
    prediction = model.predict(features)[0]

    # Confidence (probability)
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='success').inc()
    REQUEST_LATENCY.labels(endpoint='/predict').observe(duration)
    PREDICTION_COUNTER.labels(predicted_class=str(prediction)).inc()
    PREDICTION_CONFIDENCE.observe(confidence)
    INFERENCE_TIME.set(duration)
    TOTAL_PREDICTIONS.inc()

    return PredictionOutput(
      prediction=int(prediction),
      confidence=round(confidence, 4)
    )

  except Exception as e:
    ERROR_COUNT.labels(error_type=type(e).__name__).inc()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='error').inc()
    raise

  finally:
    ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics():
  """Endpoint untuk Prometheus scraping."""
  MODEL_UPTIME.set(time.time() - START_TIME)
  return Response(
    content=generate_latest(),
    media_type=CONTENT_TYPE_LATEST
  )