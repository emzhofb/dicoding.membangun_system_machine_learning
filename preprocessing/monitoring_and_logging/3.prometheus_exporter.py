from prometheus_client import Counter, Histogram, Gauge, Summary

# === Daftar Metriks yang Dimonitor ===

# 1. Total request counter
REQUEST_COUNT = Counter(
    'model_request_total',
    'Total jumlah request prediksi',
    ['method', 'endpoint', 'status']
)

# 2. Request latency
REQUEST_LATENCY = Histogram(
    'model_request_duration_seconds',
    'Waktu proses request prediksi dalam detik',
    ['endpoint']
)

# 3. Distribusi prediksi
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Distribusi hasil prediksi',
    ['predicted_class']
)

# 4. Confidence score prediksi
PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Distribusi confidence score prediksi'
)

# 5. Active requests
ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Jumlah request yang sedang diproses'
)

# 6. Error counter
ERROR_COUNT = Counter(
    'model_errors_total',
    'Total jumlah error',
    ['error_type']
)

# 7. Feature input summary
FEATURE_VALUES = Summary(
    'model_input_feature_values',
    'Summary statistik input fitur'
)

# 8. Inference time
INFERENCE_TIME = Gauge(
    'model_inference_time_seconds',
    'Waktu inferensi terakhir'
)

# 9. Total predictions
TOTAL_PREDICTIONS = Counter(
    'model_total_predictions',
    'Total semua prediksi yang telah dilakukan'
)

# 10. Model uptime
MODEL_UPTIME = Gauge(
    'model_uptime_seconds',
    'Uptime model serving dalam detik'
)
