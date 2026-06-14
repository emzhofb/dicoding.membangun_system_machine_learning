import requests
import time
import random
import json

API_URL = "http://localhost:8000"

def send_prediction(features):
    """Kirim request prediksi ke API."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features}
    )
    return response.json()

def health_check():
    """Cek apakah API berjalan."""
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Health Check: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ API belum berjalan! Jalankan serving_app.py terlebih dahulu.")
        return False

def generate_traffic(n_requests=50):
    """Generate traffic untuk mengisi metriks Prometheus."""
    print(f"\n🔄 Mengirim {n_requests} request prediksi...")
    
    for i in range(n_requests):
        # Buat fitur random (sesuaikan jumlah fitur dengan dataset kamu)
        # Contoh untuk Iris dataset (4 fitur):
        features = [
            round(random.uniform(4.0, 8.0), 1),   # sepal_length
            round(random.uniform(2.0, 4.5), 1),    # sepal_width
            round(random.uniform(1.0, 7.0), 1),    # petal_length
            round(random.uniform(0.1, 2.5), 1),    # petal_width
        ]
        
        try:
            result = send_prediction(features)
            print(f"  Request {i+1}: Input={features} → Prediction={result['prediction']}, Confidence={result['confidence']}")
        except Exception as e:
            print(f"  Request {i+1}: ERROR - {e}")
        
        time.sleep(0.1)  # Jeda kecil antar request
    
    print(f"\n✅ Selesai! {n_requests} request terkirim.")
    print(f"Cek metrics di: {API_URL}/metrics")

if __name__ == "__main__":
    if health_check():
        generate_traffic(50)
        