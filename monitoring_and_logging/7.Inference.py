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
        # Palmer Penguins dataset (6 fitur setelah di-preprocess/scaling):
        features = [
            round(random.uniform(-2.0, 2.0), 4),   # island (scaled)
            round(random.uniform(-2.0, 2.0), 4),   # culmen_length_mm (scaled)
            round(random.uniform(-2.0, 2.0), 4),   # culmen_depth_mm (scaled)
            round(random.uniform(-2.0, 2.0), 4),   # flipper_length_mm (scaled)
            round(random.uniform(-2.0, 2.0), 4),   # body_mass_g (scaled)
            round(random.uniform(-2.0, 2.0), 4),   # sex (scaled)
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
        