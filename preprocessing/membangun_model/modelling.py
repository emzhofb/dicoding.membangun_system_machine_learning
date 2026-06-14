"""
modelling.py - Training model ML dengan MLflow autolog.
Kriteria 2 (Basic): Menggunakan autolog dari MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn

def load_preprocessed_data(data_dir):
  """Load data yang sudah dipreprocess."""
  train = pd.read_csv(f"{data_dir}/train.csv")
  test = pd.read_csv(f"{data_dir}/test.csv")

  X_train = train.drop('target', axis=1)
  y_train = train['target']
  X_test = test.drop('target', axis=1)
  y_test = test['target']

  return X_train, X_test, y_train, y_test

def train_model():
  """Melatih model dengan MLflow autolog."""

  # Load data
  X_train, X_test, y_train, y_test = load_preprocessed_data(
    "../dataset_preprocessing"
  )

  # Set MLflow experiment
  mlflow.set_experiment("MSML-Experiment")

  # Aktifkan autolog — ini akan otomatis mencatat:
  # - Parameters (n_estimators, max_depth, dll)
  # - Metrics (accuracy, f1, dll)
  # - Model artifact
  mlflow.sklearn.autolog()

  with mlflow.start_run(run_name="RandomForest-autolog"):
    # Buat dan latih model
    model = RandomForestClassifier(
      n_estimators=100,
      max_depth=10,
      random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung accuracy (autolog sudah mencatat ini, tapi kita print juga)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

if __name__ == "__main__":
  train_model()
  print("\n✅ Training selesai! Buka MLflow UI dengan command:")
  print("   mlflow ui")
  print("   Lalu buka http://127.0.0.1:5000 di browser")