import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, 
  f1_score, confusion_matrix, classification_report
)
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_preprocessed_data(data_dir):
  """Load data yang sudah dipreprocess."""
  train = pd.read_csv(f"{data_dir}/train.csv")
  test = pd.read_csv(f"{data_dir}/test.csv")

  X_train = train.drop('target', axis=1)
  y_train = train['target']
  X_test = test.drop('target', axis=1)
  y_test = test['target']

  return X_train, X_test, y_train, y_test

def train_with_tuning():
  """Melatih model dengan hyperparameter tuning dan manual logging."""

  X_train, X_test, y_train, y_test = load_preprocessed_data(
    "preprocessing/dataset_preprocessing"
  )

  mlflow.set_experiment("MSML-Experiment-Tuning")

  # === Hyperparameter Grid ===
  param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
  }

  # === Grid Search ===
  rf = RandomForestClassifier(random_state=42)
  grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
  )
  grid_search.fit(X_train, y_train)

  # Ambil model terbaik
  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)

  # === Manual Logging ke MLflow ===
  with mlflow.start_run(run_name="RandomForest-tuned"):

    # --- Log Parameters ---
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("test_size", 0.2)

    # --- Log Metrics (sama seperti yang ada di autolog) ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    # --- Log Model ---
    mlflow.sklearn.log_model(best_model, "model")

    # --- Log Artifacts Tambahan ---
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    # 2. Feature Importance
    feature_importance = pd.DataFrame({
      'feature': X_train.columns,
      'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact('feature_importance.png')

    # 3. Classification Report (sebagai text file)
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
      f.write(report)
    mlflow.log_artifact('classification_report.txt')

    print(f"\nClassification Report:\n{report}")

    # Cleanup file lokal
    for f in ['confusion_matrix.png', 'feature_importance.png', 'classification_report.txt']:
      if os.path.exists(f):
        os.remove(f)

if __name__ == "__main__":
  train_with_tuning()
  print("\n✅ Training dengan tuning selesai!")
  print("   Jalankan: mlflow ui")
  print("   Buka: http://127.0.0.1:5000")