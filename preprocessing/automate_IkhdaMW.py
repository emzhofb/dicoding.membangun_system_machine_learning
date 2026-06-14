"""
Script otomatisasi preprocessing data.
Mengkonversi langkah-langkah dari notebook eksperimen menjadi fungsi yang
dapat dijalankan secara otomatis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(filepath):
  """Load dataset dari file CSV."""
  df = pd.read_csv(filepath)
  print(f"Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
  return df

def clean_data(df):
  """Bersihkan data: handle missing values dan duplikat."""
  # Handle missing values
  initial_rows = df.shape[0]
  df = df.dropna()
  print(f"Missing values dihapus: {initial_rows - df.shape[0]} baris")

  # Handle duplikat
  initial_rows = df.shape[0]
  df = df.drop_duplicates()
  print(f"Duplikat dihapus: {initial_rows - df.shape[0]} baris")

  return df

def encode_features(df, categorical_columns=None):
  """Encode kolom kategorikal."""
  le = LabelEncoder()
  if categorical_columns:
    for col in categorical_columns:
      df[col] = le.fit_transform(df[col])
      print(f"Encoded kolom: {col}")
  return df

def split_and_scale(df, target_column, test_size=0.2, random_state=42):
  """Split data dan lakukan scaling."""
  X = df.drop(target_column, axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )

  scaler = StandardScaler()
  X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
  )
  X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
  )

  print(f"Train size: {X_train_scaled.shape}")
  print(f"Test size: {X_test_scaled.shape}")

  return X_train_scaled, X_test_scaled, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
  """Simpan data yang sudah dipreprocess."""
  os.makedirs(output_dir, exist_ok=True)

  train_data = X_train.copy()
  train_data['target'] = y_train.values
  train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

  test_data = X_test.copy()
  test_data['target'] = y_test.values
  test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

  print(f"Data tersimpan di: {output_dir}")

def run_preprocessing(input_path, target_column, output_dir, categorical_columns=None):
  """
  Fungsi utama yang menjalankan seluruh pipeline preprocessing.

  Args:
    input_path: Path ke dataset mentah
    target_column: Nama kolom target
    output_dir: Folder output untuk data terpreprocess
    categorical_columns: List kolom kategorikal (opsional)
  """
  print("=" * 50)
  print("PREPROCESSING PIPELINE")
  print("=" * 50)

  # Step 1: Load data
  print("\n[1/4] Loading data...")
  df = load_data(input_path)

  # Step 2: Clean data
  print("\n[2/4] Cleaning data...")
  df = clean_data(df)

  # Step 3: Encode
  print("\n[3/4] Encoding features...")
  df = encode_features(df, categorical_columns)

  # Step 4: Split dan scale
  print("\n[4/4] Splitting dan scaling...")
  X_train, X_test, y_train, y_test = split_and_scale(df, target_column)

  # Simpan hasil
  save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

  print("\n" + "=" * 50)
  print("PREPROCESSING SELESAI!")
  print("=" * 50)

  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  # ===== SESUAIKAN PARAMETER INI =====
  INPUT_PATH = "../iris.csv"          # Path ke dataset mentah
  TARGET_COLUMN = "species"                        # Nama kolom target
  OUTPUT_DIR = "dataset_preprocessing"        # Folder output
  CATEGORICAL_COLUMNS = None                      # List kolom kategorikal, misal: ["sex", "embarked"]
  # ===================================

  run_preprocessing(INPUT_PATH, TARGET_COLUMN, OUTPUT_DIR, CATEGORICAL_COLUMNS)
