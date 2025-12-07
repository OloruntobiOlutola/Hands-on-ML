# ...existing code...
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import kagglehub
import __main__

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

RESULTS_DIR = "results"

def load_test_data():
    dataset_path = kagglehub.dataset_download("lakshmi25npathi/santander-customer-transaction-prediction-dataset")
    df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    # Remove the ID_code column
    df = df.drop(columns=['ID_code'])
    total_outliers = 0

    # Detect outliers using the IQR method

    outlier_indices = set()

    for col in df.columns.drop('target'):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_idx = df[outlier_mask].index.tolist()
        
        #print(f"Column: {col}, Outliers detected: {len(outlier_idx)}")
        
        # Add indices to the set (automatically removes duplicates)
        outlier_indices.update(outlier_idx)
        total_outliers += len(outlier_idx)

    # (Optional) View the unique outlier rows
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df_cleaned = df.drop(index=outlier_indices)
    X = df_cleaned.drop(columns=['target'])
    y = df_cleaned['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    rus = RandomUnderSampler(sampling_strategy='majority')
    X_resampled_down, y_resampled_down = rus.fit_resample(X_train_copy, y_train_copy)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler = StandardScaler()
    X_resampled_train_scaled = scaler.fit_transform(X_resampled_down)
    X_resampled_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test

def main():
    X_test, y_test = load_test_data()

    pkl_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.pkl")))
    if not pkl_files:
        print("Aucun fichier .pkl dans le dossier 'results/'.")
        return

    for p in pkl_files:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"\nModel: {name}  -- ERREUR au chargement: {e}")
            continue

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"\nModel: {name}  -- ERREUR predict(): {e}")
            continue

        print("\n" + "="*80)
        print(f"Model: {name}")
        print("-"*80)
        print(classification_report(y_test, y_pred, zero_division=0))
    print("\nÉvaluation terminée.")
    
if __name__ == "__main__":
    main()
# ...existing code...