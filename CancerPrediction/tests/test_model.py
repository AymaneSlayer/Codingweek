import json
import numpy as np
import pandas as pd
import xgboost as xgb

def load_csv(file_path):
    """Charge les données à partir d'un fichier CSV."""
    data = pd.read_csv(file_path)
    return data.values  # Convertir en numpy array

def load_model(model_path):
    """Charge le modèle XGBoost à partir d'un fichier JSON."""
    model = xgb.Booster()
    model.load_model(model_path)  # Charger un modèle XGBoost depuis un fichier JSON
    return model

def predict(model, X_test):
    """Effectue une prédiction avec le modèle XGBoost."""
    dmatrix = xgb.DMatrix(X_test)  # Convertir en format XGBoost
    predictions = model.predict(dmatrix)
    return predictions

if __name__ == "__main__":
    csv_file = "CancerPrediction/data/processed_data.csv"  # Remplacez par le chemin réel de votre fichier CSV
    model_file = "CancerPrediction/models/xgboost_trained_model.json"  # Remplacez par le chemin de votre modèle JSON

    X_test = load_csv(csv_file)
    model = load_model(model_file)
    results = predict(model, X_test)
    
    print("Prédictions:", results)
