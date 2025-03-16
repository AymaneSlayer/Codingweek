import json
import joblib  # Pour charger un modèle entraîné
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def predict(model_path, json_data):
    model = joblib.load(model_path)
    X_test = np.array(json_data["features"])  
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    json_file = "test_data.json"  
    model_file = "model.pkl"  

    data = load_json(json_file)
    results = predict(model_file, data)
    
    print("Prédictions:", results)
