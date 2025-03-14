import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer

def load_data(file_path):

    return pd.read_csv(file_path)

def process_data(df):

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
    
    # Imputer les valeurs manquantes par la médiane
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def main():
    old_data_file = "CancerPrediction/data/data.csv"           
    processed_data_file = "CancerPrediction/data/processed_data.csv"  
    
    try:
        old_data = load_data(old_data_file)
    except Exception as e:
        print(f"Erreur lors du chargement de {old_data_file} : {e}")
        return

    print("=== Ancienne Data (Données d'origine) ===")
    print(old_data.head())

    processed_data = process_data(old_data.copy())

    print("\n=== Data Processed (Données traitées) ===")
    print(processed_data.head())

    try:
        processed_data.to_csv(processed_data_file, index=False)
        print(f"\nLes données traitées ont été sauvegardées dans {processed_data_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {processed_data_file} : {e}")

if __name__ == "__main__":
    main()
