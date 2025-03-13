import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Pour la modélisation
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
data = pd.read_csv(url, header=0)
data = data.replace('?', pd.NA) # Replace '?' with NA
data = data.apply(pd.to_numeric, errors='coerce') # Convert to numeric values

threshold = len(data) * 0.5  
data_cleaned = data.dropna(thresh=threshold, axis=1)
data_cleaned = data_cleaned.copy()  # Pour éviter les avertissements

# Conversion de "Biopsy" en entier
data_cleaned.loc[:, "Biopsy"] = data_cleaned["Biopsy"].astype(int)
# Liste des colonnes avant et après traitement
columns_before = data.columns
columns_after = data_cleaned.columns

# Colonnes supprimées
removed_columns = list(set(columns_before) - set(columns_after))
# Liste des colonnes supposées numériques (pour conversion explicite)
numeric_cols_suspected = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis"
]
# Conversion explicite en float pour les colonnes suspectées
for col in numeric_cols_suspected:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
columns_to_treat = ['Age', 'First sexual intercourse']
for col in columns_to_treat:
    skew_val = data_cleaned[col].skew()
    data_cleaned[col + '_log'] = np.log1p(data_cleaned[col])
    winsorized_values = np.array(winsorize(data_cleaned[col + '_log'], limits=(0.05, 0.05)))
    data_cleaned[col + '_log_winsorized'] = winsorized_values
for col in data.columns:
    if col == "Biopsy":
        continue
    if data[col].dtype in [np.float64, np.int64]:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        mode_val = data[col].mode(dropna=True)
        if not mode_val.empty:
            data[col].fillna(mode_val[0], inplace=True)
        else:
            data[col].fillna("Inconnu", inplace=True)
# Vérifier les valeurs manquantes après imputation
missing_values = data.isnull().sum()

# Séparation des features (X) et de la cible (y) à partir de data_cleaned
X = data_cleaned.drop(columns=['Biopsy'])
y = data_cleaned['Biopsy']

# Division en ensembles d'entraînement et de test (sans stratification ici pour SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation des valeurs manquantes dans X_train avec la médiane (pour SMOTE)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# Application de SMOTE sur l'ensemble d'entraînement imputé
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)
nouveau_data = pd.DataFrame(X_train_res, columns=X.columns)
nouveau_data['Biopsy'] = y_train_res

# Ajout de la colonne 'Biopsy' au DataFrame 'data_final'
data_final = nouveau_data.copy()

# Retirer les colonnes non nécessaires
colonnes_a_exclure = ['Age', 'First sexual intercourse']
data_final = data_final.drop(columns=colonnes_a_exclure)

# Calcul de la matrice de corrélation en valeurs absolues
corr_matrix = data_final.corr().abs()

# Extraction de la partie supérieure triangulaire pour éviter les doublons
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identification des colonnes à supprimer si corrélation > 0.8 avec au moins une autre variable
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

# Suppression des colonnes identifiées
data_final_reduced = data_final.drop(columns=to_drop)

# Ajout de la colonne 'Biopsy' au DataFrame réduit
data_final_reduced['Biopsy'] = data_final['Biopsy']
def reduce_memory_usage(df):
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':  # Pour les colonnes numériques
            c_min = df[col].min()
            c_max = df[col].max()

            # Conversion des types int en fonction de la plage des valeurs
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        else:  
            df[col] = df[col].astype('category')
    
    return df


# Exemple d'utilisation de la fonction sur votre DataFrame 'data_final_reduced'
data_final_reduced_optimized = reduce_memory_usage(data_final_reduced)

data_final_reduced_optimized.to_csv('processed_data.csv', index=False)