import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
import json
import shap

# Pour la modélisation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Modèles
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier  # Assurez-vous d'avoir installé catboost (pip install catboost)
from sklearn.svm import SVC

# Pipeline et normalisation pour SVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('CancerPrediction/data/data.csv')
data = data.replace('?', pd.NA) # Replace '?' with NA
data = data.apply(pd.to_numeric, errors='coerce') # Convert to numeric values
print(data.head())
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
print("Colonnes supprimées : ", removed_columns)
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
print(data[numeric_cols_suspected].dtypes)
columns_to_treat = ['Age', 'First sexual intercourse']
for col in columns_to_treat:
    skew_val = data_cleaned[col].skew()
    data_cleaned[col + '_log'] = np.log1p(data_cleaned[col])
    winsorized_values = np.array(winsorize(data_cleaned[col + '_log'], limits=(0.05, 0.05)))
    data_cleaned[col + '_log_winsorized'] = winsorized_values
    print(f"Colonne '{col}' (skewness = {skew_val:.2f}): transformation log et winsorisation appliquées.")
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
print("Valeurs manquantes après imputation :")
print(missing_values)
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
print(nouveau_data.columns)
print("\nRépartition des classes après SMOTE:")
print(pd.Series(y_train_res).value_counts())
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

# Affichage des colonnes supprimées et des dimensions du DataFrame réduit
print("Colonnes supprimées :", to_drop)
print("Dimensions du DataFrame réduit :", data_final_reduced.shape)

# Affichage de la matrice de corrélation du DataFrame réduit via une heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data_final_reduced.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de Corrélation - Données Réduites")
plt.show()

# Affichage des colonnes du DataFrame réduit
print(data_final_reduced.columns)

print("Aperçu du dataset (5 premières lignes) :")
print(data_final_reduced.head())

print("\nDimensions (lignes, colonnes) :", data_final_reduced.shape)
print("\nInformations sur les colonnes :")
data_final_reduced.info()

print("\nStatistiques descriptives (variables numériques) :")
print(data_final_reduced.describe())

print("\nValeurs manquantes par colonne après imputation :")
print(data_final_reduced.isnull().sum())
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2  # Taille initiale en MB
    print(f'Mémoire avant optimisation : {start_mem:.2f} MB')
    
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
            
            # Conversion des types float en fonction de la plage des valeurs
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        else:  # Pour les colonnes 'object', on les convertit en 'category' si possible
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2  # Taille après optimisation
    print(f'Mémoire après optimisation : {end_mem:.2f} MB')
    print(f'Reduction de mémoire : {100 * (start_mem - end_mem) / start_mem:.2f}%')
    
    return df

# Exemple d'utilisation de la fonction sur votre DataFrame 'data_final_reduced'
data_final_reduced_optimized = reduce_memory_usage(data_final_reduced)

# Vérification de la cible
target_col = 'Biopsy'
if target_col in data_final_reduced_optimized.columns:
    print("\nRépartition de la cible :")
    print(data_final_reduced_optimized[target_col].value_counts())
    print(data_final_reduced_optimized[target_col].value_counts(normalize=True) * 100)
else:
    print(f"\nATTENTION : la colonne '{target_col}' n'existe pas.")
if target_col not in data_final_reduced_optimized.columns:
    print("Impossible de procéder à la modélisation : cible introuvable.")
    import sys
    sys.exit()

# Séparation des features (X) et de la cible (y)
X_final = data_final_reduced_optimized.drop(columns=['Biopsy'])
y_final = data_final_reduced_optimized['Biopsy']

# Division des données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Initialisation du modèle XGBClassifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Entraînement du modèle
xgb_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = xgb_model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Affichage des résultats
print(f"Accuracy du modèle : {accuracy:.4f}")
print(f"ROC AUC Score : {roc_auc:.4f}")

# Sauvegarder le modèle XGBoost sous format .json
model_filename = 'xgboost_trained_model.json'
xgb_model.save_model(model_filename)
print(f"Modèle XGBoost sauvegardé sous le nom : {model_filename}")

# Charger le modèle depuis un fichier .json
loaded_xgb_model = XGBClassifier()
loaded_xgb_model.load_model(model_filename)

# Faire des prédictions avec le modèle chargé
y_pred_loaded = loaded_xgb_model.predict(X_test)

# Réévaluation du modèle chargé
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
roc_auc_loaded = roc_auc_score(y_test, y_pred_loaded)

# Affichage des résultats pour le modèle chargé
print(f"Accuracy du modèle chargé : {accuracy_loaded:.4f}")
print(f"ROC AUC Score du modèle chargé : {roc_auc_loaded:.4f}")
# Utilisation de TreeExplainer pour XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)  # Utilisation de X_test pour générer les valeurs SHAP

# Cas binaire : shap_values est généralement un array 2D
#   - Si shap_values est une liste (cas multi-classes), prendre shap_values[1] pour la classe positive
#   - Sinon, directement shap_values

# Pour la cohérence, on vérifie si shap_values est une liste (multi-class) ou non
if isinstance(shap_values, list):
    # On récupère la composante liée à la classe 1 (positive)
    sv = shap_values[1]
else:
    sv = shap_values

# --- Waterfall Plot ---
# Utilisation de la première instance (index 0) de X_test pour le graphique SHAP
fig, ax = plt.subplots(figsize=(8, 6))
shap.waterfall_plot(shap.Explanation(values=sv[0],  # Valeurs SHAP pour la première observation de X_test
                                     base_values=explainer.expected_value,
                                     data=X_test.iloc[0, :],  # Les données associées à l'observation
                                     feature_names=X_test.columns),  # Noms des caractéristiques
                    max_display=10, show=False)

# Afficher le graphique avec Matplotlib
plt.show()