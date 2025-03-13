import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pour la modélisation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Modèles
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Pipeline et normalisation pour SVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

#####################################################
# PARTIE 1 : CHARGEMENT & IMPUTATION DES DONNEES
#####################################################

DATA_PATH = 'C:/Users/MSI/Desktop/data.csv'
df = pd.read_csv(DATA_PATH)

# Remplacer "?" par NaN
df.replace("?", np.nan, inplace=True)

# Liste des colonnes supposées numériques
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

# Conversion explicite en float
for col in numeric_cols_suspected:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Imputation des valeurs manquantes
for col in df.columns:
    if col == "Biopsy":
        continue
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna("Inconnu", inplace=True)

#####################################################
# PARTIE 2 : ANALYSE EXPLORATOIRE DES DONNEES (EDA)
#####################################################

print("Aperçu du dataset (5 premières lignes) :")
print(df.head())

print("\nDimensions (lignes, colonnes) :", df.shape)
print("\nInformations sur les colonnes :")
df.info()

print("\nStatistiques descriptives (variables numériques) :")
print(df.describe())

print("\nValeurs manquantes par colonne après imputation :")
print(df.isnull().sum())

# Vérification de la cible
target_col = 'Biopsy'
if target_col in df.columns:
    print("\nRépartition de la cible :")
    print(df[target_col].value_counts())
    print(df[target_col].value_counts(normalize=True)*100)
else:
    print(f"\nATTENTION : la colonne '{target_col}' n'existe pas.")

#####################################################
# PARTIE 3 : MODELES DE MACHINE LEARNING
#####################################################

if target_col not in df.columns:
    print("Impossible de procéder à la modélisation : cible introuvable.")
    import sys
    sys.exit()

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model_results = []

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'class_weight': ['balanced']}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_f1 = f1_score(y_test, rf_preds)
model_results.append({'Model': 'RandomForest', 'F1-score': rf_f1})

# XGBoost
xgb_params = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 6], 'scale_pos_weight': [1, 3]}
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_preds)
model_results.append({'Model': 'XGBoost', 'F1-score': xgb_f1})

# SVM
svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])
svm_params = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf'], 'svc__class_weight': ['balanced']}
svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring='f1')
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_
svm_preds = svm_best.predict(X_test)
svm_f1 = f1_score(y_test, svm_preds)
model_results.append({'Model': 'SVM', 'F1-score': svm_f1})

# Comparaison des résultats
results_df = pd.DataFrame(model_results).sort_values(by='F1-score', ascending=False)
print("\nRésultats comparatifs :")
print(results_df)

best_model = results_df.iloc[0]['Model']
print(f"\n>>> Le meilleur modèle est : {best_model} <<<")

# Prédire si un patient a un cancer ou non
best_estimator = {'RandomForest': rf_best, 'XGBoost': xgb_best, 'SVM': svm_best}[best_model]
patient_pred = best_estimator.predict(X_test)

for i, pred in enumerate(patient_pred[:859]):  # Afficher les 10 premières prédictions
    diagnostic = "POSITIF au cancer" if pred == 1 else "NÉGATIF au cancer"
    print(f"Patient {i+1}: {diagnostic}")

