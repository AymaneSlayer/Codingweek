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

DATA_PATH = 'C:/Users/eDH/Desktop/codingweek/risk_factors_cervical_cancer.csv'
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
    # Ajoutez toutes les colonnes que vous jugez "numériques"
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
# PARTIE 2 : ANALYSE EXPLORATOIRE DES DONNEES (EDA) - SANS CORRÉLATION
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

# Vérifier l'existence de la cible
if target_col not in df.columns:
    print("Impossible de procéder à la modélisation : cible introuvable.")
    import sys
    sys.exit()

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Conserver uniquement les colonnes numériques
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nDimensions X_train:", X_train.shape, "y_train:", y_train.shape)
print("Dimensions X_test:", X_test.shape, "y_test:", y_test.shape)

model_results = []

#==========================
# A) Random Forest
#==========================
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'class_weight': ['balanced']
}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)[:, 1]

rf_roc = roc_auc_score(y_test, rf_proba)
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

model_results.append({
    'Model': 'RandomForest',
    'ROC-AUC': rf_roc,
    'Accuracy': rf_acc,
    'Precision': rf_prec,
    'Recall': rf_rec,
    'F1-score': rf_f1
})

#==========================
# B) XGBoost
#==========================
xgb_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'scale_pos_weight': [1, 3]
}
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1')
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(X_test)
xgb_proba = xgb_best.predict_proba(X_test)[:, 1]

xgb_roc = roc_auc_score(y_test, xgb_proba)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_prec = precision_score(y_test, xgb_preds)
xgb_rec = recall_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)

model_results.append({
    'Model': 'XGBoost',
    'ROC-AUC': xgb_roc,
    'Accuracy': xgb_acc,
    'Precision': xgb_prec,
    'Recall': xgb_rec,
    'F1-score': xgb_f1
})

#==========================
# C) SVM
#==========================
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=42))
])
svm_params = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__class_weight': ['balanced']
}
svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring='f1')
svm_grid.fit(X_train, y_train)

svm_best = svm_grid.best_estimator_
svm_preds = svm_best.predict(X_test)
svm_proba = svm_best.predict_proba(X_test)[:, 1]

svm_roc = roc_auc_score(y_test, svm_proba)
svm_acc = accuracy_score(y_test, svm_preds)
svm_prec = precision_score(y_test, svm_preds)
svm_rec = recall_score(y_test, svm_preds)
svm_f1 = f1_score(y_test, svm_preds)

model_results.append({
    'Model': 'SVM',
    'ROC-AUC': svm_roc,
    'Accuracy': svm_acc,
    'Precision': svm_prec,
    'Recall': svm_rec,
    'F1-score': svm_f1
})

# Comparaison finale
results_df = pd.DataFrame(model_results)
results_df.sort_values(by='F1-score', ascending=False, inplace=True)

print("\nRésultats comparatifs :")
print(results_df)

best_model_name = results_df.iloc[0]['Model']
best_f1 = results_df.iloc[0]['F1-score']
print(f"\n>>> Le meilleur modèle semble être : {best_model_name} avec un F1-score = {best_f1:.3f} <<<")