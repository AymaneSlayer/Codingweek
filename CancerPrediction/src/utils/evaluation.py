import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Pour la modélisation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Modèles
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier  
from sklearn.svm import SVC

# Pipeline et normalisation pour SVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

data  = pd.read_csv('CancerPrediction/data/processed_data.csv')
target_col = 'Biopsy'
if target_col in data.columns:
    print("\nRépartition de la cible :")
    print(data[target_col].value_counts())
    print(data[target_col].value_counts(normalize=True) * 100)
else:
    print(f"\nATTENTION : la colonne '{target_col}' n'existe pas.")
if target_col not in data.columns:
    print("Impossible de procéder à la modélisation : cible introuvable.")
    import sys
    sys.exit()

# Séparation des features et de la cible pour la modélisation
X = data.drop(columns=[target_col])
y = data[target_col].astype(int)

# On ne garde que les variables numériques
X = X.select_dtypes(include=[np.number])

# Division en ensembles d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model_results = []

# -------------------
# Random Forest
# -------------------
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
rf_f1 = f1_score(y_test, rf_preds)
model_results.append({'Model': 'RandomForest', 'F1-score': rf_f1})

# -------------------
# XGBoost
# -------------------
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
xgb_f1 = f1_score(y_test, xgb_preds)
model_results.append({'Model': 'XGBoost', 'F1-score': xgb_f1})

# -------------------
# CatBoost Classifier
# -------------------
cat_params = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}
cat_model = CatBoostClassifier(random_state=42, verbose=0)
cat_grid = GridSearchCV(cat_model, cat_params, cv=5, scoring='f1')
cat_grid.fit(X_train, y_train)
cat_best = cat_grid.best_estimator_
cat_preds = cat_best.predict(X_test)
cat_f1 = f1_score(y_test, cat_preds)
model_results.append({'Model': 'CatBoost', 'F1-score': cat_f1})

# -------------------
# SVM avec Pipeline (imputation + scaling + SVC)
# -------------------
svm_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
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
svm_f1 = f1_score(y_test, svm_preds)
model_results.append({'Model': 'SVM', 'F1-score': svm_f1})
models = {
    'RandomForest': rf_best,
    'XGBoost': xgb_best,
    'CatBoost': cat_best,
    'SVM': svm_best
}

results_metrics = []

for name, model in models.items():
    # Prédictions de classes
    y_pred = model.predict(X_test)
    # Calcul des probabilités pour le ROC-AUC (selon méthode disponible)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # solution de secours
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    
    results_metrics.append({
        "Model": name,
        "ROC-AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

results_metrics_df = pd.DataFrame(results_metrics)
print("\nÉvaluation des modèles :")
print(results_metrics_df)