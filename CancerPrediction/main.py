import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import warnings
warnings.filterwarnings("ignore")

#############################################
# PARTIE 1 : CHARGEMENT & IMPUTATION DES DONNÉES
#############################################

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
data = pd.read_csv(url, header=0)
data = data.replace('?', pd.NA)
data = data.apply(pd.to_numeric, errors='coerce')

# Suppression des colonnes avec plus de 50% de valeurs manquantes
threshold = len(data) * 0.5  
data_cleaned = data.dropna(thresh=threshold, axis=1).copy()

# Conversion de "Biopsy" en entier
data_cleaned.loc[:, "Biopsy"] = data_cleaned["Biopsy"].astype(int)

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

# Colonnes à traiter pour transformation log et winsorisation
columns_to_treat = ['Age', 'First sexual intercourse']

# Transformation log et winsorisation pour réduire l'asymétrie et limiter les outliers
for col in columns_to_treat:
    skew_val = data_cleaned[col].skew()
    data_cleaned[col + '_log'] = np.log1p(data_cleaned[col])
    winsorized_values = np.array(winsorize(data_cleaned[col + '_log'], limits=(0.05, 0.05)))
    data_cleaned[col + '_log_winsorized'] = winsorized_values
    print(f"Colonne '{col}' (skewness = {skew_val:.2f}): transformation log et winsorisation appliquées.")

# Conversion explicite en float pour les colonnes suspectées
for col in numeric_cols_suspected:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Imputation des valeurs manquantes sur l'original (hors cible)
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

#############################################
# PARTIE 2 : PRÉPARATION DES DONNÉES POUR LA MODÉLISATION
#############################################

# Séparation des features et de la cible à partir de data_cleaned
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

print("\nRépartition des classes après SMOTE:")
print(pd.Series(y_train_res).value_counts())

#############################################
# PARTIE 3 : RÉDUCTION DE LA MULTICOLLINÉARITÉ
#############################################

# On retire les colonnes originales utilisées pour les transformations
colonnes_a_exclure = ['Age', 'First sexual intercourse']
data_final = data_cleaned.drop(columns=colonnes_a_exclure)

# Calcul de la matrice de corrélation en valeurs absolues
corr_matrix = data_final.corr().abs()

# Extraction de la partie supérieure triangulaire pour éviter les doublons
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identification des colonnes à supprimer si corrélation > 0.8 avec au moins une autre variable
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

# Suppression des colonnes identifiées
data_final_reduced = data_final.drop(columns=to_drop)

print("Colonnes supprimées :", to_drop)
print("Dimensions du DataFrame réduit :", data_final_reduced.shape)

# Affichage de la matrice de corrélation via une heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data_final_reduced.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de Corrélation - Données Réduites")
plt.show()

#############################################
# PARTIE 4 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
#############################################

print("Aperçu du dataset (5 premières lignes) :")
print(data_final_reduced.head())

print("\nDimensions (lignes, colonnes) :", data_final_reduced.shape)
print("\nInformations sur les colonnes :")
data_final_reduced.info()

print("\nStatistiques descriptives (variables numériques) :")
print(data_final_reduced.describe())

print("\nValeurs manquantes par colonne après imputation :")
print(data_final_reduced.isnull().sum())

# Vérification de la cible
target_col = 'Biopsy'
if target_col in data_final_reduced.columns:
    print("\nRépartition de la cible :")
    print(data_final_reduced[target_col].value_counts())
    print(data_final_reduced[target_col].value_counts(normalize=True) * 100)
else:
    print(f"\nATTENTION : la colonne '{target_col}' n'existe pas.")

#############################################
# PARTIE 5 : OPTIMISATION DE LA MÉMOIRE
#############################################

def optimize_memory(df):
    """Optimise l'utilisation mémoire en ajustant les types de données."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMémoire utilisée avant optimisation: {start_mem:.2f} Mo")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:  # On traite uniquement les colonnes numériques
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Mémoire utilisée après optimisation: {end_mem:.2f} Mo")
    print(f"Réduction de {(100 * (start_mem - end_mem) / start_mem):.1f}%\n")
    return df

# Application de l'optimisation sur le DataFrame réduit
data_final_optimized = optimize_memory(data_final_reduced.copy())

#############################################
# PARTIE 6 : MODÉLISATION
#############################################

if target_col not in data_final_optimized.columns:
    print("Impossible de procéder à la modélisation : cible introuvable.")
    import sys
    sys.exit()

# Séparation des features et de la cible pour la modélisation
X_model = data_final_optimized.drop(columns=[target_col])
y_model = data_final_optimized[target_col].astype(int)

# On ne garde que les variables numériques
X_model = X_model.select_dtypes(include=[np.number])

# Division en ensembles d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.2, stratify=y_model, random_state=42
)

model_results = []

# ------------------- XGBoost -------------------
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

# ------------------- CatBoost -------------------
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

# ------------------- SVM avec Pipeline -------------------
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

#############################################
# PARTIE 7 : ÉVALUATION DES MODÈLES
#############################################

models = {
    'XGBoost': xgb_best,
    'CatBoost': cat_best,
    'SVM': svm_best
}

results_metrics = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred
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

#############################################
# PARTIE 8 : SHAP EXPLAINABILITY
#############################################

# Sélection du meilleur modèle selon le F1-score
best_model_name = results_metrics_df.sort_values('F1-score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nMeilleur modèle selon F1-score: {best_model_name}")

print("Calcul des valeurs SHAP pour le meilleur modèle...")

# En fonction du type de modèle, on choisit l'explainer approprié
if best_model_name in ['XGBoost', 'CatBoost']:
    # Utilisation de TreeExplainer pour les modèles basés sur des arbres
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
else:
    # Pour le modèle SVM (pipeline), on utilise KernelExplainer
    # On sélectionne un sous-échantillon de X_train comme background pour réduire le temps de calcul
    background = shap.sample(X_train, 100, random_state=42)
    # Définition d'une fonction de prédiction qui retourne la probabilité de la classe positive
    predict_fn = lambda x: best_model.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test)

# Visualisation de l'importance des features via un summary plot
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)


# Sélection du meilleur modèle selon le F1-score (RandomForest dans ce cas)
best_model_name = results_metrics_df.sort_values('F1-score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nMeilleur modèle selon F1-score: {best_model_name}")
