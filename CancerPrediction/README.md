# Cervical Cancer Risk Prediction

## Project Overview

This repository contains the code and resources for predicting cervical cancer risk using machine learning models. The project focuses on data preprocessing, handling class imbalance, feature engineering, model training, evaluation, and explainability through SHAP.

## Repository Structure

Dockerfile: Defines the environment setup for running the application in a containerized environment.

notebooks: Jupyter notebooks containing exploratory data analysis (EDA), feature engineering, and initial model training.

app.py: Streamlit/Flask-based web application for user interaction and predictions.

main.py: The main script for data processing, model training, and evaluation.

requirements.txt: List of dependencies required for running the project.

## Data Preprocessing

### Loading and Cleaning:
The dataset is loaded from an online source. Any placeholder for missing values (e.g., “?”) is replaced with proper missing values (NA). Additionally, columns with more than 50% missing data are dropped to ensure data quality.

### Type Conversion and Transformation:
Columns that are assumed to be numeric (such as “Age” and “First sexual intercourse”) are explicitly converted to numeric types. For variables with skewed distributions, a logarithmic transformation is applied, followed by winsorization to limit the impact of extreme values.

### Imputation:
Missing values are imputed using the median for numeric columns and the mode for categorical ones. This step ensures the dataset is complete for subsequent analysis.

## Key Questions & Insights

### Was the dataset balanced? How was class imbalance handled, and what was the impact?
No, the dataset was highly imbalanced, with only 6.41% of positive cases (Biopsy = 1). To address this, SMOTE was applied to balance the classes, This method generated synthetic samples for the minority class, increasing its representation in the dataset.We eventually got a database that consists of 642 samples for each case, the dataset was evenly distributed (50% positive, 50% negative).

### Which ML model performed best? Provide performance metrics.
We trained and optimized three models—XGBoost, CatBoost, and SVM (with imputation and scaling in a pipeline) and we evaluated them on key metrics such as ROC-AUC, Accuracy, Precision, Recall, and F1-score. XGBoost achieved nearly perfect scores (ROC-AUC ~0.9965, Accuracy ~0.9883, Precision ~0.9771, Recall 1.0, and F1-score ~0.9884), while CatBoost showed very similar performance but slightly lower metrics, and SVM performed a bit lower in terms of Accuracy and F1-score compared to XGBoost. Based on these results, XGBoost was determined to be the best-performing model overall.

### According to SHAP, which features most strongly influence cervical cancer predictions?
SHAP was used to interpret the model’s predictions. The top influencing features were:
Schiller-Number of sexual partners-Hormonal Contraceptives (years)-Age log-Citology, in this order.We got these results by A TreeExplainer that got applied on the XGBoost model to generate SHAP values using the test set.A waterfall plot is produced for a single observation (the first instance from the test set) to visually demonstrate how each feature contributes to pushing the model output toward a particular prediction

## Memory Optimization

The program includes a function that goes through each column of your DataFrame and converts the data types to smaller, more efficient types. For example, it changes larger integer and float types to smaller ones when possible, and turns text columns into categorical data. This helps reduce the memory usage of your dataset (52%), making the program run faster and handle larger amounts of data more easily.

## SHAP Interpretability

The program uses SHAP to show how much each feature affects the predictions of the XGBoost model. A TreeExplainer calculates SHAP values, and a waterfall plot displays these values for one example, showing the contribution of each feature. This makes it easier to understand why the model makes certain predictions.

## Streamlit Interface

We chose Streamlit over Flask because it’s much easier and faster to set up, with built-in components for interactive interfaces. Our app uses a custom background and CSS to style the sidebar and input fields. We also use caching to load our XGBoost model quickly. Additionally, we integrated SHAP to explain the model’s predictions by generating a waterfall plot that shows each feature’s impact on the risk score. This combination of tools makes our cervical cancer risk prediction tool both efficient and user-friendly.


##  Insights from Prompt Engineering

We will declare the prompts used in the Data Preprocessing task in our project

### Prompt 1: Missing Values 
"Écris un code simple pour vérifier s'il y a des valeurs manquantes dans le dataset et suggérer comment les traiter"
```bash
#Vérification des valeurs manquantes
print("Valeurs manquantes avant imputation:")
print(data.isnull().sum())
#Imputation : médiane pour les numériques et mode pour les catégorielles
for col in data.columns:
    if data[col].dtype in [np.float64, np.int64]:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        mode_val = data[col].mode(dropna=True)
        data[col].fillna(mode_val[0] if not mode_val.empty else "Inconnu", inplace=True)
#Vérification après imputation
print("\nValeurs manquantes après imputation:")
print(data.isnull().sum())
```
### Prompt 2: Outliers
"Écris un code simple pour détecter et gérer les outliers sur les colonnes avec une distribution très asymétrique, en appliquant une transformation logarithmique et une winsorisation."
```bash
#Pour les colonnes 'Age' et 'First sexual intercourse'
colonnes_a_traiter = ['Age', 'First sexual intercourse']
for col in colonnes_a_traiter:
    skew_val = data_cleaned[col].skew()
    #Transformation logarithmique
    data_cleaned[col + '_log'] = np.log1p(data_cleaned[col])
    #Winsorisation pour limiter les valeurs extrêmes
    data_cleaned[col + '_log_winsorized'] = winsorize(data_cleaned[col + '_log'], limits=(0.05, 0.05))
    print(f"Colonne '{col}' (skewness = {skew_val:.2f}): transformation log et winsorisation appliquées.")
```

### Prompt 3: Class Imbalance
"Écris un code simple pour appliquer SMOTE pour équilibrer le dataset (cas d'un déséquilibre significatif, par exemple 85% 'No risk' et 15% 'At risk')."
```bash
#Vérification de la répartition de la cible
print("Répartition des classes avant SMOTE:")
print(y.value_counts())
#Application de SMOTE pour équilibrer les classes
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)
#Affichage de la nouvelle répartition
print("\nRépartition des classes après SMOTE:")
print(pd.Series(y_train_res).value_counts())
```
### Prompt 4: Corrélation
"écris moi un code qui affiche la matrice de corrélation et supprime les features trop corrélées (coefficient de coorélation > 0.8)."
```bash
#Ajout de la colonne 'Biopsy' au DataFrame final
data_final = nouveau_data.copy()
#Retrait des colonnes non nécessaires
colonnes_a_exclure = ['Age', 'First sexual intercourse']
data_final = data_final.drop(columns=colonnes_a_exclure)
#Calcul de la matrice de corrélation en valeurs absolues
corr_matrix = data_final.corr().abs()
#Extraction de la partie supérieure triangulaire pour éviter les doublons
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#Identification des colonnes à supprimer si corrélation > 0.8
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
#Suppression des colonnes identifiées
data_final_reduced = data_final.drop(columns=to_drop)
#Réintégration de la colonne cible
data_final_reduced['Biopsy'] = data_final['Biopsy']
#Affichage des colonnes supprimées et des dimensions du DataFrame réduit
print("Colonnes supprimées :", to_drop)
print("Dimensions du DataFrame réduit :", data_final_reduced.shape)
plt.figure(figsize=(12, 10))
sns.heatmap(data_final_reduced.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de Corrélation - Données Réduites")
plt.show()
#Affichage des informations sur le DataFrame réduit
print(data_final_reduced.columns)
print(data_final_reduced['Biopsy'].value_counts())
print("Aperçu du dataset (5 premières lignes) :")
print(data_final_reduced.head())
print("\nDimensions (lignes, colonnes) :", data_final_reduced.shape)
print("\nInformations sur les colonnes :")
data_final_reduced.info()
print("\nStatistiques descriptives (variables numériques) :")
print(data_final_reduced.describe())
print("\nValeurs manquantes par colonne après imputation :")
print(data_final_reduced.isnull().sum())
#Vérification de la présence de la cible
target_col = 'Biopsy'
if target_col in data_final_reduced.columns:
    print("\nRépartition de la cible :")
    print(data_final_reduced[target_col].value_counts())
    print(data_final_reduced[target_col].value_counts(normalize=True) * 100)
else:
    print(f"\nATTENTION : la colonne '{target_col}' n'existe pas.")
```
Some insights concerning the prompts:

  The prompts proved highly effective in breaking down complex preprocessing tasks into clear, manageable steps. They guided the process by clearly addressing 
  missing values, outlier management, class imbalance, and correlation analysis, which helped us alot especially when we had to choose a method between a bunch of 
  them (Oversampling (SMOTE),Undersampling,Class-weighting technique).
  
  This approach made the project more reproducible.However, there is little to no room for improvement except exploring other methods that could make a better 
  memory optimisation, but we chose the simpler methods over efficiency so we can have a better understanding.

