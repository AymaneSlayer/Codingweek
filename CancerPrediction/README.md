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
Schiller-Number of sexual partners-Hormonal Contraceptives (years)-Age log-Citology, in this order.We got 
these results by A TreeExplainer that got applied on the XGBoost model to generate SHAP values using the test set.A waterfall plot is produced for a single observation (the first instance from the test set) to visually 
demonstrate how each feature contributes to pushing the model output toward a particular prediction
