# Cervical Cancer Risk Prediction

## Project Overview

This repository contains the code and resources for predicting cervical cancer risk using machine learning models. The project focuses on data preprocessing, handling class imbalance, feature engineering, model training, evaluation, and explainability through SHAP.

## Repository Structure

Dockerfile: Defines the environment setup for running the application in a containerized environment.

notebooks: Jupyter notebooks containing exploratory data analysis (EDA), feature engineering, and initial model training.

app.py: Streamlit/Flask-based web application for user interaction and predictions.

main.py: The main script for data processing, model training, and evaluation.

requirements.txt: List of dependencies required for running the project.

## Key Questions & Insights

### Was the dataset balanced? How was class imbalance handled, and what was the impact?
No, the dataset was highly imbalanced, with only 6.41% of positive cases (Biopsy = 1). To address this, SMOTE was applied to balance the classes, This method generated synthetic samples for the minority class, increasing its representation in the dataset.We eventually got a database that consists of 642 samples for each case, the dataset was evenly distributed (50% positive, 50% negative).

### Which ML model performed best? Provide performance metrics.
Among the evaluated models (XGBoost, SVM, and CatBoost), the best-performing model was XGBoost based on key performance metrics like ROC-AUC,Accuracy,Precision,    Recall and F1-score. it got 0.870412,0.970930,0.750000,0.818182 and 0.782609 respectively, performing moderately better than CatBoost that got 0.943535,0.959302,   82 and 0.72. which makes making XGBoost the most reliable choice for cervical cancer 
prediction.

# According to SHAP, which features most strongly influence cervical cancer predictions?
  SHAP was used to interpret the model’s predictions. The top influencing features were:
  SCHILLER
