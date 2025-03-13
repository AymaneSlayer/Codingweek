# Was the dataset balanced? How was class imbalance handled, and what was the impact?
  The dataset was imbalanced, as cervical cancer cases were significantly fewer than non-cases. To address this, we applied the SMOTE technique: This method generated 
  synthetic samples for the minority class, increasing its representation in the dataset.We eventually got a database that consists of 642 samples for each case,

# Which ML model performed best? Provide performance metrics.
  Among the evaluated models (XGBoost, SVM, and CatBoost), the best-performing model was XGBoost based on key performance metrics like ROC-AUC,Accuracy,Precision,    
  Recall and F1-score. it got 0.870412,0.970930,0.750000,0.818182 and 0.782609 respectively, performing moderately better than CatBoost that got 0.943535,0.959302,   
  0.70,0.636364 and 0.666667 and SVM with 0.837380,0.959302,0.642857,0.818182 and 0.72. which makes making XGBoost the most reliable choice for cervical cancer 
  prediction.

# According to SHAP, which features most strongly influence cervical cancer predictions?
  SHAP was used to interpret the model’s predictions. The top influencing features were:
  SCHILLER
