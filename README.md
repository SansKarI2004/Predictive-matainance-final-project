# Predictive Maintenance System

A machine learning-based predictive maintenance system for smart manufacturing using the AI4I 2020 dataset. The system predicts machine failures based on operational data.

## Features

- Load and preprocess the AI4I 2020 dataset
- Train multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- Apply SMOTE to handle imbalanced classes
- Use Stratified K-Fold cross-validation and GridSearchCV for hyperparameter tuning
- Evaluate models using:
  - F1 Score
  - ROC AUC Score
  - Confusion Matrix
  - Precision and Recall
- Plot ROC curves for all models
- Streamlit user interface for real-time predictions

## How to Use

1. Run the model training script:
   