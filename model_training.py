import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, 
                             precision_score, recall_score, roc_curve, auc,
                             classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_data(file_path='ai4i2020.csv'):
    """
    Load dataset from CSV file.
    
    Args:
        file_path (str): Path to the dataset file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Try to load from URL if file not found locally
        if not os.path.exists(file_path):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
            print(f"File {file_path} not found. Downloading from {url}...")
            df = pd.read_csv(url)
            # Save the file locally for future use
            df.to_csv(file_path, index=False)
            print(f"Dataset saved to {file_path}")
        else:
            df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        tuple: Preprocessed features and target, column transformer, feature names
    """
    print("Starting data preprocessing...")
    
    # Create target variable 'Machine_Failure' based on failure types
    failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df['Machine_Failure'] = df[failure_columns].max(axis=1)
    
    # Drop unnecessary columns
    columns_to_drop = ['UDI', 'Product ID'] + failure_columns
    df_processed = df.drop(columns=columns_to_drop)
    
    # Split features and target
    X = df_processed.drop('Machine_Failure', axis=1)
    y = df_processed['Machine_Failure']
    
    # Identify categorical and numerical columns
    categorical_cols = ['Type']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Get feature names after transformation
    cat_encoder = OneHotEncoder(drop='first')
    cat_encoder.fit(X[categorical_cols])
    encoded_features = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(encoded_features)
    
    print(f"Preprocessing complete. Features: {len(feature_names)}, Samples: {len(y)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, preprocessor, feature_names

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        tuple: Resampled features and target
    """
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

def build_models():
    """
    Build machine learning models with parameter grids.
    
    Returns:
        dict: Dictionary of models and their parameter grids
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        }
    }
    
    return models

def train_evaluate_models(X, y, preprocessor, models):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X: Features
        y: Target
        preprocessor: Column transformer for preprocessing
        models: Dictionary of models and their parameter grids
    
    Returns:
        tuple: Best model pipeline, evaluation results, test data
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Apply SMOTE only on the training data
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # Initialize variables to track best model
    best_model = None
    best_score = 0
    model_results = {}
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Loop through models
    for model_name, model_info in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create pipeline with preprocessing and classifier
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_info['model'])
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            model_info['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_resampled, y_resampled)
        
        # Get best model from grid search
        best_pipeline = grid_search.best_estimator_
        
        # Predict on test set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        model_results[model_name] = {
            'pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'y_prob': y_prob
        }
        
        # Print results
        print(f"{model_name} Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        # Update best model if current model is better
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model_name
    
    print(f"\nBest Model: {best_model} with ROC AUC of {best_score:.4f}")
    
    return model_results[best_model]['pipeline'], model_results, (X_test, y_test)

def plot_roc_curves(model_results, X_test, y_test):
    """
    Plot ROC curves for all models.
    
    Args:
        model_results: Dictionary of model evaluation results
        X_test: Test features
        y_test: Test target
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
        roc_auc = results['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.close()
    print("ROC curves saved to 'roc_curves.png'")

def plot_confusion_matrices(model_results, y_test):
    """
    Plot confusion matrices for all models.
    
    Args:
        model_results: Dictionary of model evaluation results
        y_test: Test target
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models*4, 4))
    
    for i, (model_name, results) in enumerate(model_results.items()):
        cm = results['confusion_matrix']
        
        if n_models > 1:
            ax = axes[i]
        else:
            ax = axes
            
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    print("Confusion matrices saved to 'confusion_matrices.png'")

def save_model(model, preprocessor, feature_names):
    """
    Save the best model and preprocessor.
    
    Args:
        model: Best model pipeline
        preprocessor: Column transformer for preprocessing
        feature_names: List of feature names
    """
    # Save the best model
    joblib.dump(model, 'best_model.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Model and feature names saved successfully")

def main():
    """Main function to orchestrate the model training process."""
    print("Starting predictive maintenance model training...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    X, y, preprocessor, feature_names = preprocess_data(df)
    
    # Build models
    models = build_models()
    
    # Train and evaluate models
    best_model, model_results, (X_test, y_test) = train_evaluate_models(X, y, preprocessor, models)
    
    # Plot ROC curves
    plot_roc_curves(model_results, X_test, y_test)
    
    # Plot confusion matrices
    plot_confusion_matrices(model_results, y_test)
    
    # Save model and preprocessor
    save_model(best_model, preprocessor, feature_names)
    
    print("Model training complete!")

if __name__ == "__main__":
    main()
