import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import mlflow

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads raw CSV and fixes structural data type issues."""
    logging.info(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Fix hidden blank spaces in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def preprocess_features(df: pd.DataFrame):
    """Applies encoding and returns feature matrix (X), target (y), and feature names."""
    logging.info("Preprocessing features and applying encoders...")
    
    # Drop useless ID and map target
    df_clean = df.drop(columns=['customerID'])
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate categorical sets
    binary_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 2 and col != 'Churn']
    multi_cols = [col for col in df_clean.columns if df_clean[col].nunique() > 2 and df_clean[col].dtype == 'object']
    
    # Label Encode binary columns
    for col in binary_cols:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])
        
    # One-Hot Encode multi-class columns
    df_clean = pd.get_dummies(df_clean, columns=multi_cols, drop_first=True)
    
    X = df_clean.drop(columns=['Churn']).astype(float)
    y = df_clean['Churn']
    
    return X, y, X.columns.tolist()

def train_and_evaluate():
    """Main execution pipeline for training, tracking, and saving the model."""
    # Ensure models output directory exists
    os.makedirs('../models', exist_ok=True)
    
    # 1. Load & Preprocess
    data_path = '../WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = load_and_clean_data(data_path)
    X, y, feature_names = preprocess_features(df)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Data split complete. Training on {X_train.shape[0]} samples.")

    # 3. Calculate Class Weights for Imbalance
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 4. Set up MLflow Tracking
    mlflow.set_experiment("Production_Churn_Pipeline")
    
    with mlflow.start_run(run_name="Optimized_XGBoost_Pipeline"):
        params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 4,
            'scale_pos_weight': scale_weight,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        logging.info("Training optimized XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        logging.info(f"Model Evaluation Accuracy: {acc:.4f}")
        print("\n" + classification_report(y_test, y_pred))
        
        # 5. Save Model and Feature Columns for UI persistence
        model_artifact = {
            'model': model,
            'features': feature_names
        }
        joblib.dump(model_artifact, '../models/xgboost_churn_model.pkl')
        logging.info("Pipeline successful! Model artifact saved to models/xgboost_churn_model.pkl")

if __name__ == "__main__":
    # Executing this file runs the pipeline
    train_and_evaluate()