"""
Sample training script for a simple scikit-learn model.
"""

import argparse
import logging
import os
import pickle
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data.
    
    Args:
        data_path: Path to the training data
        
    Returns:
        Tuple of features and labels
    """
    # For demo purposes, create synthetic data
    # Replace this with your actual data loading logic
    logger.info("Loading training data...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 classes
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='target')
    
    logger.info(f"Loaded data with shape: {X_df.shape}")
    return X_df, y_series

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        Trained model
    """
    logger.info("Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return model

def save_model(model: RandomForestClassifier, output_path: str):
    """
    Save the trained model.
    
    Args:
        model: Trained model
        output_path: Path to save the model
    """
    logger.info(f"Saving model to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Model saved successfully")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train a Random Forest model')
    parser.add_argument('--data-path', type=str, default='data/train.csv',
                      help='Path to training data')
    parser.add_argument('--model-output-path', type=str, default='model/model.pkl',
                      help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.data_path)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model, args.model_output_path)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()