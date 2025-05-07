#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Training for ECG Signal Anomaly Detection

This script trains an Isolation Forest model on latent vectors derived from an autoencoder
to detect anomalous ECG signals using an unsupervised approach.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure proper imports by determining the project root
# Find the ai_module directory in the path
file_path = Path(__file__).resolve()
module_dir = None
for parent in file_path.parents:
    if parent.name == "ai_module":
        module_dir = parent
        break
    
if module_dir is None:
    raise ImportError("Cannot find ai_module directory in path hierarchy")

# Add project root to path for reliable imports
sys.path.insert(0, str(module_dir))

# Now use absolute imports
from utils.load_latents import load_latents

# Constants and paths
MODELS_DIR = module_dir / "models"
REPORTS_DIR = module_dir / "reports"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_isolation_forest(df_train, feature_cols):
    """
    Train Isolation Forest model on the training subset of latent vectors.
    
    Args:
        df_train (pd.DataFrame): DataFrame containing training data
        feature_cols (list): List of feature column names
        
    Returns:
        sklearn.ensemble.IsolationForest: Trained model
    """
    try:
        logger.info("Training Isolation Forest model")
        
        # Extract feature matrix for training
        X_train = df_train[feature_cols].values
        
        logger.info(f"Training with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        
        # Initialize model with specified parameters
        model = IsolationForest(
            n_estimators=100,
            random_state=42,
            contamination='auto',
            n_jobs=-1  # Use all available CPU cores
        )
        
        # Train model
        model.fit(X_train)
        
        logger.info("Isolation Forest training completed")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training Isolation Forest model: {e}")
        logger.error(traceback.format_exc())
        return None


def save_model(model, output_path=None):
    """
    Save trained Isolation Forest model to disk.
    
    Args:
        model: Trained Isolation Forest model
        output_path (Path, optional): Path to save model
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if model is None:
            logger.error("Cannot save None model")
            return False
        
        # Use default path if not specified
        if output_path is None:
            output_path = MODELS_DIR / "isolation_forest.pkl"
        
        # Save model
        joblib.dump(model, output_path)
        
        logger.info(f"Model saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        logger.error(traceback.format_exc())
        return False


def generate_predictions(model, df, feature_cols):
    """
    Generate anomaly scores and predictions for all data.
    
    Args:
        model: Trained Isolation Forest model
        df (pd.DataFrame): DataFrame containing all data
        feature_cols (list): List of feature column names
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly scores and predictions
    """
    try:
        if model is None:
            logger.error("Cannot generate predictions with None model")
            return None
        
        logger.info(f"Generating predictions for {len(df)} samples")
        
        # Create a copy to avoid modifying the original
        results_df = df.copy()
        
        # Extract feature matrix
        X = results_df[feature_cols].values
        
        # Generate anomaly scores (negative of decision function)
        # Higher score = more anomalous
        anomaly_scores = -model.decision_function(X)
        
        # Generate predictions (1 = anomaly, -1 = normal in Isolation Forest)
        predictions = model.predict(X)
        
        # Convert predictions to format where 1 = anomaly, 0 = normal
        is_anomaly = np.where(predictions == -1, 1, 0)
        
        # Add results to DataFrame
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = is_anomaly
        
        # Calculate basic statistics
        n_anomalies = np.sum(is_anomaly)
        anomaly_rate = n_anomalies / len(is_anomaly)
        
        logger.info(f"Identified {n_anomalies} anomalies ({anomaly_rate:.2%} of all samples)")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        logger.error(traceback.format_exc())
        return None


def save_results(results_df, output_path=None):
    """
    Save results DataFrame to CSV.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
        output_path (Path, optional): Path to save results
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if results_df is None or results_df.empty:
            logger.error("Cannot save empty or None results")
            return False
        
        # Use default path if not specified
        if output_path is None:
            output_path = REPORTS_DIR / "anomaly_scores.csv"
        
        # Select columns to save
        # Start with required columns
        columns_to_save = ['file_name', 'split', 'anomaly_score', 'is_anomaly']
        
        # Add reconstruction_error if it exists
        if 'reconstruction_error' in results_df.columns:
            columns_to_save.append('reconstruction_error')
        
        # Filter columns
        output_df = results_df[columns_to_save]
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """
    Main function to load data, train model, generate predictions, and save results.
    """
    try:
        logger.info("Starting Isolation Forest training process")
        
        # Load latent vectors
        df_all = load_latents()
        if df_all is None or df_all.empty:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Identify feature columns
        feature_cols = [col for col in df_all.columns if col.startswith('feature_')]
        if not feature_cols:
            logger.error("No feature columns found, aborting")
            return
        
        logger.info(f"Found {len(feature_cols)} feature columns")
        
        # Split into train/test
        df_train = df_all[df_all['split'] == 'train'].copy()
        
        if len(df_train) == 0:
            logger.error("No training samples found, aborting")
            return
        
        logger.info(f"Using {len(df_train)} samples for training")
        
        # Train Isolation Forest model
        model = train_isolation_forest(df_train, feature_cols)
        if model is None:
            logger.error("Failed to train model, aborting")
            return
        
        # Save trained model
        if not save_model(model):
            logger.error("Failed to save model, continuing with prediction")
        
        # Generate predictions for all samples
        results_df = generate_predictions(model, df_all, feature_cols)
        if results_df is None:
            logger.error("Failed to generate predictions, aborting")
            return
        
        # Save results
        if not save_results(results_df):
            logger.error("Failed to save results")
        
        # Print summary statistics
        logger.info("Anomaly detection summary:")
        train_anomalies = results_df[(results_df['split'] == 'train') & (results_df['is_anomaly'] == 1)]
        test_anomalies = results_df[(results_df['split'] == 'test') & (results_df['is_anomaly'] == 1)]
        
        logger.info(f"Train set: {len(train_anomalies)}/{len(df_train)} samples ({len(train_anomalies)/len(df_train):.2%}) identified as anomalies")
        
        test_set = df_all[df_all['split'] == 'test']
        if len(test_set) > 0:
            logger.info(f"Test set: {len(test_anomalies)}/{len(test_set)} samples ({len(test_anomalies)/len(test_set):.2%}) identified as anomalies")
        
        logger.info("Isolation Forest training and evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in Isolation Forest training process: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()