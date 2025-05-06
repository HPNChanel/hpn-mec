# TODO: Implement this module

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
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATENTS_DIR = PROJECT_ROOT / "data/processed/latents"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_latent_vectors():
    """
    Load all latent vector files from the latents directory.
    
    Returns:
        tuple: (feature_matrix, file_names)
    """
    try:
        # Get all .npy files in the latents directory
        latent_files = list(LATENTS_DIR.glob("*.npy"))
        
        if not latent_files:
            logger.error(f"No latent vector files found in {LATENTS_DIR}")
            return None, None
        
        logger.info(f"Found {len(latent_files)} latent vector files")
        
        # Initialize lists to store data
        features = []
        file_names = []
        
        # Load each latent vector file
        for file_path in latent_files:
            try:
                # Load latent vector
                latent = np.load(file_path)
                
                # Ensure vector is 1D (flatten if needed)
                latent = latent.flatten()
                
                # Add to lists
                features.append(latent)
                file_names.append(file_path.stem)
                
            except Exception as e:
                logger.warning(f"Error loading latent vector {file_path.name}: {e}")
                continue
        
        # Convert list of features to 2D array
        feature_matrix = np.vstack(features)
        
        logger.info(f"Loaded feature matrix with shape: {feature_matrix.shape}")
        
        return feature_matrix, file_names
        
    except Exception as e:
        logger.error(f"Error loading latent vectors: {e}")
        return None, None


def load_reconstruction_errors():
    """
    Load reconstruction errors from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with reconstruction errors
    """
    try:
        # Define path to reconstruction errors CSV
        recon_errors_path = REPORTS_DIR / "reconstruction_scores.csv"
        
        # Check if file exists
        if not recon_errors_path.exists():
            logger.warning(f"Reconstruction errors file not found: {recon_errors_path}")
            return None
        
        # Load CSV into DataFrame
        df = pd.read_csv(recon_errors_path)
        
        logger.info(f"Loaded reconstruction errors for {len(df)} samples")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading reconstruction errors: {e}")
        return None


def train_isolation_forest(X):
    """
    Train Isolation Forest model on latent vectors.
    
    Args:
        X (numpy.ndarray): Feature matrix of latent vectors
        
    Returns:
        sklearn.ensemble.IsolationForest: Trained model
    """
    try:
        logger.info("Training Isolation Forest model")
        
        # Initialize model with specified parameters
        model = IsolationForest(
            n_estimators=100,
            random_state=42,
            contamination='auto',
            n_jobs=-1  # Use all available CPU cores
        )
        
        # Train model
        model.fit(X)
        
        logger.info("Isolation Forest training completed")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training Isolation Forest model: {e}")
        return None


def save_model(model):
    """
    Save trained Isolation Forest model to disk.
    
    Args:
        model: Trained Isolation Forest model
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if model is None:
            logger.error("Cannot save None model")
            return False
        
        # Define save path
        model_path = MODELS_DIR / "isolation_forest.pkl"
        
        # Save model
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def generate_anomaly_scores(model, X, file_names):
    """
    Generate anomaly scores and predictions from the trained model.
    
    Args:
        model: Trained Isolation Forest model
        X (numpy.ndarray): Feature matrix of latent vectors
        file_names (list): List of file names corresponding to each sample
        
    Returns:
        pd.DataFrame: DataFrame with anomaly scores and predictions
    """
    try:
        if model is None or X is None or file_names is None:
            logger.error("Cannot generate scores with None inputs")
            return None
        
        logger.info("Generating anomaly scores")
        
        # Generate raw anomaly scores (negative of decision function)
        # Higher score = more anomalous
        anomaly_scores = -model.decision_function(X)
        
        # Generate binary predictions (1 = anomaly, -1 = normal)
        predictions = model.predict(X)
        
        # Convert predictions to 0/1 format (1 = anomaly, 0 = normal)
        # In Isolation Forest, -1 = anomaly, 1 = normal, so we need to convert
        is_anomaly = np.where(predictions == -1, 1, 0)
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'file_name': file_names,
            'anomaly_score': anomaly_scores,
            'is_anomaly': is_anomaly
        })
        
        logger.info(f"Generated anomaly scores for {len(results_df)} samples")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error generating anomaly scores: {e}")
        return None


def merge_with_reconstruction_errors(anomaly_df, recon_errors_df):
    """
    Merge anomaly scores with reconstruction errors.
    
    Args:
        anomaly_df (pd.DataFrame): DataFrame with anomaly scores
        recon_errors_df (pd.DataFrame): DataFrame with reconstruction errors
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        if anomaly_df is None:
            logger.error("Cannot merge with None anomaly DataFrame")
            return None
        
        if recon_errors_df is None:
            logger.warning("No reconstruction errors to merge, returning anomaly scores only")
            return anomaly_df
        
        logger.info("Merging anomaly scores with reconstruction errors")
        
        # Prepare reconstruction errors DataFrame
        # Extract filename without extension from 'filename' column if needed
        if 'filename' in recon_errors_df.columns:
            recon_errors_df['file_name'] = recon_errors_df['filename'].apply(
                lambda x: Path(x).stem
            )
        
        # Merge DataFrames on file_name
        merged_df = pd.merge(
            anomaly_df,
            recon_errors_df[['file_name', 'reconstruction_error']],
            on='file_name',
            how='left'
        )
        
        logger.info(f"Merged DataFrame has {len(merged_df)} rows and {merged_df.shape[1]} columns")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging with reconstruction errors: {e}")
        return anomaly_df


def save_results(results_df):
    """
    Save results DataFrame to CSV.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if results_df is None or results_df.empty:
            logger.error("Cannot save empty or None results")
            return False
        
        # Define save path
        results_path = REPORTS_DIR / "anomaly_scores.csv"
        
        # Save to CSV
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def main():
    """
    Main function to load data, train model, and generate results.
    """
    try:
        logger.info("Starting Isolation Forest training process")
        
        # Load latent vectors
        X, file_names = load_latent_vectors()
        if X is None or file_names is None:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Load reconstruction errors
        recon_errors_df = load_reconstruction_errors()
        
        # Train Isolation Forest model
        model = train_isolation_forest(X)
        if model is None:
            logger.error("Failed to train model, aborting")
            return
        
        # Save trained model
        save_model(model)
        
        # Generate anomaly scores
        results_df = generate_anomaly_scores(model, X, file_names)
        if results_df is None:
            logger.error("Failed to generate anomaly scores, aborting")
            return
        
        # Merge with reconstruction errors
        merged_df = merge_with_reconstruction_errors(results_df, recon_errors_df)
        
        # Save results
        save_results(merged_df)
        
        logger.info("Isolation Forest training and evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in Isolation Forest training process: {e}")


if __name__ == "__main__":
    main()