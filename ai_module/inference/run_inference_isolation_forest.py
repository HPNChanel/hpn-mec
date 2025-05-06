#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Inference Script for ECG Anomaly Detection

This script loads a trained Isolation Forest model and runs inference on latent
vectors to detect anomalous ECG signals. Results are saved to CSV format.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

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
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "inference_isolation_result.csv"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """
    Load the trained Isolation Forest model.
    
    Returns:
        model: Trained Isolation Forest model or None if loading fails
    """
    try:
        # Define model path
        model_path = MODELS_DIR / "isolation_forest.pkl"
        
        # Check if model exists
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Load model
        model = joblib.load(model_path)
        
        logger.info(f"Model loaded successfully from {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_latent_vectors():
    """
    Load all latent vector files from the latents directory.
    
    Returns:
        tuple: (feature_matrix, file_names) or (None, None) if loading fails
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
        
        if not features:
            logger.error("No valid latent vectors could be loaded")
            return None, None
        
        # Convert list of features to 2D array
        feature_matrix = np.vstack(features)
        
        logger.info(f"Loaded feature matrix with shape: {feature_matrix.shape}")
        
        return feature_matrix, file_names
        
    except Exception as e:
        logger.error(f"Error loading latent vectors: {e}")
        return None, None


def run_inference(model, X, file_names):
    """
    Run inference using the Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest model
        X (numpy.ndarray): Feature matrix of latent vectors
        file_names (list): List of file names corresponding to each sample
        
    Returns:
        pd.DataFrame: DataFrame with inference results or None if inference fails
    """
    try:
        if model is None or X is None or file_names is None:
            logger.error("Cannot run inference with None inputs")
            return None
        
        logger.info("Running inference with Isolation Forest model")
        
        # Generate anomaly scores (negative of decision function for isolation forest)
        # Higher values indicate more anomalous samples
        anomaly_scores = -model.decision_function(X)
        
        # Generate predictions (1 = normal, -1 = anomaly in isolation forest)
        predictions = model.predict(X)
        
        # Convert predictions to format where 1 = anomaly, 0 = normal
        is_anomaly = np.where(predictions == -1, 1, 0)
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'file_name': file_names,
            'anomaly_score': anomaly_scores,
            'is_anomaly': is_anomaly
        })
        
        logger.info(f"Inference completed for {len(results_df)} samples")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return None


def load_reconstruction_errors():
    """
    Load reconstruction errors from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with reconstruction errors or None if loading fails
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
        
        # If filename column exists, create file_name column from it
        if 'filename' in df.columns and 'file_name' not in df.columns:
            df['file_name'] = df['filename'].apply(lambda x: Path(x).stem)
        
        logger.info(f"Loaded reconstruction errors for {len(df)} samples")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading reconstruction errors: {e}")
        return None


def merge_with_reconstruction_errors(results_df):
    """
    Merge inference results with reconstruction errors.
    
    Args:
        results_df (pd.DataFrame): DataFrame with inference results
        
    Returns:
        pd.DataFrame: Merged DataFrame or original DataFrame if merging fails
    """
    try:
        if results_df is None or results_df.empty:
            logger.error("Cannot merge with empty or None results")
            return results_df
        
        # Load reconstruction errors
        recon_errors_df = load_reconstruction_errors()
        
        if recon_errors_df is None or recon_errors_df.empty:
            logger.warning("No reconstruction errors to merge, returning original results")
            return results_df
        
        logger.info("Merging inference results with reconstruction errors")
        
        # Merge DataFrames on file_name
        if 'file_name' not in recon_errors_df.columns:
            logger.warning("file_name column not found in reconstruction errors DataFrame")
            return results_df
        
        # Select only needed columns from reconstruction errors DataFrame
        if 'reconstruction_error' in recon_errors_df.columns:
            recon_errors_df = recon_errors_df[['file_name', 'reconstruction_error']]
        else:
            logger.warning("reconstruction_error column not found in reconstruction errors DataFrame")
            return results_df
        
        # Merge DataFrames
        merged_df = pd.merge(
            results_df,
            recon_errors_df,
            on='file_name',
            how='left'
        )
        
        logger.info(f"Merged DataFrame has {len(merged_df)} rows")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging with reconstruction errors: {e}")
        return results_df


def save_results(results_df, output_path):
    """
    Save inference results to CSV file.
    
    Args:
        results_df (pd.DataFrame): DataFrame with inference results
        output_path (Path): Path to save results
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if results_df is None or results_df.empty:
            logger.error("Cannot save empty or None results")
            return False
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Inference results saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained Isolation Forest model"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save inference results CSV (default: %(default)s)"
    )
    
    parser.add_argument(
        "--merge-recon",
        action="store_true",
        help="Merge results with reconstruction errors from autoencoder"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run inference with Isolation Forest model.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Convert output path string to Path object
        output_path = Path(args.output)
        
        logger.info("Starting Isolation Forest inference")
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model, aborting")
            return
        
        # Load latent vectors
        X, file_names = load_latent_vectors()
        if X is None or file_names is None:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Run inference
        results_df = run_inference(model, X, file_names)
        if results_df is None:
            logger.error("Failed to run inference, aborting")
            return
        
        # Merge with reconstruction errors if requested
        if args.merge_recon:
            logger.info("Merging with reconstruction errors")
            results_df = merge_with_reconstruction_errors(results_df)
        
        # Save results
        save_results(results_df, output_path)
        
        logger.info("Isolation Forest inference completed")
        
    except Exception as e:
        logger.error(f"Error in Isolation Forest inference: {e}")


if __name__ == "__main__":
    main()