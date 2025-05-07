#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Inference Script for ECG Signal Anomaly Detection

This script loads a trained Isolation Forest model and runs inference on latent
vectors to detect anomalous ECG signals. Results are saved to CSV format.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
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

# Add module directory to path for imports
sys.path.insert(0, str(module_dir))

# Now use absolute imports
from utils.load_latents import load_latents

# Constants and paths
MODELS_DIR = module_dir / "models"
REPORTS_DIR = module_dir / "reports"
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "inference_isolation_result.csv"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """
    Load the trained Isolation Forest model.
    
    Returns:
        model: Trained Isolation Forest model or None if loading fails
    """
    try:
        # Check if model exists
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            return None
        
        # Load model
        model = joblib.load(MODEL_PATH)
        
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None


def run_inference(model, df_latents):
    """
    Run inference using the Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest model
        df_latents (pd.DataFrame): DataFrame with latent vectors
        
    Returns:
        pd.DataFrame: DataFrame with inference results or None if inference fails
    """
    try:
        if model is None or df_latents is None or df_latents.empty:
            logger.error("Cannot run inference with None inputs")
            return None
        
        logger.info(f"Running inference on {len(df_latents)} samples")
        
        # Extract feature columns
        feature_cols = [col for col in df_latents.columns if col.startswith('feature_')]
        if not feature_cols:
            logger.error("No feature columns found in latents DataFrame")
            return None
            
        logger.info(f"Using {len(feature_cols)} features for inference")
        
        # Extract feature matrix
        X = df_latents[feature_cols].values
        
        # Create a copy for results
        results_df = df_latents[['file_name']].copy()
        
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
        
        # Calculate summary statistics
        n_anomalies = np.sum(is_anomaly)
        anomaly_rate = n_anomalies / len(is_anomaly)
        
        logger.info(f"Inference completed. Identified {n_anomalies} anomalies ({anomaly_rate:.2%})")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        logger.error(traceback.format_exc())
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
        
        # Define path to reconstruction errors CSV
        recon_errors_path = REPORTS_DIR / "reconstruction_scores.csv"
        
        # Check if file exists
        if not recon_errors_path.exists():
            logger.warning(f"Reconstruction errors file not found: {recon_errors_path}")
            return results_df
        
        logger.info("Merging inference results with reconstruction errors")
        
        # Load CSV into DataFrame
        recon_df = pd.read_csv(recon_errors_path)
        
        # Determine which column has the file names in reconstruction errors DataFrame
        file_col = None
        for col_name in ['file_name', 'filename']:
            if col_name in recon_df.columns:
                file_col = col_name
                break
        
        if file_col is None:
            logger.error("Could not find file name column in reconstruction errors")
            return results_df
        
        # Determine which column has the reconstruction errors
        error_col = None
        for col_name in ['reconstruction_error', 'mse', 'error']:
            if col_name in recon_df.columns:
                error_col = col_name
                break
        
        if error_col is None:
            logger.error("Could not find reconstruction error column")
            return results_df
        
        # Extract file stems from recon_df if needed
        if '.' in recon_df[file_col].iloc[0]:
            recon_df['file_name'] = recon_df[file_col].apply(
                lambda x: Path(x).stem
            )
        else:
            recon_df['file_name'] = recon_df[file_col]
        
        # Merge DataFrames
        columns_to_merge = ['file_name', error_col]
        merged_df = pd.merge(
            results_df,
            recon_df[columns_to_merge],
            on='file_name',
            how='left'
        )
        
        # Rename the error column to a standardized name if it's not already
        if error_col != 'reconstruction_error':
            merged_df.rename(columns={error_col: 'reconstruction_error'}, inplace=True)
        
        logger.info(f"Merged DataFrame has {len(merged_df)} rows")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging with reconstruction errors: {e}")
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        df_latents = load_latents()
        if df_latents is None or df_latents.empty:
            logger.error("Failed to load latent vectors, aborting")
            return
            
        # Drop split column if it exists, as we want to run inference on all data
        if 'split' in df_latents.columns:
            df_latents = df_latents.drop(columns=['split'])
            logger.info("Dropped 'split' column to run inference on all data")
        
        # Run inference
        results_df = run_inference(model, df_latents)
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
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()