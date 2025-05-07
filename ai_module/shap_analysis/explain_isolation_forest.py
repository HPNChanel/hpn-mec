#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Explanation for Isolation Forest Anomaly Detection

This script uses SHAP (SHapley Additive exPlanations) to explain the predictions
of the trained Isolation Forest model for ECG signal anomaly detection.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import shap
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
PLOTS_DIR = REPORTS_DIR / "plots"
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
ANOMALY_SCORES_PATH = REPORTS_DIR / "anomaly_scores.csv"

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """
    Load the trained Isolation Forest model.
    
    Returns:
        model: Trained Isolation Forest model
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


def load_anomaly_scores():
    """
    Load anomaly scores from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly scores
    """
    try:
        # Check if file exists
        if not ANOMALY_SCORES_PATH.exists():
            logger.error(f"Anomaly scores file not found: {ANOMALY_SCORES_PATH}")
            return None
        
        # Load CSV
        df = pd.read_csv(ANOMALY_SCORES_PATH)
        
        logger.info(f"Loaded anomaly scores for {len(df)} samples")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading anomaly scores: {e}")
        logger.error(traceback.format_exc())
        return None


def select_anomalous_samples(df_scores, latents_df, n_samples=50):
    """
    Select the top most anomalous samples from the dataset.
    
    Args:
        df_scores (pd.DataFrame): DataFrame with anomaly scores
        latents_df (pd.DataFrame): DataFrame with latent vectors
        n_samples (int): Number of samples to select
        
    Returns:
        tuple: (X_anomalous, file_names, df_merged)
    """
    try:
        if df_scores is None or latents_df is None:
            logger.error("Cannot select samples from None DataFrames")
            return None, None, None
        
        # Filter anomalous samples
        anomalies = df_scores[df_scores['is_anomaly'] == 1].copy()
        
        if len(anomalies) == 0:
            logger.warning("No anomalous samples found, using top scores instead")
            # If no anomalies found, take top scores instead
            anomalies = df_scores.sort_values('anomaly_score', ascending=False).copy()
        
        # Sort by anomaly score (descending)
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Limit to n_samples
        if len(anomalies) > n_samples:
            logger.info(f"Limiting to top {n_samples} anomalous samples")
            anomalies = anomalies.head(n_samples)
        
        # Merge with latents
        merged_df = pd.merge(
            anomalies,
            latents_df,
            on='file_name',
            how='inner',
            suffixes=('', '_latent')
        )
        
        if len(merged_df) == 0:
            logger.error("No matching samples found after merging")
            return None, None, None
        
        logger.info(f"Selected {len(merged_df)} anomalous samples for explanation")
        
        # Extract feature columns
        feature_cols = [col for col in merged_df.columns if col.startswith('feature_')]
        X_anomalous = merged_df[feature_cols].values
        file_names = merged_df['file_name'].tolist()
        
        return X_anomalous, file_names, merged_df
        
    except Exception as e:
        logger.error(f"Error selecting anomalous samples: {e}")
        logger.error(traceback.format_exc())
        return None, None, None


def create_model_wrapper(model):
    """
    Create a wrapper function for the model's decision_function.
    
    Args:
        model: Trained Isolation Forest model
        
    Returns:
        callable: Wrapped model function for SHAP
    """
    try:
        # For Isolation Forest, we typically use negative of decision_function
        # as anomaly score (higher score = more anomalous)
        def model_function(X):
            # Ensure X is a 2D array
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            # Return negative of decision function (higher = more anomalous)
            return -model.decision_function(X)
        
        logger.info("Created model wrapper for SHAP analysis")
        
        return model_function
        
    except Exception as e:
        logger.error(f"Error creating model wrapper: {e}")
        logger.error(traceback.format_exc())
        return None


def compute_shap_values(model, X_samples, X_background=None):
    """
    Compute SHAP values for the model predictions.
    
    Args:
        model: Trained Isolation Forest model
        X_samples (numpy.ndarray): Feature matrix for samples to explain
        X_background (numpy.ndarray, optional): Background data for SHAP explainer
        
    Returns:
        tuple: (shap_values, explainer)
    """
    try:
        # Create model wrapper
        model_func = create_model_wrapper(model)
        if model_func is None:
            logger.error("Model wrapper creation failed")
            return None, None
        
        # If background data not provided, use subset of samples
        if X_background is None:
            logger.info("Using samples subset as background data")
            # Use at most 100 samples for background
            n_background = min(100, len(X_samples))
            idx = np.random.choice(len(X_samples), n_background, replace=False)
            X_background = X_samples[idx]
        
        logger.info(f"Initializing SHAP KernelExplainer with {len(X_background)} background samples")
        
        # Suppress warnings during SHAP computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Initialize KernelExplainer
            explainer = shap.KernelExplainer(model_func, X_background)
            
            # Compute SHAP values (limit samples for efficiency if needed)
            max_explain = min(50, len(X_samples))
            if len(X_samples) > max_explain:
                logger.info(f"Computing SHAP values for {max_explain} samples (out of {len(X_samples)})")
                X_subset = X_samples[:max_explain]
            else:
                X_subset = X_samples
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_subset, nsamples=100)
        
        logger.info(f"SHAP values computed successfully with shape: {np.array(shap_values).shape}")
        
        return shap_values, explainer
        
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        logger.error(traceback.format_exc())
        return None, None


def plot_shap_summary(shap_values, X, feature_names=None):
    """
    Create and save SHAP summary plot (beeswarm).
    
    Args:
        shap_values: SHAP values
        X (numpy.ndarray): Feature matrix
        feature_names (list, optional): Names of features
        
    Returns:
        str: Path to saved plot
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Suppress warnings during plotting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create summary plot
            shap.summary_plot(
                shap_values, 
                X, 
                feature_names=feature_names,
                show=False
            )
        
        # Add title
        plt.title("SHAP Feature Importance for Isolation Forest Anomaly Detection", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = PLOTS_DIR / "explain_iforest_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP summary plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting SHAP summary: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_shap_bar(shap_values, feature_names=None):
    """
    Create and save SHAP bar plot (mean absolute values).
    
    Args:
        shap_values: SHAP values
        feature_names (list, optional): Names of features
        
    Returns:
        str: Path to saved plot
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(0)
        
        # If feature_names not provided, create generic names
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(mean_abs_shap))]
        
        # Create DataFrame for plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        shap_df = shap_df.sort_values('importance', ascending=False)
        
        # Plot top 20 features
        top_n = min(20, len(shap_df))
        plt.barh(
            y=shap_df['feature'].values[:top_n],
            width=shap_df['importance'].values[:top_n],
            color='skyblue'
        )
        
        # Add labels and title
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance (Mean Absolute SHAP Values)', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_path = PLOTS_DIR / "explain_iforest_bar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP bar plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting SHAP bar chart: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main function to generate SHAP explanations for Isolation Forest model.
    """
    try:
        logger.info("Starting SHAP explanation for Isolation Forest")
        
        # Set matplotlib style
        plt.style.use('fivethirtyeight')
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model, aborting")
            return
        
        # Load latent vectors
        latents_df = load_latents()
        if latents_df is None:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Load anomaly scores
        scores_df = load_anomaly_scores()
        if scores_df is None:
            logger.error("Failed to load anomaly scores, aborting")
            return
        
        # Select anomalous samples
        X_anomalous, file_names, merged_df = select_anomalous_samples(scores_df, latents_df)
        if X_anomalous is None:
            logger.error("Failed to select anomalous samples, aborting")
            return
        
        # Identify feature columns
        feature_cols = [col for col in latents_df.columns if col.startswith('feature_')]
        
        # Compute SHAP values
        shap_values, explainer = compute_shap_values(model, X_anomalous)
        if shap_values is None:
            logger.error("Failed to compute SHAP values, aborting")
            return
        
        # Plot SHAP summary
        summary_path = plot_shap_summary(shap_values, X_anomalous, feature_cols)
        
        # Plot SHAP bar chart
        bar_path = plot_shap_bar(shap_values, feature_cols)
        
        # Final message
        if summary_path and bar_path:
            logger.info("SHAP explanation for Isolation Forest completed successfully")
            logger.info(f"Summary plot: {summary_path}")
            logger.info(f"Bar plot: {bar_path}")
        else:
            logger.warning("SHAP explanation completed with some errors")
        
    except Exception as e:
        logger.error(f"Error in SHAP explanation: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()