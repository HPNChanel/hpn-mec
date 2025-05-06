#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis for Isolation Forest Model

This script uses SHAP (SHapley Additive exPlanations) to explain the predictions
of the trained Isolation Forest model for ECG signal anomaly detection.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import shap

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
PLOTS_DIR = REPORTS_DIR / "plots"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """
    Load the trained Isolation Forest model.
    
    Returns:
        model: Trained Isolation Forest model
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


def load_latent_vectors(max_samples=None):
    """
    Load all latent vector files from the latents directory.
    
    Args:
        max_samples (int, optional): Maximum number of samples to load
        
    Returns:
        tuple: (feature_matrix, file_names)
    """
    try:
        # Get all .npy files in the latents directory
        latent_files = list(LATENTS_DIR.glob("*.npy"))
        
        if not latent_files:
            logger.error(f"No latent vector files found in {LATENTS_DIR}")
            return None, None
        
        # Limit number of files if specified
        if max_samples is not None and max_samples < len(latent_files):
            latent_files = latent_files[:max_samples]
            logger.info(f"Limited to {max_samples} latent vector files")
        else:
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
        def wrapped_model(X):
            # Return negative of decision function (higher = more anomalous)
            return -model.decision_function(X)
        
        logger.info("Created model wrapper for SHAP analysis")
        
        return wrapped_model
        
    except Exception as e:
        logger.error(f"Error creating model wrapper: {e}")
        return None


def generate_shap_explanations(model, X, max_samples=100):
    """
    Generate SHAP values for the model predictions.
    
    Args:
        model: Trained Isolation Forest model
        X (numpy.ndarray): Feature matrix
        max_samples (int): Maximum number of samples to explain
        
    Returns:
        shap.Explanation: SHAP explanation object
    """
    try:
        # Create model wrapper
        model_func = create_model_wrapper(model)
        if model_func is None:
            logger.error("Model wrapper creation failed")
            return None
        
        # Limit number of samples for SHAP analysis if needed
        if max_samples and X.shape[0] > max_samples:
            logger.info(f"Limiting SHAP analysis to {max_samples} samples")
            X_subset = X[:max_samples]
        else:
            X_subset = X
        
        logger.info(f"Generating SHAP explanations for {X_subset.shape[0]} samples")
        
        # Create feature names
        feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
        
        # Suppress warnings during SHAP computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Initialize SHAP explainer
            # Use KernelExplainer for model-agnostic explanations
            # This works with any black-box model
            explainer = shap.KernelExplainer(model_func, X_subset)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_subset)
            
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values,
                data=X_subset,
                feature_names=feature_names
            )
        
        logger.info("SHAP explanations generated successfully")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        return None


def plot_shap_summary(shap_explanation, output_dir):
    """
    Create and save SHAP summary plot (beeswarm plot).
    
    Args:
        shap_explanation: SHAP explanation object
        output_dir (Path): Directory to save plot
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Create figure for better control
        plt.figure(figsize=(12, 8))
        
        # Suppress warnings during plotting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create summary plot (beeswarm plot)
            shap.summary_plot(
                shap_explanation.values,
                shap_explanation.data,
                feature_names=shap_explanation.feature_names,
                show=False
            )
        
        # Add title
        plt.title("SHAP Summary Plot for Isolation Forest", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "explain_isolation_forest_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP summary plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting SHAP summary: {e}")
        return None


def plot_shap_bar(shap_explanation, output_dir):
    """
    Create and save SHAP bar plot of mean absolute values.
    
    Args:
        shap_explanation: SHAP explanation object
        output_dir (Path): Directory to save plot
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Create figure for better control
        plt.figure(figsize=(10, 8))
        
        # Suppress warnings during plotting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create bar plot of mean absolute SHAP values
            shap.plots.bar(
                shap_explanation,
                show=False
            )
        
        # Add title
        plt.title("Mean Impact on Model Output (Magnitude)", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "explain_isolation_forest_bar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP bar plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting SHAP bar chart: {e}")
        return None


def main():
    """
    Main function to generate SHAP explanations for Isolation Forest model.
    """
    try:
        logger.info("Starting SHAP analysis for Isolation Forest")
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model, aborting")
            return
        
        # Load latent vectors (limit to 100 samples for performance)
        X, file_names = load_latent_vectors(max_samples=100)
        if X is None:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Generate SHAP explanations
        shap_explanation = generate_shap_explanations(model, X)
        if shap_explanation is None:
            logger.error("Failed to generate SHAP explanations, aborting")
            return
        
        # Create and save plots
        summary_path = plot_shap_summary(shap_explanation, PLOTS_DIR)
        bar_path = plot_shap_bar(shap_explanation, PLOTS_DIR)
        
        # Log completion
        logger.info("SHAP analysis for Isolation Forest completed")
        logger.info(f"Summary plot saved to: {summary_path}")
        logger.info(f"Bar plot saved to: {bar_path}")
        
    except Exception as e:
        logger.error(f"Error in SHAP analysis for Isolation Forest: {e}")


if __name__ == "__main__":
    main()