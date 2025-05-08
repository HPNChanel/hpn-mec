#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP-based Explanation for Isolation Forest Anomaly Detection

This script loads a trained Isolation Forest model and uses SHAP (SHapley 
Additive exPlanations) to interpret which latent features contribute most
to anomaly detection decisions in health data.

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
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_dir_exists(path):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Path to check/create
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def load_data(model_path, latents_path, labels_path=None, scores_path=None):
    """
    Load the trained model, latent vectors, and optionally labels and scores.
    
    Args:
        model_path: Path to the trained Isolation Forest model
        latents_path: Path to the latent vectors
        labels_path: Path to the anomaly labels (optional)
        scores_path: Path to the anomaly scores (optional)
        
    Returns:
        tuple: (model, latent_vectors, labels, scores)
    """
    try:
        # Load model
        logger.info(f"Loading Isolation Forest model from {model_path}")
        model = joblib.load(model_path)
        
        # Load latent vectors
        logger.info(f"Loading latent vectors from {latents_path}")
        latent_vectors = np.load(latents_path)
        logger.info(f"Loaded latent vectors with shape {latent_vectors.shape}")
        
        # Initialize labels and scores as None
        labels = None
        scores = None
        
        # Load labels if path provided
        if labels_path:
            logger.info(f"Loading anomaly labels from {labels_path}")
            labels = np.load(labels_path)
            logger.info(f"Loaded labels with shape {labels.shape}")
        
        # Load scores if path provided
        if scores_path:
            logger.info(f"Loading anomaly scores from {scores_path}")
            scores = np.load(scores_path)
            logger.info(f"Loaded scores with shape {scores.shape}")
        
        return model, latent_vectors, labels, scores
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def get_top_k_anomalies(latent_vectors, labels=None, scores=None, top_k=10):
    """
    Get the top-k most anomalous samples based on labels or scores.
    
    Args:
        latent_vectors: Latent space representations
        labels: Binary anomaly labels (1=anomaly, 0=normal)
        scores: Anomaly scores (higher=more anomalous)
        top_k: Number of top anomalies to return
        
    Returns:
        tuple: (top_k_vectors, top_k_indices)
    """
    # If we have scores, use them to sort
    if scores is not None:
        # Get indices of top_k highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        logger.info(f"Selected top {top_k} samples by anomaly score")
    
    # If we have labels but no scores, use labels
    elif labels is not None:
        # Only consider anomalies (label=1)
        anomaly_indices = np.where(labels == 1)[0]
        
        # If fewer anomalies than top_k, use all anomalies
        if len(anomaly_indices) <= top_k:
            top_indices = anomaly_indices
            logger.info(f"Selected all {len(top_indices)} anomalies (fewer than requested top_k={top_k})")
        else:
            # Randomly select top_k anomalies
            np.random.shuffle(anomaly_indices)
            top_indices = anomaly_indices[:top_k]
            logger.info(f"Randomly selected {top_k} samples from {len(anomaly_indices)} anomalies")
    
    # If neither labels nor scores, use random samples
    else:
        # Randomly select top_k samples
        top_indices = np.random.choice(len(latent_vectors), top_k, replace=False)
        logger.info(f"Randomly selected {top_k} samples (no scores or labels provided)")
    
    # Get the corresponding latent vectors
    top_vectors = latent_vectors[top_indices]
    
    return top_vectors, top_indices

def explain_with_shap(model, X, feature_names=None, top_k=None, use_predict=True):
    """
    Explain model decisions using SHAP TreeExplainer.
    
    Args:
        model: Trained Isolation Forest model
        X: Latent vectors to explain
        feature_names: Names of latent features (optional)
        top_k: Limit explanation to top-k samples (optional)
        use_predict: Whether to use model.predict instead of decision_function
        
    Returns:
        tuple: (explainer, shap_values, base_value)
    """
    logger.info("Creating SHAP TreeExplainer")
    
    # Select function to explain (predict or decision_function)
    if use_predict:
        # Note: For Isolation Forest, predict returns 1 for normal, -1 for anomaly
        # We negate it so that positive SHAP value = contribution towards being anomalous
        func = lambda x: -model.predict(x)
        logger.info("Using model.predict function for SHAP explanation")
    else:
        # For decision_function, lower values = more anomalous
        # We also negate it for consistent interpretation
        func = lambda x: -model.decision_function(x)
        logger.info("Using model.decision_function for SHAP explanation")
    
    # Create explainer with the selected function
    explainer = shap.TreeExplainer(model)
    
    # Use subset of data if top_k specified
    if top_k is not None and top_k < len(X):
        X_sample = X[:top_k]
        logger.info(f"Explaining top {top_k} samples (out of {len(X)})")
    else:
        X_sample = X
        logger.info(f"Explaining all {len(X)} samples")
    
    # Calculate SHAP values
    logger.info("Calculating SHAP values (this may take a while for large datasets)")
    shap_values = explainer.shap_values(X_sample)
    
    # For Isolation Forest, shap_values might be a single array or a list
    # Ensure we have a single array for easier handling
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]
    
    # Log shape of SHAP values
    logger.info(f"Generated SHAP values with shape {shap_values.shape}")
    
    return explainer, shap_values, explainer.expected_value

def plot_shap_summary(shap_values, X, feature_names=None, top_k=None, output_dir=None, 
                     timestamp=False, show_plots=True):
    """
    Create and save SHAP summary visualizations.
    
    Args:
        shap_values: SHAP values from explainer
        X: Input samples
        feature_names: Names for features
        top_k: Number of top features to show
        output_dir: Directory to save plots
        timestamp: Whether to add timestamp to filenames
        show_plots: Whether to display plots
        
    Returns:
        dict: Paths to saved figures
    """
    # Generate timestamp suffix if requested
    suffix = ""
    if timestamp:
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Ensure output directory exists
    if output_dir:
        output_dir = ensure_dir_exists(output_dir)
    
    # Prepare dictionary to store output paths
    output_paths = {}
    
    # 1. Bar plot of feature importance (mean absolute SHAP value)
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance (Mean |SHAP Value|)", fontsize=14)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(0)
    
    # Combine with feature names and sort
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })
    features_df = features_df.sort_values('Importance', ascending=False)
    
    # Limit to top_k features if specified
    if top_k and top_k < len(features_df):
        features_df = features_df.head(top_k)
    
    # Create bar plot
    plt.barh(y=features_df['Feature'], width=features_df['Importance'])
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir:
        bar_path = output_dir / f"shap_bar{suffix}.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bar plot to {bar_path}")
        output_paths['bar_plot'] = bar_path
    
    # Show figure if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        max_display=top_k if top_k else 20,
        show=False  # Don't show yet, we'll handle this manually
    )
    plt.title("SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir:
        summary_path = output_dir / f"shap_summary{suffix}.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary plot to {summary_path}")
        output_paths['summary_plot'] = summary_path
    
    # Show figure if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return output_paths

def create_individual_explanations(explainer, shap_values, X, indices, top_k=5, 
                                  output_dir=None, timestamp=False):
    """
    Create and save explanations for individual samples.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        X: Input samples
        indices: Indices of samples to explain
        top_k: Number of top samples to explain
        output_dir: Directory to save plots
        timestamp: Whether to add timestamp to filenames
        
    Returns:
        list: Paths to saved figures
    """
    # Generate timestamp suffix if requested
    suffix = ""
    if timestamp:
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output directory exists
    if output_dir:
        output_dir = ensure_dir_exists(output_dir)
    
    # Limit to top_k
    if top_k and top_k < len(indices):
        indices = indices[:top_k]
    
    # List to store output paths
    output_paths = []
    
    # Create plots for each sample
    for i, idx in enumerate(indices):
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create waterfall plot using Matplotlib
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[i], 
                    base_values=explainer.expected_value,
                    data=X[i]
                ),
                max_display=20,
                show=False
            )
            
            plt.title(f"SHAP Explanation for Sample {idx}", fontsize=14)
            plt.tight_layout()
            
            # Save figure if output directory provided
            if output_dir:
                path = output_dir / f"shap_sample_{idx}{suffix}.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved explanation for sample {idx} to {path}")
                output_paths.append(path)
            
            plt.close()
        
        except Exception as e:
            logger.warning(f"Error creating explanation for sample {idx}: {e}")
    
    return output_paths

def save_shap_values(shap_values, base_value, output_dir, timestamp=False):
    """
    Save SHAP values and base value to numpy files.
    
    Args:
        shap_values: SHAP values from explainer
        base_value: Base (expected) value from explainer
        output_dir: Directory to save files
        timestamp: Whether to add timestamp to filenames
        
    Returns:
        tuple: (shap_values_path, base_value_path)
    """
    # Generate timestamp suffix if requested
    suffix = ""
    if timestamp:
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output directory exists
    output_dir = ensure_dir_exists(output_dir)
    
    # Save SHAP values
    shap_values_path = output_dir / f"shap_values{suffix}.npy"
    np.save(shap_values_path, shap_values)
    logger.info(f"Saved SHAP values to {shap_values_path}")
    
    # Save base value
    base_value_path = output_dir / f"base_value{suffix}.npy"
    np.save(base_value_path, base_value)
    logger.info(f"Saved base value to {base_value_path}")
    
    return shap_values_path, base_value_path

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Explain Isolation Forest with SHAP")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="ai_module/models/isolation_forest.pkl",
        help="Path to the trained Isolation Forest model (default: %(default)s)"
    )
    
    parser.add_argument(
        "--latents-path",
        type=str,
        default="data/processed/latent_vectors/latents.npy",
        help="Path to latent vectors file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/processed/latent_vectors/anomaly_labels.npy",
        help="Path to anomaly labels file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--scores-path",
        type=str,
        default="data/processed/latent_vectors/anomaly_scores.npy",
        help="Path to anomaly scores file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/shap",
        help="Directory to save outputs (default: %(default)s)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top anomalies to explain (default: %(default)s)"
    )
    
    parser.add_argument(
        "--feature-top-k",
        type=int,
        default=15,
        help="Number of top features to show in visualizations (default: %(default)s)"
    )
    
    parser.add_argument(
        "--use-decision-function",
        action="store_true",
        help="Use decision_function instead of predict for explanations"
    )
    
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="Save figures without displaying them"
    )
    
    parser.add_argument(
        "--individual-explanations",
        action="store_true",
        help="Generate individual explanations for top-k anomalies"
    )
    
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output filenames"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to load data, generate SHAP explanations, and save results.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    model, latent_vectors, labels, scores = load_data(
        model_path=args.model_path,
        latents_path=args.latents_path,
        labels_path=args.labels_path,
        scores_path=args.scores_path
    )
    
    # Get top-k anomalies
    top_vectors, top_indices = get_top_k_anomalies(
        latent_vectors=latent_vectors,
        labels=labels,
        scores=scores,
        top_k=args.top_k
    )
    
    # Generate SHAP explanations
    explainer, shap_values, base_value = explain_with_shap(
        model=model,
        X=latent_vectors,  # Use all vectors for base explanations
        top_k=None,  # We'll limit visualizations later
        use_predict=not args.use_decision_function
    )
    
    # Save SHAP values and base value
    save_shap_values(
        shap_values=shap_values,
        base_value=base_value,
        output_dir=args.output_dir,
        timestamp=args.timestamp
    )
    
    # Create and save visualizations
    plot_shap_summary(
        shap_values=shap_values,
        X=latent_vectors,
        top_k=args.feature_top_k,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        show_plots=not args.save_fig
    )
    
    # Create individual explanations if requested
    if args.individual_explanations:
        create_individual_explanations(
            explainer=explainer,
            shap_values=shap_values,
            X=latent_vectors,
            indices=top_indices,
            top_k=args.top_k,
            output_dir=args.output_dir,
            timestamp=args.timestamp
        )
    
    logger.info("SHAP explanation generation complete.")

if __name__ == "__main__":
    main()