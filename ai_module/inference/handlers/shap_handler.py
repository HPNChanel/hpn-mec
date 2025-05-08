#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Explanation Handler for Health Anomaly Detection

This module provides utilities for generating SHAP explanations for Isolation Forest
anomaly predictions, helping interpret which features contribute most to anomaly detection.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import joblib
import shap
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_explainer(
    model: Optional[IsolationForest] = None, 
    model_path: Optional[str] = None
) -> shap.Explainer:
    """
    Load or initialize a SHAP TreeExplainer for the Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest model (if already loaded)
        model_path: Path to the saved model file (if model not provided)
        
    Returns:
        shap.Explainer: Initialized SHAP explainer
        
    Raises:
        ValueError: If neither model nor model_path is provided
        TypeError: If model is not an IsolationForest
    """
    try:
        # Check if we need to load the model
        if model is None and model_path is None:
            # Try to load from default path
            model_path = Path(__file__).parent.parent.parent / "models" / "isolation_forest.pkl"
            if not model_path.exists():
                logger.warning(f"Model file not found at default path: {model_path}")
                # Check if there are alternative model files
                model_dir = model_path.parent
                alternate_models = list(model_dir.glob("*isolation*forest*.pkl"))
                
                if alternate_models:
                    # Use the first alternative model
                    model_path = alternate_models[0]
                    logger.warning(f"Using alternative model: {model_path}")
                else:
                    raise ValueError("No isolation forest model files found and no model provided")
            
        if model is None and model_path is not None:
            # Load model from file
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            logger.info(f"Loading Isolation Forest model from {model_path}")
            model = joblib.load(model_path)
        
        # Validate model type
        if not isinstance(model, IsolationForest):
            raise TypeError(f"Model must be an IsolationForest, got {type(model).__name__}")
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP TreeExplainer")
        explainer = shap.TreeExplainer(model)
        
        # Log success
        logger.info("SHAP TreeExplainer initialized successfully")
        
        return explainer
        
    except Exception as e:
        logger.error(f"Error loading SHAP explainer: {e}")
        raise

def explain_predictions(
    explainer: shap.Explainer,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 3,
    batch_size: Optional[int] = None
) -> List[List[str]]:
    """
    Generate SHAP explanations for anomaly predictions.
    
    Args:
        explainer: SHAP explainer
        X: Input feature matrix (latent vectors)
        feature_names: Names of features (optional)
        top_k: Number of top features to return
        batch_size: Batch size for processing large inputs (optional)
        
    Returns:
        List[List[str]]: Top contributing features for each sample
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError(f"Input must be a 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        logger.info(f"Generating SHAP explanations for {n_samples} samples with {n_features} features")
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            logger.warning(f"Number of feature names ({len(feature_names)}) doesn't match "
                          f"number of features ({n_features}). Using default names.")
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Process in batches if batch_size is provided and data is large enough
        if batch_size is not None and n_samples > batch_size:
            logger.info(f"Processing in batches of size {batch_size}")
            all_shap_values = []
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = X[i:end_idx]
                
                # Compute SHAP values for batch
                batch_values = explainer.shap_values(batch)
                
                # Handle different return types from SHAP
                if isinstance(batch_values, list) and len(batch_values) == 1:
                    batch_values = batch_values[0]
                
                all_shap_values.append(batch_values)
            
            # Combine batches
            shap_values = np.vstack(all_shap_values)
            
        else:
            # For small datasets, process all at once
            shap_values = explainer.shap_values(X)
            
            # Handle different return types from SHAP
            if isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]
        
        # Get top-k features for each sample
        top_features = []
        for i in range(n_samples):
            # Get absolute SHAP values for this sample
            abs_values = np.abs(shap_values[i])
            
            # Get indices of top-k features
            top_indices = np.argsort(abs_values)[::-1][:top_k]
            
            # Map indices to feature names
            sample_top_features = [feature_names[idx] for idx in top_indices]
            top_features.append(sample_top_features)
        
        logger.info(f"Generated explanations with top-{top_k} features for {n_samples} samples")
        
        return top_features
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        raise

def explain_with_values(
    explainer: shap.Explainer,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    batch_size: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate SHAP explanations with the full SHAP values matrix.
    
    Args:
        explainer: SHAP explainer
        X: Input feature matrix (latent vectors)
        feature_names: Names of features (optional)
        batch_size: Batch size for processing large inputs (optional)
        
    Returns:
        Tuple[np.ndarray, List[str]]: 
            - SHAP values array of shape [n_samples, n_features]
            - Feature names list
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError(f"Input must be a 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        logger.info(f"Generating full SHAP values for {n_samples} samples with {n_features} features")
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            logger.warning(f"Number of feature names ({len(feature_names)}) doesn't match "
                          f"number of features ({n_features}). Using default names.")
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Process in batches if batch_size is provided and data is large enough
        if batch_size is not None and n_samples > batch_size:
            logger.info(f"Processing in batches of size {batch_size}")
            all_shap_values = []
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = X[i:end_idx]
                
                # Compute SHAP values for batch
                batch_values = explainer.shap_values(batch)
                
                # Handle different return types from SHAP
                if isinstance(batch_values, list) and len(batch_values) == 1:
                    batch_values = batch_values[0]
                
                all_shap_values.append(batch_values)
            
            # Combine batches
            shap_values = np.vstack(all_shap_values)
            
        else:
            # For small datasets, process all at once
            shap_values = explainer.shap_values(X)
            
            # Handle different return types from SHAP
            if isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]
        
        logger.info(f"Generated full SHAP values with shape {shap_values.shape}")
        
        return shap_values, feature_names
        
    except Exception as e:
        logger.error(f"Error generating full SHAP values: {e}")
        raise

def get_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Calculate global feature importance based on mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values array [n_samples, n_features]
        feature_names: Names of features
        top_k: Number of top features to return (optional)
        
    Returns:
        List[Tuple[str, float]]: (feature_name, importance) pairs sorted by importance
    """
    try:
        # Calculate mean absolute SHAP values
        mean_abs_values = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance pairs
        importance_pairs = list(zip(feature_names, mean_abs_values))
        
        # Sort by importance (descending)
        sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            sorted_pairs = sorted_pairs[:top_k]
        
        return sorted_pairs
        
    except Exception as e:
        logger.error(f"Error calculating global feature importance: {e}")
        raise

# Test function for the handler
def test_handler(model_path: Optional[str] = None, n_samples: int = 10):
    """
    Test the SHAP handler with random data.
    
    Args:
        model_path: Path to the saved model file
        n_samples: Number of random samples to generate
    """
    try:
        # Load model
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "isolation_forest.pkl"
        else:
            model_path = Path(model_path)
            
        model = joblib.load(model_path)
        
        # Initialize explainer
        explainer = load_explainer(model)
        
        # Infer input dimensions from model's n_features_in_ attribute
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            # Default to 16 (typical latent dim)
            n_features = 16
            logger.warning(f"Could not determine input dimensions, using default: {n_features}")
        
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        
        # Generate feature names
        feature_names = [f"latent_{i}" for i in range(n_features)]
        
        # Generate explanations
        top_features = explain_predictions(explainer, X, feature_names, top_k=3)
        
        # Print results
        print("\nSHAP Explanation Test Results:")
        print(f"Input shape: {X.shape}")
        for i, features in enumerate(top_features):
            print(f"Sample {i} top features: {', '.join(features)}")
        
        # Test global importance
        shap_values, _ = explain_with_values(explainer, X, feature_names)
        global_importance = get_global_importance(shap_values, feature_names, top_k=5)
        
        print("\nGlobal Feature Importance:")
        for feature, importance in global_importance:
            print(f"{feature}: {importance:.6f}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    # Add parent directories to path if running as script
    if __package__ is None:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test SHAP handler")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to the saved model file"
    )
    parser.add_argument(
        "--n-samples", 
        type=int, 
        default=10,
        help="Number of random samples to generate"
    )
    
    args = parser.parse_args()
    
    # Run test
    test_handler(args.model_path, args.n_samples)