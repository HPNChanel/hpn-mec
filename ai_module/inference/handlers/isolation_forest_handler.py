#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Handler for Isolation Forest Model

This module provides utilities for loading the trained Isolation Forest model
and running inference to detect anomalies in the latent space.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, Union, Any
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_path: Optional[str] = None) -> IsolationForest:
    """
    Load the trained Isolation Forest model from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        IsolationForest: Loaded Isolation Forest model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If loaded model is not an IsolationForest
    """
    try:
        # Use Path for file path resolution
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "isolation_forest.pkl"
        else:
            model_path = Path(model_path)
        
        # Check if model file exists
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            # Check if there are alternative model files
            model_dir = model_path.parent
            alternate_models = list(model_dir.glob("*isolation*forest*.pkl"))
            
            if alternate_models:
                # Use the first alternative model
                model_path = alternate_models[0]
                logger.warning(f"Using alternative model: {model_path}")
            else:
                raise FileNotFoundError(f"No isolation forest model files found in {model_dir}")
        
        logger.info(f"Loading Isolation Forest model from {model_path}")
        
        # Load model using joblib
        model = joblib.load(model_path)
        
        # Validate that it's an IsolationForest model
        if not isinstance(model, IsolationForest):
            raise ValueError(f"Loaded model is not an IsolationForest, got {type(model).__name__}")
        
        # Log model parameters
        logger.info(f"Model loaded successfully with parameters:")
        logger.info(f"  n_estimators: {model.n_estimators}")
        logger.info(f"  contamination: {model.contamination}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading Isolation Forest model: {e}")
        raise

def predict_anomalies(
    model: IsolationForest, 
    X: np.ndarray, 
    batch_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict anomalies using the Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest model
        X: Input feature matrix (latent vectors)
        batch_size: Batch size for processing large inputs (optional)
        
    Returns:
        tuple: (anomaly_labels, anomaly_scores)
            - anomaly_labels: Binary labels (1=anomaly, 0=normal)
            - anomaly_scores: Anomaly scores (higher=more anomalous)
            
    Raises:
        ValueError: If input is not a 2D array
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError(f"Input must be a 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        logger.info(f"Running anomaly detection on {n_samples} samples with {n_features} features")
        
        # Process in batches if batch_size is provided and data is large enough
        if batch_size is not None and n_samples > batch_size:
            logger.info(f"Processing in batches of size {batch_size}")
            
            # Initialize arrays for results
            anomaly_scores = np.zeros(n_samples)
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                end_idx = min(i + batch_size, n_samples)
                batch = X[i:end_idx]
                
                # Compute decision function for batch (negative = more anomalous)
                batch_scores = -model.decision_function(batch)
                
                # Store results
                anomaly_scores[i:end_idx] = batch_scores
                
            # Run predict on full dataset (for consistency, as predict uses different logic than decision_function)
            raw_predictions = model.predict(X)
            
        else:
            # For small datasets, process all at once
            # Compute decision function (negative = more anomalous)
            anomaly_scores = -model.decision_function(X)
            
            # Get predictions
            raw_predictions = model.predict(X)
        
        # Convert predictions from IsolationForest format (-1=anomaly, 1=normal)
        # to binary format (1=anomaly, 0=normal)
        anomaly_labels = np.where(raw_predictions == -1, 1, 0)
        
        # Log prediction results
        n_anomalies = np.sum(anomaly_labels)
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/n_samples:.2%})")
        logger.info(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
        
        return anomaly_labels, anomaly_scores
        
    except Exception as e:
        logger.error(f"Error predicting anomalies: {e}")
        raise

def predict_anomalies_threshold(
    model: IsolationForest, 
    X: np.ndarray, 
    threshold: float,
    batch_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict anomalies using a custom threshold on the anomaly scores.
    
    Args:
        model: Trained Isolation Forest model
        X: Input feature matrix (latent vectors)
        threshold: Custom threshold for anomaly detection
        batch_size: Batch size for processing large inputs (optional)
        
    Returns:
        tuple: (anomaly_labels, anomaly_scores)
            - anomaly_labels: Binary labels (1=anomaly, 0=normal)
            - anomaly_scores: Anomaly scores (higher=more anomalous)
    """
    try:
        # Get anomaly scores (ignore original labels)
        _, anomaly_scores = predict_anomalies(model, X, batch_size)
        
        # Apply custom threshold
        anomaly_labels = np.where(anomaly_scores > threshold, 1, 0)
        
        # Log prediction results
        n_samples = len(anomaly_scores)
        n_anomalies = np.sum(anomaly_labels)
        logger.info(f"Applied custom threshold {threshold:.4f}")
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/n_samples:.2%})")
        
        return anomaly_labels, anomaly_scores
        
    except Exception as e:
        logger.error(f"Error predicting anomalies with threshold: {e}")
        raise

def get_anomaly_threshold(model: IsolationForest, X: np.ndarray, contamination: float = 0.1) -> float:
    """
    Calculate the anomaly threshold based on a desired contamination rate.
    
    Args:
        model: Trained Isolation Forest model
        X: Input feature matrix (latent vectors)
        contamination: Desired contamination rate (proportion of anomalies)
        
    Returns:
        float: Anomaly threshold
    """
    try:
        # Get anomaly scores
        _, anomaly_scores = predict_anomalies(model, X)
        
        # Calculate threshold based on percentile
        threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
        
        logger.info(f"Calculated anomaly threshold for {contamination:.2%} contamination: {threshold:.4f}")
        
        return threshold
        
    except Exception as e:
        logger.error(f"Error calculating anomaly threshold: {e}")
        raise

# Test function for the handler
def test_handler(model_path: Optional[str] = None, n_samples: int = 10):
    """
    Test the Isolation Forest handler with random data.
    
    Args:
        model_path: Path to the saved model file
        n_samples: Number of random samples to generate
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Infer input dimensions from model's n_features_in_ attribute
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            # Default to 16 (typical latent dim)
            n_features = 16
            logger.warning(f"Could not determine input dimensions, using default: {n_features}")
        
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        
        # Run prediction
        labels, scores = predict_anomalies(model, X)
        
        # Print results
        print("\nTest Results:")
        print(f"Input shape: {X.shape}")
        print(f"Predictions: {labels}")
        print(f"Scores: {scores}")
        print(f"Anomalies detected: {np.sum(labels)}/{n_samples}")
        
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
    parser = argparse.ArgumentParser(description="Test Isolation Forest handler")
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