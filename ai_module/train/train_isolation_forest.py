#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Training for Anomaly Detection

This script trains an Isolation Forest model on preprocessed latent vectors
to detect anomalies in health data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import argparse
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_isolation_forest(
    X_train, 
    n_estimators=100, 
    contamination=0.1, 
    random_state=42,
    verbose=1
):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X_train: Training data (latent vectors)
        n_estimators: Number of base estimators (trees)
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        
    Returns:
        IsolationForest: Trained model
    """
    logger.info(f"Training Isolation Forest with {n_estimators} estimators, "
                f"contamination={contamination}")
    
    # Create and train model
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=verbose
    )
    
    model.fit(X_train)
    
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, X_train=None, threshold=None):
    """
    Evaluate the Isolation Forest model.
    
    Args:
        model: Trained IsolationForest model
        X_test: Test data
        X_train: Training data (for comparing distributions)
        threshold: Custom anomaly threshold (if None, use model's default)
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating Isolation Forest model")
    
    # Get predictions (-1 for anomalies, 1 for normal in scikit-learn)
    y_pred = model.predict(X_test)
    
    # Get anomaly scores (decision function: lower = more anomalous)
    scores_test = -model.decision_function(X_test)
    
    # Count anomalies
    anomaly_count = np.sum(y_pred == -1)
    anomaly_rate = anomaly_count / len(y_pred)
    
    logger.info(f"Detected {anomaly_count} anomalies ({anomaly_rate:.2%}) in test set")
    logger.info(f"Score range: [{scores_test.min():.4f}, {scores_test.max():.4f}]")
    
    # If threshold provided, apply it
    if threshold is not None:
        custom_preds = (scores_test > threshold).astype(int)
        custom_anomaly_count = np.sum(custom_preds)
        custom_anomaly_rate = custom_anomaly_count / len(custom_preds)
        
        logger.info(f"With custom threshold {threshold:.4f}: "
                   f"{custom_anomaly_count} anomalies ({custom_anomaly_rate:.2%})")
    
    # If training data provided, compare score distributions
    if X_train is not None:
        scores_train = -model.decision_function(X_train)
        
        logger.info("Score statistics:")
        logger.info(f"  Train: mean={scores_train.mean():.4f}, std={scores_train.std():.4f}")
        logger.info(f"  Test:  mean={scores_test.mean():.4f}, std={scores_test.std():.4f}")
        
        # Optional: calculate additional metrics like KS test, etc.
    
    # Return evaluation metrics
    metrics = {
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "score_min": scores_test.min(),
        "score_max": scores_test.max(),
        "score_mean": scores_test.mean(),
        "score_std": scores_test.std()
    }
    
    if threshold is not None:
        metrics.update({
            "custom_threshold": threshold,
            "custom_anomaly_count": custom_anomaly_count,
            "custom_anomaly_rate": custom_anomaly_rate
        })
    
    return metrics, scores_test

def find_optimal_threshold(model, X_train, target_rate=0.01):
    """
    Find an anomaly threshold that yields the target anomaly rate.
    
    Args:
        model: Trained IsolationForest model
        X_train: Training data
        target_rate: Target anomaly rate
        
    Returns:
        float: Optimal threshold
    """
    # Get anomaly scores
    scores = -model.decision_function(X_train)
    
    # Find percentile that gives target rate
    threshold = np.percentile(scores, 100 * (1 - target_rate))
    
    logger.info(f"Optimal threshold for {target_rate:.2%} anomaly rate: {threshold:.4f}")
    
    return threshold

def save_model(model, model_path, threshold=None):
    """
    Save the trained model and optional threshold.
    
    Args:
        model: Trained IsolationForest model
        model_path: Path to save the model
        threshold: Custom anomaly threshold (optional)
    """
    # Create directory if it doesn't exist
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # If threshold provided, attach it to the model
    if threshold is not None:
        model._anomaly_threshold = threshold
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    """
    Main function to load data, train model, and save results
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Isolation Forest for anomaly detection")
    
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of estimators (default: 100)")
    parser.add_argument("--contamination", type=float, default=0.1,
                        help="Expected proportion of anomalies (default: 0.1)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--target-rate", type=float, default=0.01,
                        help="Target anomaly rate for threshold optimization (default: 0.01)")
    parser.add_argument("--optimize-threshold", action="store_true",
                        help="Find optimal threshold for target anomaly rate")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Verbosity level (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to input data file (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Use Path for file path handling
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define paths
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Try several possible latent vector paths
        possible_data_paths = [
            script_dir.parent / "data" / "processed" / "latents" / "latents.npy",
            project_root / "data" / "processed" / "latents" / "latents.npy",
            project_root / "ai_module" / "data" / "processed" / "latents" / "latents.npy"
        ]
        
        data_path = None
        for path in possible_data_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            logger.error("Could not find latent vectors file. Please specify with --data-path")
            sys.exit(1)
    
    # Define output path
    model_dir = script_dir.parent / "models"
    model_path = model_dir / "isolation_forest.pkl"
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    logger.info(f"Loading data from {data_path}")
    
    try:
        X = np.load(data_path)
        logger.info(f"Loaded data with shape {X.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Split data into train and test sets
    X_train, X_test = train_test_split(
        X, 
        test_size=args.test_size,
        random_state=args.seed
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    model = train_isolation_forest(
        X_train=X_train,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    # Find optimal threshold if requested
    threshold = None
    if args.optimize_threshold:
        threshold = find_optimal_threshold(
            model=model,
            X_train=X_train,
            target_rate=args.target_rate
        )
    
    # Evaluate model
    metrics, scores = evaluate_model(
        model=model,
        X_test=X_test,
        X_train=X_train,
        threshold=threshold
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save model
    save_model(
        model=model,
        model_path=model_path,
        threshold=threshold
    )
    
    logger.info("Training and evaluation completed successfully")

if __name__ == "__main__":
    main()