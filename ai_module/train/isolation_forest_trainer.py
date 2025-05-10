#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Training for Anomaly Detection

This script trains an Isolation Forest model on latent representations
from the autoencoder to detect anomalies in health data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import argparse
import numpy as np
import pickle
import json
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, List
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
MODELS_DIR = Path(__file__).parent.parent / "models"
LATENTS_DIR = Path("data/processed/latents")
SCORES_DIR = Path("data/processed/scores")


def load_latent_vectors(latent_path: Union[str, Path]) -> np.ndarray:
    """
    Load latent vectors from file
    
    Args:
        latent_path: Path to latent vectors file
        
    Returns:
        numpy.ndarray: Latent vectors
    """
    latent_path = Path(latent_path)
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent vectors file not found: {latent_path}")
    
    try:
        latent_vectors = np.load(latent_path)
        logger.info(f"Loaded latent vectors with shape {latent_vectors.shape} from {latent_path}")
        return latent_vectors
    except Exception as e:
        logger.error(f"Error loading latent vectors: {e}")
        raise


def train_isolation_forest(
    X_train: np.ndarray, 
    n_estimators: int = 100, 
    contamination: float = 0.1, 
    random_state: int = 42,
    verbose: int = 1,
    n_jobs: int = -1,
    max_samples: Union[str, int] = 'auto',
    max_features: float = 1.0,
    bootstrap: bool = False
) -> IsolationForest:
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X_train: Training data (latent vectors)
        n_estimators: Number of base estimators (trees)
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        n_jobs: Number of jobs to run in parallel
        max_samples: Number of samples to draw for each tree
        max_features: Number of features to draw for each tree
        bootstrap: Whether to use bootstrap when sampling
        
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
        n_jobs=n_jobs,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        verbose=verbose
    )
    
    model.fit(X_train)
    
    logger.info("Model training completed")
    
    return model


def evaluate_model(
    model: IsolationForest, 
    X_test: np.ndarray, 
    X_train: Optional[np.ndarray] = None, 
    threshold: Optional[float] = None
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate the Isolation Forest model.
    
    Args:
        model: Trained IsolationForest model
        X_test: Test data
        X_train: Training data (for comparing distributions)
        threshold: Custom anomaly threshold (if None, use model's default)
        
    Returns:
        tuple: (evaluation_metrics, anomaly_scores)
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


def find_optimal_threshold(model: IsolationForest, X_train: np.ndarray, target_rate: float = 0.01) -> float:
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


def save_model(model: IsolationForest, model_path: Union[str, Path], threshold: Optional[float] = None) -> None:
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
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")


def compute_anomaly_scores(
    model: IsolationForest, 
    X: np.ndarray,
    output_dir: Union[str, Path],
    filename: str = "if_anomaly_scores.npy",
    timestamp: bool = False
) -> np.ndarray:
    """
    Compute and save anomaly scores
    
    Args:
        model: Trained Isolation Forest model
        X: Input data (latent vectors)
        output_dir: Directory to save scores
        filename: Filename for scores
        timestamp: Whether to add timestamp to filename
        
    Returns:
        numpy.ndarray: Anomaly scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename if requested
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename.split('.')[0]}_{timestamp_str}.{filename.split('.')[-1]}"
    
    # Compute anomaly scores (negative of decision function)
    # In Isolation Forest, lower decision values = more anomalous
    # We invert to make higher scores = more anomalous (for consistency)
    scores = -model.decision_function(X)
    
    # Save scores
    output_path = output_dir / filename
    np.save(output_path, scores)
    logger.info(f"Saved anomaly scores with shape {scores.shape} to {output_path}")
    
    return scores


def save_model_metadata(
    model: IsolationForest, 
    metadata_path: Union[str, Path],
    latent_dim: Optional[int] = None,
    threshold: Optional[float] = None
) -> None:
    """
    Save model metadata
    
    Args:
        model: Trained Isolation Forest model
        metadata_path: Path to save metadata
        latent_dim: Dimension of latent space
        threshold: Anomaly threshold (if applicable)
    """
    metadata_path = Path(metadata_path)
    
    # Extract model parameters
    params = model.get_params()
    
    # Create metadata dictionary
    metadata = {
        "model_type": "isolation_forest",
        "n_estimators": params["n_estimators"],
        "contamination": params["contamination"],
        "max_samples": str(params["max_samples"]),
        "max_features": params["max_features"],
        "bootstrap": params["bootstrap"],
        "latent_dim": latent_dim,
        "anomaly_threshold": threshold,
        "creation_time": str(datetime.now())
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved model metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        raise


def main():
    """
    Main entry point for training Isolation Forest
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Isolation Forest for anomaly detection")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected proportion of outliers")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--latent-path", type=str, help="Path to latent vectors file")
    parser.add_argument("--target-rate", type=float, default=0.01, help="Target anomaly rate for threshold")
    parser.add_argument("--optimize-threshold", action="store_true", help="Optimize anomaly threshold")
    parser.add_argument("--bootstrap", action="store_true", help="Use bootstrap when sampling")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to saved files")
    args = parser.parse_args()
    
    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine latent path
    latent_path = Path(args.latent_path) if args.latent_path else LATENTS_DIR / "latents.npy"
    
    # Load latent vectors
    try:
        latent_vectors = load_latent_vectors(latent_path)
    except FileNotFoundError:
        logger.error(f"Latent vectors not found at {latent_path}. Please run train_autoencoder.py first.")
        return
    
    # Split data into training and test sets
    X_train, X_test = train_test_split(
        latent_vectors, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")
    
    # Train model
    model = train_isolation_forest(
        X_train=X_train,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
        bootstrap=args.bootstrap
    )
    
    # Find optimal threshold if requested
    threshold = None
    if args.optimize_threshold:
        threshold = find_optimal_threshold(model, X_train, target_rate=args.target_rate)
    
    # Evaluate model
    metrics, _ = evaluate_model(model, X_test, X_train, threshold=threshold)
    
    # Save model
    model_path = MODELS_DIR / "isolation_forest.pkl"
    save_model(model, model_path, threshold=threshold)
    
    # Save model metadata
    metadata_path = MODELS_DIR / "isolation_forest_model_meta.json"
    save_model_metadata(model, metadata_path, latent_dim=latent_vectors.shape[1], threshold=threshold)
    
    # Compute and save anomaly scores for all data
    compute_anomaly_scores(
        model=model,
        X=latent_vectors,
        output_dir=SCORES_DIR,
        filename="if_anomaly_scores.npy",
        timestamp=args.timestamp
    )
    
    logger.info("Isolation Forest training completed successfully")


if __name__ == "__main__":
    main()