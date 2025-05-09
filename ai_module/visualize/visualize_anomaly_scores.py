#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Utilities for Anomaly Detection Results

This module provides functions to visualize results from anomaly detection models,
such as histograms of anomaly scores, reconstruction errors, and comparative analyses.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
REPORTS_DIR = Path("reports/plots")
PROCESSED_DIR = Path("data/processed")
SCORES_DIR = PROCESSED_DIR / "scores"
LATENTS_DIR = PROCESSED_DIR / "latents"
LABELS_DIR = PROCESSED_DIR / "labels"


def setup_plot_style() -> None:
    """
    Set up the matplotlib style for consistent visualizations
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20


def load_numpy_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load data from a numpy file
    
    Args:
        file_path: Path to the numpy file
        
    Returns:
        numpy.ndarray: Loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = np.load(file_path)
        logger.info(f"Loaded data with shape {data.shape} from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def plot_anomaly_histogram(
    scores: np.ndarray,
    title: str = "Anomaly Score Distribution",
    threshold: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    bins: int = 50,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[str]:
    """
    Plot histogram of anomaly scores
    
    Args:
        scores: Anomaly scores
        title: Plot title
        threshold: Anomaly threshold to show in the plot
        labels: True labels (0=normal, 1=anomaly) if available
        bins: Number of histogram bins
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        str: Path to the saved plot if output_path provided, None otherwise
    """
    # Set up plot style
    setup_plot_style()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # If labels are provided, create separate histograms
    if labels is not None and len(labels) == len(scores):
        # Get scores for normal and anomaly samples
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        # Plot histograms
        sns.histplot(normal_scores, bins=bins, alpha=0.6, label="Normal", color="blue", kde=True)
        sns.histplot(anomaly_scores, bins=bins, alpha=0.6, label="Anomaly", color="red", kde=True)
        
        # Add statistics to title
        title = f"{title}\nNormal: μ={normal_scores.mean():.4f}, σ={normal_scores.std():.4f} | Anomaly: μ={anomaly_scores.mean():.4f}, σ={anomaly_scores.std():.4f}"
    else:
        # Plot single histogram
        sns.histplot(scores, bins=bins, kde=True)
        
        # Add statistics to title
        title = f"{title}\nμ={scores.mean():.4f}, σ={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}"
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold = {threshold:.4f}")
        
        # If we have labels, calculate metrics at threshold
        if labels is not None:
            # Predict anomalies based on threshold
            pred_anomalies = (scores > threshold).astype(int)
            
            # Calculate true/false positives/negatives
            tp = np.sum((pred_anomalies == 1) & (labels == 1))
            fp = np.sum((pred_anomalies == 1) & (labels == 0))
            tn = np.sum((pred_anomalies == 0) & (labels == 0))
            fn = np.sum((pred_anomalies == 0) & (labels == 1))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add metrics to legend
            plt.text(
                0.98, 0.02, 
                f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}", 
                transform=plt.gca().transAxes, 
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )
    
    # Set labels and title
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(title)
    
    # Add legend if needed
    if labels is not None or threshold is not None:
        plt.legend()
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return the path to the saved plot
    return str(output_path) if output_path else None


def plot_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 5,
    feature_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (15, 10)
) -> Optional[str]:
    """
    Plot original vs reconstructed samples to visualize reconstruction error
    
    Args:
        original: Original feature vectors
        reconstructed: Reconstructed feature vectors
        sample_indices: Indices of specific samples to plot (optional)
        n_samples: Number of random samples to plot if indices not provided
        feature_names: Names of features for x-axis
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        str: Path to the saved plot if output_path provided, None otherwise
    """
    # Verify input dimensions
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    # Select samples to plot
    if sample_indices is None:
        # Randomly select samples
        sample_indices = np.random.choice(len(original), min(n_samples, len(original)), replace=False)
    else:
        # Use provided indices, ensure they're valid
        sample_indices = [i for i in sample_indices if 0 <= i < len(original)]
        if not sample_indices:
            raise ValueError("No valid sample indices provided")
    
    # Get default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(original.shape[1])]
    
    # Set up plot style
    setup_plot_style()
    
    # Create subplot grid
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    
    # Handle the case of a single sample (axes not array)
    if n_rows == 1:
        axes = [axes]
    
    # Plot each sample
    for i, (ax, idx) in enumerate(zip(axes, sample_indices)):
        # Get original and reconstructed data for this sample
        orig = original[idx]
        recon = reconstructed[idx]
        
        # Calculate reconstruction error
        mse = np.mean((orig - recon) ** 2)
        
        # Plot original values
        ax.plot(orig, 'b-', marker='o', label='Original')
        
        # Plot reconstructed values
        ax.plot(recon, 'r-', marker='x', label='Reconstructed')
        
        # Add legend and title
        if i == 0:
            ax.legend(loc='upper right')
        ax.set_title(f"Sample {idx}: MSE = {mse:.4f}")
        
        # Set feature names on x-axis for bottom subplot
        if i == n_rows - 1:
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    # Add overall title
    plt.suptitle("Original vs Reconstructed Samples", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reconstruction error plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return the path to the saved plot
    return str(output_path) if output_path else None


def compare_anomaly_scores(
    scores_dict: Dict[str, np.ndarray],
    title: str = "Comparison of Anomaly Detection Methods",
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (15, 10)
) -> Optional[str]:
    """
    Compare anomaly scores from different methods
    
    Args:
        scores_dict: Dictionary mapping method names to score arrays
        title: Plot title
        labels: True labels (0=normal, 1=anomaly) if available
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        str: Path to the saved plot if output_path provided, None otherwise
    """
    # Check that we have scores to compare
    if not scores_dict:
        raise ValueError("No scores provided for comparison")
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Create DataFrame for correlation analysis
    scores_df = pd.DataFrame(scores_dict)
    
    # Add labels if available
    if labels is not None and len(labels) == len(next(iter(scores_dict.values()))):
        scores_df['Label'] = labels
    
    # Plot 1: Heatmap of correlations between methods
    corr = scores_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Correlation Between Methods")
    
    # Plot 2: Pairplot or scatter if we have 2 methods
    if len(scores_dict) == 2:
        method1, method2 = list(scores_dict.keys())
        
        # Create scatter plot with labels if available
        if labels is not None:
            # Split by label
            normal_mask = (labels == 0)
            anomaly_mask = (labels == 1)
            
            # Plot normal points
            axes[1].scatter(
                scores_dict[method1][normal_mask], 
                scores_dict[method2][normal_mask], 
                alpha=0.6, 
                label="Normal", 
                color="blue"
            )
            
            # Plot anomaly points
            axes[1].scatter(
                scores_dict[method1][anomaly_mask], 
                scores_dict[method2][anomaly_mask], 
                alpha=0.6, 
                label="Anomaly", 
                color="red"
            )
            
            axes[1].legend()
        else:
            # Simple scatter without labels
            axes[1].scatter(
                scores_dict[method1], 
                scores_dict[method2], 
                alpha=0.6
            )
        
        # Set labels
        axes[1].set_xlabel(method1)
        axes[1].set_ylabel(method2)
        axes[1].set_title("Scatter Plot of Anomaly Scores")
    else:
        # For 3+ methods, show distribution plots
        for method, scores in scores_dict.items():
            sns.kdeplot(scores, label=method, ax=axes[1])
        
        axes[1].set_xlabel("Anomaly Score")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Distribution of Anomaly Scores")
        axes[1].legend()
    
    # Set overall title
    plt.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return the path to the saved plot
    return str(output_path) if output_path else None


def main():
    """
    Main entry point for visualization
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize anomaly detection results")
    parser.add_argument("--scores-path", type=str, help="Path to anomaly scores file")
    parser.add_argument("--labels-path", type=str, help="Path to labels file (optional)")
    parser.add_argument("--threshold", type=float, help="Anomaly threshold to use (optional)")
    parser.add_argument("--model-type", type=str, choices=["autoencoder", "isolation_forest", "combined"], 
                      default="combined", help="Model type to visualize (default: combined)")
    parser.add_argument("--output-dir", type=str, default="reports/plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filenames")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filenames if requested
    timestamp_str = ""
    if args.timestamp:
        timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Determine scores paths
    if args.scores_path:
        # Use provided path
        scores_paths = {args.model_type: Path(args.scores_path)}
    else:
        # Use default paths based on model type
        if args.model_type == "combined":
            # Look for both types of scores
            scores_paths = {
                "autoencoder": SCORES_DIR / "ae_reconstruction_errors.npy",
                "isolation_forest": SCORES_DIR / "if_anomaly_scores.npy"
            }
        else:
            # Look for specific model scores
            if args.model_type == "autoencoder":
                scores_paths = {"autoencoder": SCORES_DIR / "ae_reconstruction_errors.npy"}
            else:  # isolation_forest
                scores_paths = {"isolation_forest": SCORES_DIR / "if_anomaly_scores.npy"}
    
    # Load scores
    scores_dict = {}
    for model, path in scores_paths.items():
        try:
            scores_dict[model] = load_numpy_file(path)
        except FileNotFoundError:
            logger.warning(f"Could not find scores for {model} at {path}")
    
    if not scores_dict:
        logger.error("No scores found. Please run model training/evaluation first.")
        return
    
    # Load labels if path provided
    labels = None
    if args.labels_path:
        try:
            labels = load_numpy_file(args.labels_path)
            logger.info(f"Loaded labels with shape {labels.shape}")
        except FileNotFoundError:
            logger.warning(f"Labels file not found: {args.labels_path}")
    
    # Create output paths
    output_paths = {}
    for model in scores_dict.keys():
        output_paths[model] = output_dir / f"{model}_histogram{timestamp_str}.png"
    
    # Plot histograms for each model
    for model, scores in scores_dict.items():
        plot_anomaly_histogram(
            scores=scores,
            title=f"{model.replace('_', ' ').title()} Anomaly Score Distribution",
            threshold=args.threshold,
            labels=labels,
            output_path=output_paths[model] if not args.show else None,
            show_plot=args.show
        )
    
    # If we have multiple models, compare them
    if len(scores_dict) > 1:
        comparison_path = output_dir / f"anomaly_score_comparison{timestamp_str}.png"
        compare_anomaly_scores(
            scores_dict=scores_dict,
            title="Comparison of Anomaly Detection Methods",
            labels=labels,
            output_path=comparison_path if not args.show else None,
            show_plot=args.show
        )
    
    logger.info("Visualization completed successfully")


if __name__ == "__main__":
    main()