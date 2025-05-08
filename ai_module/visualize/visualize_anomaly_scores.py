#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Tools for Health Data Anomalies

This script visualizes the distribution of anomaly scores and latent space
for a health anomaly detection system using AutoEncoder + Isolation Forest.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

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

def load_data(scores_path, labels_path, latents_path):
    """
    Load anomaly scores, labels, and latent vectors.
    
    Args:
        scores_path: Path to anomaly scores
        labels_path: Path to anomaly labels
        latents_path: Path to latent vectors
        
    Returns:
        tuple: (scores, labels, latents)
    """
    try:
        # Load anomaly scores
        logger.info(f"Loading anomaly scores from {scores_path}")
        scores = np.load(scores_path)
        logger.info(f"Loaded scores with shape {scores.shape}")
        
        # Load anomaly labels
        logger.info(f"Loading anomaly labels from {labels_path}")
        labels = np.load(labels_path)
        logger.info(f"Loaded labels with shape {labels.shape}")
        
        # Load latent vectors
        logger.info(f"Loading latent vectors from {latents_path}")
        latents = np.load(latents_path)
        logger.info(f"Loaded latent vectors with shape {latents.shape}")
        
        # Validate data
        if len(scores) != len(labels) or len(scores) != len(latents):
            logger.warning(f"Data size mismatch: scores={len(scores)}, labels={len(labels)}, latents={len(latents)}")
        
        return scores, labels, latents
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def reduce_dimensions(latents, method='pca', random_state=42):
    """
    Reduce latent space to 2D using PCA or t-SNE.
    
    Args:
        latents: Latent vectors
        method: Dimensionality reduction method ('pca' or 'tsne')
        random_state: Random seed for reproducibility
        
    Returns:
        numpy.ndarray: 2D representation of latent space
    """
    try:
        logger.info(f"Reducing {latents.shape[1]} dimensions to 2D using {method.upper()}")
        
        if method.lower() == 'tsne':
            # t-SNE for non-linear dimensionality reduction
            reducer = TSNE(
                n_components=2,
                random_state=random_state,
                n_jobs=-1,  # Use all available cores
                verbose=1
            )
            
            # Warn about t-SNE computation time for large datasets
            if len(latents) > 5000:
                logger.warning(f"t-SNE may be slow for {len(latents)} samples")
        else:
            # PCA for linear dimensionality reduction (default)
            reducer = PCA(n_components=2, random_state=random_state)
        
        # Perform dimensionality reduction
        reduced = reducer.fit_transform(latents)
        
        logger.info(f"Reduced dimensions to shape {reduced.shape}")
        
        # For PCA, log explained variance
        if method.lower() == 'pca':
            explained_variance = reducer.explained_variance_ratio_.sum() * 100
            logger.info(f"PCA explained variance: {explained_variance:.2f}%")
        
        return reduced
    
    except Exception as e:
        logger.error(f"Error reducing dimensions: {e}")
        sys.exit(1)

def plot_histogram(scores, labels, output_dir, threshold=None, bins=50, timestamp=False, save_only=False):
    """
    Create and save histogram of anomaly scores.
    
    Args:
        scores: Anomaly scores
        labels: Anomaly labels
        output_dir: Directory to save plot
        threshold: Anomaly threshold (optional)
        bins: Number of histogram bins
        timestamp: Whether to add timestamp to filename
        save_only: Whether to only save plot without displaying
        
    Returns:
        str: Path to saved figure
    """
    try:
        # Generate timestamp suffix if requested
        suffix = ""
        if timestamp:
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Plot histogram
        sns.histplot(scores, bins=bins, kde=True, color="skyblue")
        
        # If threshold not provided, try to infer from labels
        if threshold is None and labels is not None:
            # Find minimum score for anomaly samples
            anomaly_scores = scores[labels == 1]
            if len(anomaly_scores) > 0:
                threshold = min(anomaly_scores)
                logger.info(f"Inferred threshold from labels: {threshold:.4f}")
        
        # Add threshold line if available
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold:.4f}')
            
            # Shade anomalous region
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            x_max = max(scores) * 1.1  # Add some padding
            plt.fill_between([threshold, x_max], 0, y_max, 
                            color='red', alpha=0.1, label='Anomalous Region')
            
            # Add annotation for anomaly count and percentage
            if labels is not None:
                anomaly_count = np.sum(labels == 1)
                anomaly_percent = 100 * anomaly_count / len(labels)
                plt.text(threshold * 1.05, y_max * 0.9, 
                        f"Anomalies: {anomaly_count} ({anomaly_percent:.2f}%)",
                        color='red', fontsize=12)
        
        # Add mean and median lines
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        plt.axvline(x=mean_score, color='green', linestyle='-', 
                   label=f'Mean: {mean_score:.4f}')
        plt.axvline(x=median_score, color='purple', linestyle='-.', 
                   label=f'Median: {median_score:.4f}')
        
        # Add labels and title
        plt.xlabel('Anomaly Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Anomaly Scores', fontsize=16)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"hist_anomaly_scores{suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved histogram to {output_path}")
        
        # Show figure if requested
        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error plotting histogram: {e}")
        return None

def plot_scatter(latents, labels, output_dir, method='pca', timestamp=False, save_only=False):
    """
    Create and save scatter plot of 2D latent features.
    
    Args:
        latents: Latent vectors
        labels: Anomaly labels
        output_dir: Directory to save plot
        method: Dimensionality reduction method ('pca' or 'tsne')
        timestamp: Whether to add timestamp to filename
        save_only: Whether to only save plot without displaying
        
    Returns:
        str: Path to saved figure
    """
    try:
        # Generate timestamp suffix if requested
        suffix = ""
        if timestamp:
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Reduce dimensions to 2D
        reduced_latents = reduce_dimensions(latents, method=method)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Create custom colormap
        colors = np.array(['#1f77b4', '#d62728'])  # Blue for normal, red for anomaly
        cmap = ListedColormap(colors)
        
        # Plot scatter
        scatter = plt.scatter(
            reduced_latents[:, 0],
            reduced_latents[:, 1],
            c=labels,
            cmap=cmap,
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidths=0.5
        )
        
        # Calculate stats for legend
        normal_count = np.sum(labels == 0)
        anomaly_count = np.sum(labels == 1)
        total_count = len(labels)
        
        # Add legend
        legend_labels = [
            f'Normal ({normal_count}, {100*normal_count/total_count:.1f}%)',
            f'Anomaly ({anomaly_count}, {100*anomaly_count/total_count:.1f}%)'
        ]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
        
        # Add labels and title
        method_name = 'PCA' if method.lower() == 'pca' else 't-SNE'
        plt.xlabel(f'{method_name} Component 1', fontsize=14)
        plt.ylabel(f'{method_name} Component 2', fontsize=14)
        plt.title(f'2D Projection of Latent Space using {method_name}', fontsize=16)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"scatter_latent_space_{method.lower()}{suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {output_path}")
        
        # Show figure if requested
        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error plotting scatter: {e}")
        return None

def plot_kde(latents, labels, output_dir, method='pca', timestamp=False, save_only=False):
    """
    Create and save KDE plot of anomaly density in latent space.
    
    Args:
        latents: Latent vectors
        labels: Anomaly labels
        output_dir: Directory to save plot
        method: Dimensionality reduction method ('pca' or 'tsne')
        timestamp: Whether to add timestamp to filename
        save_only: Whether to only save plot without displaying
        
    Returns:
        str: Path to saved figure
    """
    try:
        # Generate timestamp suffix if requested
        suffix = ""
        if timestamp:
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Reduce dimensions to 2D
        reduced_latents = reduce_dimensions(latents, method=method)
        
        # Only proceed if we have anomalies
        if np.sum(labels == 1) == 0:
            logger.warning("No anomalies found, skipping KDE plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Create dataframe for Seaborn
        import pandas as pd
        df = pd.DataFrame({
            'Component 1': reduced_latents[:, 0],
            'Component 2': reduced_latents[:, 1],
            'Class': ['Normal' if label == 0 else 'Anomaly' for label in labels]
        })
        
        # Plot KDE for normal samples
        sns.kdeplot(
            data=df[df['Class'] == 'Normal'],
            x='Component 1',
            y='Component 2',
            fill=True,
            alpha=0.5,
            levels=5,
            cmap="Blues",
            ax=ax,
            label='Normal'
        )
        
        # Plot KDE for anomaly samples
        sns.kdeplot(
            data=df[df['Class'] == 'Anomaly'],
            x='Component 1',
            y='Component 2',
            fill=True,
            alpha=0.5,
            levels=5,
            cmap="Reds",
            ax=ax,
            label='Anomaly'
        )
        
        # Add scatter points
        sns.scatterplot(
            data=df,
            x='Component 1',
            y='Component 2',
            hue='Class',
            palette={'Normal': 'blue', 'Anomaly': 'red'},
            alpha=0.7,
            s=30,
            edgecolor='w',
            linewidth=0.5,
            ax=ax
        )
        
        # Add labels and title
        method_name = 'PCA' if method.lower() == 'pca' else 't-SNE'
        plt.xlabel(f'{method_name} Component 1', fontsize=14)
        plt.ylabel(f'{method_name} Component 2', fontsize=14)
        plt.title(f'Density Distribution in Latent Space using {method_name}', fontsize=16)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Class")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"anomaly_kde_{method.lower()}{suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved KDE plot to {output_path}")
        
        # Show figure if requested
        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error plotting KDE: {e}")
        return None

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Visualize anomaly scores and latent space")
    
    parser.add_argument(
        "--latents-path",
        type=str,
        default="data/processed/latent_vectors/latents.npy",
        help="Path to latent vectors file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--scores-path",
        type=str,
        default="data/processed/latent_vectors/anomaly_scores.npy",
        help="Path to anomaly scores file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/processed/latent_vectors/anomaly_labels.npy",
        help="Path to anomaly labels file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/visuals",
        help="Directory to save visualizations (default: %(default)s)"
    )
    
    parser.add_argument(
        "--use-tsne",
        action="store_true",
        help="Use t-SNE instead of PCA for dimensionality reduction"
    )
    
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots without displaying them"
    )
    
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output filenames"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for anomaly scores (optional)"
    )
    
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histogram (default: %(default)s)"
    )
    
    parser.add_argument(
        "--no-kde",
        action="store_true",
        help="Skip KDE plot generation"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to load data and create visualizations.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = ensure_dir_exists(Path(args.output_dir))
    
    # Load data
    scores, labels, latents = load_data(
        scores_path=args.scores_path,
        labels_path=args.labels_path,
        latents_path=args.latents_path
    )
    
    # Create histogram
    plot_histogram(
        scores=scores,
        labels=labels,
        output_dir=output_dir,
        threshold=args.threshold,
        bins=args.bins,
        timestamp=args.timestamp,
        save_only=args.save_only
    )
    
    # Create scatter plot
    method = 'tsne' if args.use_tsne else 'pca'
    plot_scatter(
        latents=latents,
        labels=labels,
        output_dir=output_dir,
        method=method,
        timestamp=args.timestamp,
        save_only=args.save_only
    )
    
    # Create KDE plot if requested
    if not args.no_kde:
        plot_kde(
            latents=latents,
            labels=labels,
            output_dir=output_dir,
            method=method,
            timestamp=args.timestamp,
            save_only=args.save_only
        )
    
    logger.info("Visualization generation complete.")

if __name__ == "__main__":
    main()