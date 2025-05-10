#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Utilities

This module provides functions to create standardized visualizations for model
evaluation, anomaly detection, feature importance, and SHAP explanations.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import base64
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, f1_score, accuracy_score, precision_score, recall_score
)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import project modules
from ai_module.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("visualization")

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')

# Define consistent colors
COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',  # Green
    'quaternary': '#d62728',  # Red
    'anomaly': '#d62728',  # Red
    'normal': '#2ca02c',  # Green
    'threshold': '#ff7f0e',  # Orange
    'true_positive': '#2ca02c',  # Green
    'false_positive': '#ff7f0e',  # Orange
    'true_negative': '#1f77b4',  # Blue
    'false_negative': '#d62728',  # Red
}


def figure_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: str = "Anomaly Scores",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot anomaly scores with optional true labels and threshold.
    
    Args:
        scores: Array of anomaly scores
        labels: True labels (1 for anomaly, 0 for normal)
        threshold: Anomaly threshold value
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create index for x-axis
    x = np.arange(len(scores))
    
    # Plot scores with different colors if labels provided
    if labels is not None:
        # Anomaly points
        anomaly_idx = np.where(labels == 1)[0]
        if len(anomaly_idx) > 0:
            ax.scatter(
                anomaly_idx, scores[anomaly_idx],
                color=COLORS['anomaly'], label='Anomaly', alpha=0.7
            )
        
        # Normal points
        normal_idx = np.where(labels == 0)[0]
        if len(normal_idx) > 0:
            ax.scatter(
                normal_idx, scores[normal_idx],
                color=COLORS['normal'], label='Normal', alpha=0.7
            )
    else:
        # Plot all points with the same color
        ax.scatter(x, scores, color=COLORS['primary'], alpha=0.7)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(
            y=threshold, color=COLORS['threshold'],
            linestyle='--', label=f'Threshold: {threshold:.4f}'
        )
    
    # Set labels and title
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(title)
    
    # Add legend if needed
    if labels is not None or threshold is not None:
        ax.legend()
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved anomaly scores plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot original vs. reconstructed features to visualize reconstruction error.
    
    Args:
        original: Original feature values
        reconstructed: Reconstructed feature values
        sample_indices: Indices of samples to include (default: first 5)
        feature_names: Names of features (default: numerical indices)
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Determine which samples to plot
    if sample_indices is None:
        sample_indices = list(range(min(5, original.shape[0])))
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(original.shape[1])]
    
    # Create figure with subplots for each sample
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
    
    # Convert single axis to list for consistent indexing
    if n_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Get data for this sample
        orig = original[idx]
        recon = reconstructed[idx]
        
        # Calculate error
        error = np.abs(orig - recon)
        
        # Create x positions for bars
        x = np.arange(len(feature_names))
        width = 0.35
        
        # Plot original and reconstructed values as grouped bars
        ax.bar(x - width/2, orig, width, label='Original', color=COLORS['primary'])
        ax.bar(x + width/2, recon, width, label='Reconstructed', color=COLORS['secondary'])
        
        # Add error as a line plot
        ax2 = ax.twinx()
        ax2.plot(x, error, 'o-', color=COLORS['anomaly'], label='Error')
        ax2.set_ylabel('Absolute Error')
        
        # Set labels and title
        ax.set_title(f'Sample {idx}')
        ax.set_ylabel('Feature Value')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set x-axis labels on the bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[-1].set_xlabel('Features')
    
    # Add overall title
    plt.suptitle('Feature Reconstruction Comparison', fontsize=16)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reconstruction plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: ['Normal', 'Anomaly'])
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Default labels
    if labels is None:
        labels = ['Normal', 'Anomaly']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count' if not normalize else 'Percentage', rotation=-90, va="bottom")
    
    # Add labels
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    
    # Rotate x labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Add metrics as text
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}"
    )
    ax.text(
        1.05, 0.5, metrics_text,
        transform=ax.transAxes,
        verticalalignment='center'
    )
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels (binary)
        y_score: Predicted probabilities or scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr, lw=2, color=COLORS['primary'],
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot precision-recall curve with average precision score.
    
    Args:
        y_true: True labels (binary)
        y_score: Predicted probabilities or scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(
        recall, precision, lw=2, color=COLORS['secondary'],
        label=f'PR curve (AUC = {pr_auc:.4f})'
    )
    
    # Calculate baseline based on positive class frequency
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', 
               label=f'Baseline (No Skill): {baseline:.4f}')
    
    # Set labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add legend
    ax.legend(loc="lower left")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    top_n: Optional[int] = None,
    color: str = 'primary',
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importance_values: Importance values for each feature
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show (default: all)
        color: Color name from COLORS dict
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Ensure inputs are numpy arrays
    importance_values = np.array(importance_values)
    
    # Sort features by importance
    indices = np.argsort(importance_values)
    
    # Select top N features if specified
    if top_n is not None and top_n < len(feature_names):
        indices = indices[-top_n:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color
    bar_color = COLORS.get(color, COLORS['primary'])
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(indices))
    ax.barh(
        y_pos,
        importance_values[indices],
        align='center',
        color=bar_color,
        alpha=0.8
    )
    
    # Set y tick labels to feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Invert y-axis to show most important at the top
    ax.invert_yaxis()
    
    # Add grid lines
    ax.grid(True, axis='x', alpha=0.3)
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    include_time: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training metrics
        title: Plot title
        figsize: Figure size
        include_time: Whether to include epoch time plot
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Determine number of subplots
    n_plots = 1 + int(include_time and 'epoch_times' in history)
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Convert to list if only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    ax.plot(history.get('train_loss', []), color=COLORS['primary'], label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        ax.plot(history.get('val_loss', []), color=COLORS['secondary'], label='Validation Loss')
    
    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot epoch times if requested
    if include_time and 'epoch_times' in history and len(axes) > 1:
        ax = axes[1]
        ax.plot(history['epoch_times'], color=COLORS['tertiary'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (s)')
        ax.set_title('Epoch Training Time')
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Improve layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    title: str = "SHAP Feature Importance",
    plot_type: str = "bar",
    max_display: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Create SHAP summary plots to explain model predictions.
    
    Args:
        shap_values: SHAP values
        features: Feature values
        feature_names: Names of features
        title: Plot title
        plot_type: Type of plot ('bar', 'dot', or 'violin')
        max_display: Maximum number of features to display
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP package not available. Install with: pip install shap")
        raise ImportError("SHAP package required for this function")
    
    # Create a DataFrame for SHAP
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create SHAP summary plot
    if plot_type == "bar":
        shap.summary_plot(
            shap_values, features_df,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
    elif plot_type == "dot":
        shap.summary_plot(
            shap_values, features_df,
            plot_type="dot",
            max_display=max_display,
            show=False
        )
    elif plot_type == "violin":
        shap.summary_plot(
            shap_values, features_df,
            plot_type="violin",
            max_display=max_display,
            show=False
        )
    else:
        logger.warning(f"Unknown plot_type: {plot_type}. Using 'bar' instead.")
        shap.summary_plot(
            shap_values, features_df,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
    
    # Get current figure
    fig = plt.gcf()
    
    # Add title
    plt.title(title)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_latent_space(
    latent_vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "pca",
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Visualize latent space representations using dimensionality reduction.
    
    Args:
        latent_vectors: Latent vectors from autoencoder
        labels: Labels for coloring points (0 for normal, 1 for anomaly)
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Check latent dimension
    if latent_vectors.shape[1] <= 2:
        # Already 2D or 1D, no need for dimensionality reduction
        components = latent_vectors
    else:
        # Apply dimensionality reduction
        if method.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                logger.warning("UMAP not installed. Falling back to PCA.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                method = "pca"
        else:
            logger.warning(f"Unknown method: {method}. Using PCA instead.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            method = "pca"
        
        # Apply reduction
        components = reducer.fit_transform(latent_vectors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points with different colors if labels provided
    if labels is not None:
        # Anomaly points
        anomaly_idx = np.where(labels == 1)[0]
        if len(anomaly_idx) > 0:
            ax.scatter(
                components[anomaly_idx, 0], components[anomaly_idx, 1],
                color=COLORS['anomaly'], label='Anomaly', alpha=0.7
            )
        
        # Normal points
        normal_idx = np.where(labels == 0)[0]
        if len(normal_idx) > 0:
            ax.scatter(
                components[normal_idx, 0], components[normal_idx, 1],
                color=COLORS['normal'], label='Normal', alpha=0.7
            )
        
        # Add legend
        ax.legend()
    else:
        # Plot all points with the same color
        ax.scatter(
            components[:, 0], components[:, 1],
            color=COLORS['primary'], alpha=0.7
        )
    
    # Set labels and title
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f"{title} ({method.upper()})")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved latent space plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig


def plot_distribution_comparison(
    normal_values: np.ndarray,
    anomaly_values: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Distribution Comparison",
    figsize: Tuple[int, int] = (15, 10),
    max_features: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    as_base64: bool = False
) -> Union[plt.Figure, str]:
    """
    Compare feature distributions between normal and anomaly samples.
    
    Args:
        normal_values: Feature values for normal samples
        anomaly_values: Feature values for anomaly samples
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
        max_features: Maximum number of features to display
        save_path: Path to save the figure
        as_base64: Whether to return the figure as a base64 encoded string
        
    Returns:
        Figure object or base64 encoded string
    """
    # Limit the number of features
    n_features = min(len(feature_names), max_features)
    
    # Create figure
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Convert axes to flattened array
    axes = axes.ravel()
    
    # Plot each feature
    for i in range(n_features):
        ax = axes[i]
        
        # Plot normal distribution
        sns.kdeplot(
            normal_values[:, i],
            ax=ax,
            color=COLORS['normal'],
            label='Normal',
            fill=True,
            alpha=0.3
        )
        
        # Plot anomaly distribution
        sns.kdeplot(
            anomaly_values[:, i],
            ax=ax,
            color=COLORS['anomaly'],
            label='Anomaly',
            fill=True,
            alpha=0.3
        )
        
        # Add boxplots below the distributions
        trans = ax.get_xaxis_transform()
        ax.plot(
            normal_values[:, i],
            np.ones_like(normal_values[:, i]) * 0.1,
            '|',
            color=COLORS['normal'],
            transform=trans,
            alpha=0.5
        )
        ax.plot(
            anomaly_values[:, i],
            np.ones_like(anomaly_values[:, i]) * 0.15,
            '|',
            color=COLORS['anomaly'],
            transform=trans,
            alpha=0.5
        )
        
        # Set title and add legend
        ax.set_title(feature_names[i])
        if i == 0:  # Only add legend to first plot
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution comparison plot to {save_path}")
    
    # Return as base64 if requested
    if as_base64:
        b64_str = figure_to_base64(fig)
        plt.close(fig)
        return b64_str
    
    return fig 