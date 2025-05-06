# TODO: Implement this module

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Evaluation for ECG Signal Anomaly Detection

This script evaluates the trained Isolation Forest model by analyzing anomaly scores,
comparing them with reconstruction errors, and generating performance metrics
and visualizations.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import pearsonr, spearmanr

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


def load_latent_vectors():
    """
    Load all latent vector files from the latents directory.
    
    Returns:
        tuple: (feature_matrix, file_names)
    """
    try:
        # Get all .npy files in the latents directory
        latent_files = list(LATENTS_DIR.glob("*.npy"))
        
        if not latent_files:
            logger.error(f"No latent vector files found in {LATENTS_DIR}")
            return None, None
        
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


def load_anomaly_scores():
    """
    Load anomaly scores from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly scores and metadata
    """
    try:
        # Define path to anomaly scores CSV
        scores_path = REPORTS_DIR / "anomaly_scores.csv"
        
        # Check if file exists
        if not scores_path.exists():
            logger.error(f"Anomaly scores file not found: {scores_path}")
            return None
        
        # Load CSV into DataFrame
        df = pd.read_csv(scores_path)
        
        logger.info(f"Loaded anomaly scores for {len(df)} samples")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading anomaly scores: {e}")
        return None


def analyze_score_distribution(scores_df):
    """
    Analyze the distribution of anomaly scores.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores
        
    Returns:
        dict: Dictionary with distribution statistics
    """
    try:
        # Extract anomaly scores
        anomaly_scores = scores_df['anomaly_score'].values
        
        # Calculate statistics
        stats = {
            'mean': np.mean(anomaly_scores),
            'median': np.median(anomaly_scores),
            'std': np.std(anomaly_scores),
            'min': np.min(anomaly_scores),
            'max': np.max(anomaly_scores),
            'q1': np.percentile(anomaly_scores, 25),
            'q3': np.percentile(anomaly_scores, 75),
            'iqr': np.percentile(anomaly_scores, 75) - np.percentile(anomaly_scores, 25)
        }
        
        logger.info("Analyzed anomaly score distribution")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing score distribution: {e}")
        return None


def analyze_correlation(scores_df):
    """
    Analyze correlation between anomaly scores and reconstruction errors.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores and reconstruction errors
        
    Returns:
        dict: Dictionary with correlation statistics
    """
    try:
        # Check if reconstruction error column exists
        if 'reconstruction_error' not in scores_df.columns:
            logger.warning("Reconstruction error column not found in DataFrame")
            return None
        
        # Extract data
        anomaly_scores = scores_df['anomaly_score'].values
        recon_errors = scores_df['reconstruction_error'].values
        
        # Calculate Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(anomaly_scores, recon_errors)
        
        # Calculate Spearman correlation (rank-based, robust to outliers)
        spearman_corr, spearman_p = spearmanr(anomaly_scores, recon_errors)
        
        # Store results
        correlations = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p
        }
        
        logger.info(f"Calculated correlations: Pearson={pearson_corr:.4f} (p={pearson_p:.4f}), "
                   f"Spearman={spearman_corr:.4f} (p={spearman_p:.4f})")
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error analyzing correlation: {e}")
        return None


def compute_classification_metrics(scores_df):
    """
    Compute classification metrics if ground truth is available.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores and true labels
        
    Returns:
        dict: Dictionary with classification metrics
    """
    try:
        # Check if is_anomaly column exists and contains valid values
        if 'is_anomaly' not in scores_df.columns:
            logger.warning("is_anomaly column not found in DataFrame")
            return None
        
        # Extract ground truth labels and predictions
        y_true = scores_df['is_anomaly'].values
        
        # Convert to integer if needed
        if y_true.dtype != np.int64 and y_true.dtype != np.int32:
            y_true = y_true.astype(int)
        
        # Check if we have enough samples of each class
        n_anomalies = np.sum(y_true == 1)
        n_normal = np.sum(y_true == 0)
        
        if n_anomalies == 0 or n_normal == 0:
            logger.warning(f"Unbalanced classes: {n_normal} normal, {n_anomalies} anomalies")
            if n_anomalies == 0 or n_normal == 0:
                logger.error("Cannot compute classification metrics with only one class")
                return None
        
        # Calculate classification report
        report = classification_report(y_true, y_true, output_dict=True)
        
        # Store results
        metrics = {
            'classification_report': report,
            'n_samples': len(y_true),
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_ratio': n_anomalies / len(y_true) if len(y_true) > 0 else 0
        }
        
        logger.info(f"Computed classification metrics for {len(y_true)} samples "
                   f"({n_anomalies} anomalies, {n_normal} normal)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return None


def plot_score_histogram(scores_df, output_dir):
    """
    Plot histogram of anomaly scores.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores
        output_dir (Path): Directory to save plot
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(
            data=scores_df,
            x='anomaly_score',
            kde=True,
            bins=30,
            color='skyblue',
            edgecolor='black',
            line_kws={'linewidth': 2}
        )
        
        # Add vertical line for mean
        mean_score = scores_df['anomaly_score'].mean()
        plt.axvline(
            x=mean_score,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_score:.4f}'
        )
        
        # If is_anomaly column exists, color by class
        if 'is_anomaly' in scores_df.columns:
            # Get anomaly threshold (if available)
            anomaly_threshold = scores_df.loc[
                scores_df['is_anomaly'] == 1, 'anomaly_score'
            ].min()
            
            plt.axvline(
                x=anomaly_threshold,
                color='green',
                linestyle='-.',
                linewidth=2,
                label=f'Threshold: {anomaly_threshold:.4f}'
            )
        
        # Add labels and title
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Isolation Forest Anomaly Scores', fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "isolation_forest_score_histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved anomaly score histogram to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting score histogram: {e}")
        return None


def plot_correlation_scatter(scores_df, output_dir):
    """
    Plot scatter plot of anomaly scores vs reconstruction errors.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores and reconstruction errors
        output_dir (Path): Directory to save plot
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Check if reconstruction error column exists
        if 'reconstruction_error' not in scores_df.columns:
            logger.warning("Reconstruction error column not found in DataFrame")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # If is_anomaly column exists, color by class
        if 'is_anomaly' in scores_df.columns:
            # Create categorical palette
            palette = {0: 'blue', 1: 'red'}
            hue_column = 'is_anomaly'
            hue_order = [0, 1]
            legend_labels = ['Normal', 'Anomaly']
            
            # Plot scatter with classes
            scatter = sns.scatterplot(
                data=scores_df,
                x='reconstruction_error',
                y='anomaly_score',
                hue=hue_column,
                hue_order=hue_order,
                palette=palette,
                alpha=0.7,
                s=80
            )
            
            # Set legend labels
            handles, _ = scatter.get_legend_handles_labels()
            plt.legend(handles, legend_labels, title='Class')
            
        else:
            # Plot regular scatter
            sns.scatterplot(
                data=scores_df,
                x='reconstruction_error',
                y='anomaly_score',
                alpha=0.7,
                color='blue',
                s=80
            )
        
        # Calculate correlation for the plot title
        pearson_corr, _ = pearsonr(
            scores_df['reconstruction_error'].values,
            scores_df['anomaly_score'].values
        )
        
        # Add regression line
        sns.regplot(
            data=scores_df,
            x='reconstruction_error',
            y='anomaly_score',
            scatter=False,
            color='green',
            line_kws={'linewidth': 2}
        )
        
        # Add labels and title
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Isolation Forest Anomaly Score', fontsize=12)
        plt.title(
            f'Correlation between Reconstruction Error and Anomaly Score (r={pearson_corr:.4f})',
            fontsize=14
        )
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "isolation_forest_correlation_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation scatter plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting correlation scatter: {e}")
        return None


def plot_confusion_matrix(scores_df, output_dir):
    """
    Plot confusion matrix if ground truth is available.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with anomaly scores and true labels
        output_dir (Path): Directory to save plot
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Check if is_anomaly column exists
        if 'is_anomaly' not in scores_df.columns:
            logger.warning("is_anomaly column not found in DataFrame")
            return None
        
        # Extract ground truth labels (assuming they are reliable)
        y_true = scores_df['is_anomaly'].values
        
        # Convert to integer if needed
        if y_true.dtype != np.int64 and y_true.dtype != np.int32:
            y_true = y_true.astype(int)
        
        # Check if we have enough samples of each class
        n_anomalies = np.sum(y_true == 1)
        n_normal = np.sum(y_true == 0)
        
        if n_anomalies == 0 or n_normal == 0:
            logger.warning(f"Unbalanced classes: {n_normal} normal, {n_anomalies} anomalies")
            if n_anomalies == 0 or n_normal == 0:
                logger.error("Cannot create confusion matrix with only one class")
                return None
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_true)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Normal', 'Anomaly']
        )
        disp.plot(cmap='Blues', values_format='d')
        
        # Add title
        plt.title('Confusion Matrix for Isolation Forest', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "isolation_forest_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        return None


def save_metrics_report(stats, correlations, classification_metrics, plot_paths):
    """
    Save metrics report to text file.
    
    Args:
        stats (dict): Dictionary with distribution statistics
        correlations (dict): Dictionary with correlation statistics
        classification_metrics (dict): Dictionary with classification metrics
        plot_paths (list): List of paths to saved plots
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        # Define output path
        output_path = REPORTS_DIR / "eval_isolation_forest.txt"
        
        # Create report content
        report_lines = [
            "======================================================",
            "            ISOLATION FOREST EVALUATION REPORT         ",
            "======================================================",
            ""
        ]
        
        # Add distribution statistics
        if stats:
            report_lines.extend([
                "1. ANOMALY SCORE DISTRIBUTION",
                "----------------------------",
                f"Mean: {stats['mean']:.6f}",
                f"Median: {stats['median']:.6f}",
                f"Standard deviation: {stats['std']:.6f}",
                f"Minimum: {stats['min']:.6f}",
                f"Maximum: {stats['max']:.6f}",
                f"Q1 (25th percentile): {stats['q1']:.6f}",
                f"Q3 (75th percentile): {stats['q3']:.6f}",
                f"Interquartile range (IQR): {stats['iqr']:.6f}",
                ""
            ])
        
        # Add correlation statistics
        if correlations:
            report_lines.extend([
                "2. CORRELATION ANALYSIS",
                "------------------------",
                "Correlation between anomaly scores and reconstruction errors:",
                f"Pearson correlation: {correlations['pearson_correlation']:.6f} "
                f"(p-value: {correlations['pearson_p_value']:.6f})",
                f"Spearman correlation: {correlations['spearman_correlation']:.6f} "
                f"(p-value: {correlations['spearman_p_value']:.6f})",
                "",
                "Interpretation:",
                "  - Pearson measures linear correlation (-1 to 1)",
                "  - Spearman measures monotonic relationship (-1 to 1)",
                "  - p-value < 0.05 indicates statistical significance",
                ""
            ])
        
        # Add classification metrics
        if classification_metrics:
            # Extract data
            n_samples = classification_metrics.get('n_samples', 0)
            n_anomalies = classification_metrics.get('n_anomalies', 0)
            n_normal = classification_metrics.get('n_normal', 0)
            anomaly_ratio = classification_metrics.get('anomaly_ratio', 0)
            report = classification_metrics.get('classification_report', {})
            
            report_lines.extend([
                "3. CLASSIFICATION METRICS",
                "-------------------------",
                f"Total samples: {n_samples}",
                f"Normal samples: {n_normal} ({(1-anomaly_ratio)*100:.2f}%)",
                f"Anomaly samples: {n_anomalies} ({anomaly_ratio*100:.2f}%)",
                ""
            ])
            
            # Add classification report if available
            if report:
                # Format precision, recall, and F1 for each class
                if '0' in report:
                    report_lines.extend([
                        "Metrics for Normal class (0):",
                        f"  Precision: {report['0']['precision']:.4f}",
                        f"  Recall: {report['0']['recall']:.4f}",
                        f"  F1-score: {report['0']['f1-score']:.4f}",
                        f"  Support: {report['0']['support']}",
                        ""
                    ])
                
                if '1' in report:
                    report_lines.extend([
                        "Metrics for Anomaly class (1):",
                        f"  Precision: {report['1']['precision']:.4f}",
                        f"  Recall: {report['1']['recall']:.4f}",
                        f"  F1-score: {report['1']['f1-score']:.4f}",
                        f"  Support: {report['1']['support']}",
                        ""
                    ])
                
                # Overall metrics
                if 'accuracy' in report and 'macro avg' in report and 'weighted avg' in report:
                    report_lines.extend([
                        "Overall metrics:",
                        f"  Accuracy: {report['accuracy']:.4f}",
                        f"  Macro avg F1: {report['macro avg']['f1-score']:.4f}",
                        f"  Weighted avg F1: {report['weighted avg']['f1-score']:.4f}",
                        ""
                    ])
        
        # Add plot paths
        if plot_paths:
            report_lines.extend([
                "4. GENERATED PLOTS",
                "-----------------"
            ])
            
            for path in plot_paths:
                if path:
                    report_lines.append(f"- {path}")
            
            report_lines.append("")
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.extend([
            f"Report generated on: {timestamp}",
            "======================================================"
        ])
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved evaluation report to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving metrics report: {e}")
        return False


def main():
    """
    Main function to evaluate the Isolation Forest model.
    """
    try:
        logger.info("Starting Isolation Forest evaluation")
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model, aborting")
            return
        
        # Load latent vectors
        X, file_names = load_latent_vectors()
        if X is None:
            logger.error("Failed to load latent vectors, aborting")
            return
        
        # Load anomaly scores
        scores_df = load_anomaly_scores()
        if scores_df is None:
            logger.error("Failed to load anomaly scores, aborting")
            return
        
        # Analyze score distribution
        stats = analyze_score_distribution(scores_df)
        
        # Analyze correlation with reconstruction errors
        correlations = analyze_correlation(scores_df)
        
        # Compute classification metrics if ground truth is available
        classification_metrics = compute_classification_metrics(scores_df)
        
        # Generate plots
        plot_paths = []
        
        # Plot histogram of anomaly scores
        histogram_path = plot_score_histogram(scores_df, PLOTS_DIR)
        plot_paths.append(histogram_path)
        
        # Plot scatter plot of anomaly scores vs reconstruction errors
        scatter_path = plot_correlation_scatter(scores_df, PLOTS_DIR)
        plot_paths.append(scatter_path)
        
        # Plot confusion matrix if ground truth is available
        if classification_metrics:
            cm_path = plot_confusion_matrix(scores_df, PLOTS_DIR)
            plot_paths.append(cm_path)
        
        # Save metrics report
        save_metrics_report(stats, correlations, classification_metrics, plot_paths)
        
        logger.info("Isolation Forest evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in Isolation Forest evaluation: {e}")


if __name__ == "__main__":
    main()