#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Evaluation for Anomaly Detection in ECG Signals

This script evaluates the performance of the Isolation Forest model by
analyzing anomaly scores, creating visualizations, and comparing with
reconstruction errors from the autoencoder.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure proper imports by determining the project root
# Find the ai_module directory in the path
file_path = Path(__file__).resolve()
module_dir = None
for parent in file_path.parents:
    if parent.name == "ai_module":
        module_dir = parent
        break
    
if module_dir is None:
    raise ImportError("Cannot find ai_module directory in path hierarchy")

# Constants and paths
REPORTS_DIR = module_dir / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
ANOMALY_SCORES_PATH = REPORTS_DIR / "anomaly_scores.csv"

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_anomaly_scores():
    """
    Load anomaly scores from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly scores or None if loading fails
    """
    try:
        logger.info(f"Loading anomaly scores from {ANOMALY_SCORES_PATH}")
        
        # Check if file exists
        if not ANOMALY_SCORES_PATH.exists():
            logger.error(f"Anomaly scores file not found: {ANOMALY_SCORES_PATH}")
            return None
        
        # Load CSV into DataFrame
        df = pd.read_csv(ANOMALY_SCORES_PATH)
        
        # Verify required columns exist
        required_cols = ['file_name', 'split', 'anomaly_score', 'is_anomaly']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        logger.info(f"Loaded anomaly scores for {len(df)} samples")
        
        # Print basic statistics
        n_train = len(df[df['split'] == 'train'])
        n_test = len(df[df['split'] == 'test'])
        n_anomalies = df['is_anomaly'].sum()
        
        logger.info(f"Data distribution: {n_train} train, {n_test} test")
        logger.info(f"Anomalies: {n_anomalies} ({n_anomalies/len(df):.2%} of all samples)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading anomaly scores: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_anomaly_score_histogram(df):
    """
    Create histogram plots of anomaly scores for all data, train, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly scores
        
    Returns:
        str: Path to saved plot file or None if plotting fails
    """
    try:
        logger.info("Creating anomaly score histogram plots")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot 1: All data
        sns.histplot(df['anomaly_score'], kde=True, ax=axes[0], color='skyblue')
        axes[0].axvline(x=df['anomaly_score'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {df["anomaly_score"].mean():.4f}')
        axes[0].axvline(x=df[df['is_anomaly'] == 1]['anomaly_score'].min(), color='green', 
                        linestyle='-.', label='Anomaly Threshold')
        axes[0].set_title('Anomaly Score Distribution - All Data')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Plot 2: Train data
        train_df = df[df['split'] == 'train']
        sns.histplot(train_df['anomaly_score'], kde=True, ax=axes[1], color='lightgreen')
        axes[1].axvline(x=train_df['anomaly_score'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {train_df["anomaly_score"].mean():.4f}')
        axes[1].axvline(x=train_df[train_df['is_anomaly'] == 1]['anomaly_score'].min(), 
                        color='green', linestyle='-.', label='Anomaly Threshold')
        axes[1].set_title('Anomaly Score Distribution - Train Set')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # Plot 3: Test data
        test_df = df[df['split'] == 'test']
        if len(test_df) > 0:
            sns.histplot(test_df['anomaly_score'], kde=True, ax=axes[2], color='salmon')
            axes[2].axvline(x=test_df['anomaly_score'].mean(), color='red', linestyle='--', 
                            label=f'Mean: {test_df["anomaly_score"].mean():.4f}')
            if len(test_df[test_df['is_anomaly'] == 1]) > 0:
                axes[2].axvline(x=test_df[test_df['is_anomaly'] == 1]['anomaly_score'].min(), 
                                color='green', linestyle='-.', label='Anomaly Threshold')
            axes[2].set_title('Anomaly Score Distribution - Test Set')
            axes[2].set_xlabel('Anomaly Score')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
        else:
            axes[2].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = PLOTS_DIR / "hist_anomaly_scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved histogram plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting anomaly score histogram: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_anomaly_score_boxplot(df):
    """
    Create boxplot of anomaly scores grouped by split.
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly scores
        
    Returns:
        str: Path to saved plot file or None if plotting fails
    """
    try:
        logger.info("Creating anomaly score boxplot")
        
        plt.figure(figsize=(10, 6))
        
        # Create boxplot
        sns.boxplot(x='split', y='anomaly_score', hue='is_anomaly', 
                    data=df, palette={0: 'skyblue', 1: 'salmon'})
        
        # Add swarmplot for individual points
        sns.swarmplot(x='split', y='anomaly_score', hue='is_anomaly', 
                      data=df, dodge=True, alpha=0.5, size=4)
        
        # Add labels and title
        plt.title('Anomaly Score Distribution by Split', fontsize=14)
        plt.xlabel('Data Split', fontsize=12)
        plt.ylabel('Anomaly Score', fontsize=12)
        plt.legend(title='Is Anomaly', labels=['Normal', 'Anomaly'])
        
        # Add summary statistics as text
        stats_text = []
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            stats_text.append(f"{split} (n={len(split_df)}): "
                             f"mean={split_df['anomaly_score'].mean():.4f}, "
                             f"anomalies={split_df['is_anomaly'].sum()} "
                             f"({split_df['is_anomaly'].mean():.2%})")
        
        plt.figtext(0.5, 0.01, '\n'.join(stats_text), ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        output_path = PLOTS_DIR / "boxplot_anomaly_scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved boxplot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting anomaly score boxplot: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_reconstruction_vs_anomaly(df):
    """
    Create scatter plot of reconstruction error vs anomaly score.
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly scores and reconstruction errors
        
    Returns:
        str: Path to saved plot file or None if plotting fails
    """
    try:
        # Check if reconstruction_error column exists
        if 'reconstruction_error' not in df.columns:
            logger.warning("reconstruction_error column not found, skipping scatter plot")
            return None
        
        logger.info("Creating reconstruction error vs anomaly score scatter plot")
        
        plt.figure(figsize=(12, 8))
        
        # Calculate correlation for the title
        corr = df['reconstruction_error'].corr(df['anomaly_score'])
        
        # Create scatter plot
        scatter = plt.scatter(df['reconstruction_error'], df['anomaly_score'],
                            c=df['is_anomaly'], cmap='coolwarm', alpha=0.7, s=80)
        
        # Add regression line
        x = df['reconstruction_error']
        y = df['anomaly_score']
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, 'r--', linewidth=1)
        
        # Add labels and title
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Anomaly Score', fontsize=12)
        plt.title(f'Correlation between Reconstruction Error and Anomaly Score (r={corr:.4f})',
                fontsize=14)
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            loc="upper left", title="Is Anomaly")
        plt.gca().add_artist(legend1)
        
        # Add summary statistics as text
        stats_text = [
            f"Correlation: {corr:.4f}",
            f"Normal points: {len(df[df['is_anomaly'] == 0])}",
            f"Anomaly points: {len(df[df['is_anomaly'] == 1])}"
        ]
        
        plt.figtext(0.5, 0.01, '\n'.join(stats_text), ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        output_path = PLOTS_DIR / "scatter_recon_vs_anomaly.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved scatter plot to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error plotting reconstruction vs anomaly scatter: {e}")
        logger.error(traceback.format_exc())
        return None


def save_evaluation_summary(df, plot_paths):
    """
    Save a text summary of the evaluation results.
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly scores
        plot_paths (list): List of paths to generated plots
        
    Returns:
        str: Path to saved summary file
    """
    try:
        logger.info("Creating evaluation summary")
        
        # Create report content
        report_lines = [
            "======================================================",
            "        ISOLATION FOREST EVALUATION SUMMARY           ",
            "======================================================",
            ""
        ]
        
        # Add dataset information
        report_lines.extend([
            "1. DATASET INFORMATION",
            "---------------------",
            f"Total samples: {len(df)}",
            f"Train samples: {len(df[df['split'] == 'train'])}",
            f"Test samples: {len(df[df['split'] == 'test'])}",
            ""
        ])
        
        # Add anomaly detection results
        report_lines.extend([
            "2. ANOMALY DETECTION RESULTS",
            "----------------------------",
            f"Total anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.2%})",
            f"Train anomalies: {df[df['split'] == 'train']['is_anomaly'].sum()} ({df[df['split'] == 'train']['is_anomaly'].mean():.2%})",
            f"Test anomalies: {df[df['split'] == 'test']['is_anomaly'].sum()} ({df[df['split'] == 'test']['is_anomaly'].mean():.2%})",
            ""
        ])
        
        # Add anomaly score statistics
        report_lines.extend([
            "3. ANOMALY SCORE STATISTICS",
            "---------------------------",
            f"Overall mean: {df['anomaly_score'].mean():.6f}",
            f"Overall std: {df['anomaly_score'].std():.6f}",
            f"Min score: {df['anomaly_score'].min():.6f}",
            f"Max score: {df['anomaly_score'].max():.6f}",
            "",
            f"Normal samples mean: {df[df['is_anomaly'] == 0]['anomaly_score'].mean():.6f}",
            f"Anomaly samples mean: {df[df['is_anomaly'] == 1]['anomaly_score'].mean():.6f}",
            ""
        ])
        
        # Add reconstruction error correlation if available
        if 'reconstruction_error' in df.columns:
            corr = df['reconstruction_error'].corr(df['anomaly_score'])
            report_lines.extend([
                "4. CORRELATION WITH RECONSTRUCTION ERROR",
                "----------------------------------------",
                f"Pearson correlation: {corr:.6f}",
                ""
            ])
        
        # Add plots information
        if plot_paths:
            report_lines.extend([
                "5. GENERATED PLOTS",
                "-----------------"
            ])
            
            for path in plot_paths:
                if path:
                    report_lines.append(f"- {Path(path).name}")
            
            report_lines.append("")
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.extend([
            f"Report generated on: {timestamp}",
            "======================================================"
        ])
        
        # Write report to file
        output_path = REPORTS_DIR / "isolation_forest_evaluation.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved evaluation summary to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error saving evaluation summary: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main function to load data, create visualization plots, and save results.
    """
    try:
        logger.info("Starting Isolation Forest evaluation")
        
        # Set default style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Load anomaly scores
        df = load_anomaly_scores()
        if df is None:
            logger.error("Failed to load anomaly scores, aborting")
            return
        
        # Create visualization plots
        plot_paths = []
        
        # 1. Histogram of anomaly scores
        hist_path = plot_anomaly_score_histogram(df)
        if hist_path:
            plot_paths.append(hist_path)
        
        # 2. Boxplot of anomaly scores by split
        box_path = plot_anomaly_score_boxplot(df)
        if box_path:
            plot_paths.append(box_path)
        
        # 3. Scatter plot of reconstruction error vs anomaly score
        if 'reconstruction_error' in df.columns:
            scatter_path = plot_reconstruction_vs_anomaly(df)
            if scatter_path:
                plot_paths.append(scatter_path)
        
        # Save evaluation summary
        summary_path = save_evaluation_summary(df, plot_paths)
        
        logger.info("Isolation Forest evaluation completed")
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        
    except Exception as e:
        logger.error(f"Error in Isolation Forest evaluation: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()