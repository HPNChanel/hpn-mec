#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder Evaluation Script for ECG Signal Reconstruction

This script evaluates a trained autoencoder on ECG signal data, comparing
original signals with their reconstructions, computing error metrics, and
generating visualization plots.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import traceback

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

# Get the project root (parent of ai_module)
PROJECT_ROOT = module_dir.parent

# Add project root to path for reliable imports
sys.path.insert(0, str(PROJECT_ROOT))

# Now use absolute imports
from ai_module.utils.train_utils import (
    ensure_dir_exists,
    setup_device,
    load_autoencoder_model,
    find_signal_files,
    preprocess_signal
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = PROJECT_ROOT / "data/processed/signals"
REAL_SIGNALS_DIR = SIGNALS_DIR / "real"
FAKE_SIGNALS_DIR = SIGNALS_DIR / "fake"
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

# Ensure output directories exist
ensure_dir_exists(REAL_SIGNALS_DIR)
ensure_dir_exists(FAKE_SIGNALS_DIR)
ensure_dir_exists(PLOTS_DIR)

# Set device for evaluation
device = setup_device()


def load_signals(real_dir, fake_dir, raw_dir=None):
    """
    Load and pair original and reconstructed signals.
    
    Args:
        real_dir (Path): Directory containing original signals
        fake_dir (Path): Directory containing reconstructed signals
        raw_dir (Path, optional): Fallback directory for original signals
        
    Returns:
        list: List of signal pairs (original, reconstructed, filename)
    """
    try:
        # Check directories are different
        if real_dir.samefile(fake_dir):
            logger.error("[ERROR] Real and fake signal directories are identical!")
            return None
        
        # First try to find original signal files
        real_files, source_dir = find_signal_files(real_dir, raw_dir)
        
        if real_files is None or len(real_files) == 0:
            logger.error("[ERROR] No original signal files found, cannot proceed with evaluation")
            return None
        
        # Initialize list to store signal pairs
        signal_pairs = []
        
        # Process each original signal file
        for real_path in real_files:
            # Get corresponding reconstructed signal file
            fake_path = fake_dir / real_path.name
            
            # Check if reconstructed file exists
            if not fake_path.exists():
                logger.warning(f"[WARNING] No matching reconstructed signal for {real_path.name}")
                continue
            
            try:
                # Load signals
                real_signal = np.load(real_path)
                fake_signal = np.load(fake_path)
                
                # Add to list
                signal_pairs.append((real_signal, fake_signal, real_path.stem))
            except Exception as e:
                logger.warning(f"[WARNING] Error loading signals for {real_path.name}: {e}")
                continue
        
        if not signal_pairs:
            logger.error("[ERROR] No valid signal pairs found. Check that reconstructed signals exist.")
            return None
            
        logger.info(f"[INFO] Successfully paired {len(signal_pairs)} signal files")
        
        identical_count = sum(1 for real, fake, _ in signal_pairs 
                            if np.allclose(real, fake, rtol=1e-5, atol=1e-5))
        
        if identical_count == len(signal_pairs):
            logger.error("[ERROR] All signal pairs are identical. Critical error!")
        elif identical_count > 0:
            logger.warning(f"[WARNING] {identical_count}/{len(signal_pairs)} pairs identical")
            
        return signal_pairs
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading signals: {e}")
        logger.error(traceback.format_exc())
        return None


def compute_metrics(signal_pairs):
    """
    Compute reconstruction error metrics for each signal pair.
    
    Args:
        signal_pairs (list): List of signal pairs (original, reconstructed, filename)
        
    Returns:
        pd.DataFrame: DataFrame with reconstruction error metrics
    """
    try:
        if not signal_pairs:
            logger.error("[ERROR] No signal pairs provided for metric computation")
            return None
            
        # Initialize list to store metrics
        metrics = []
        
        # Process each signal pair
        for real_signal, fake_signal, filename in tqdm(signal_pairs, desc="Computing metrics"):
            # Ensure signals have same shape
            if real_signal.shape != fake_signal.shape:
                logger.warning(f"[WARNING] Shape mismatch for {filename}: {real_signal.shape} vs {fake_signal.shape}")
                continue
            
            # Compute metrics
            mse = np.mean((real_signal - fake_signal) ** 2)
            mae = np.mean(np.abs(real_signal - fake_signal))
            rmse = np.sqrt(mse)
            
            # Optional: compute normalized metrics
            if np.max(real_signal) > np.min(real_signal):
                norm_factor = np.max(real_signal) - np.min(real_signal)
                nrmse = rmse / norm_factor
            else:
                nrmse = np.nan
            
            # Compute correlation
            if np.std(real_signal) > 0 and np.std(fake_signal) > 0:
                correlation = np.corrcoef(real_signal.flatten(), fake_signal.flatten())[0, 1]
            else:
                correlation = np.nan
            
            # Add to metrics list
            metrics.append({
                'filename': filename,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'nrmse': nrmse,
                'correlation': correlation
            })
        
        if not metrics:
            logger.error("[ERROR] No valid metrics computed")
            return None
            
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Sort by MSE (ascending)
        metrics_df = metrics_df.sort_values(by='mse')
        
        logger.info(f"[INFO] Computed metrics for {len(metrics_df)} signal pairs")
        
        return metrics_df
        
    except Exception as e:
        logger.error(f"[ERROR] Error computing metrics: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_signal_comparison(signal_pairs, metrics_df, output_dir, n_samples=5):
    """
    Create and save plots comparing original and reconstructed signals.
    
    Args:
        signal_pairs (list): List of signal pairs (original, reconstructed, filename)
        metrics_df (pd.DataFrame): DataFrame with reconstruction error metrics
        output_dir (Path): Directory to save plots
        n_samples (int): Number of samples to plot (best and worst)
        
    Returns:
        list: Paths to saved plot files
    """
    try:
        if not signal_pairs or metrics_df is None or metrics_df.empty:
            logger.error("[ERROR] Cannot create plots without valid signal pairs and metrics")
            return []
            
        # Create dictionary for quick lookup
        signal_dict = {filename: (real, fake) for real, fake, filename in signal_pairs}
        
        # Get best and worst reconstructions based on MSE
        best_files = metrics_df.head(n_samples)['filename'].tolist()
        worst_files = metrics_df.tail(n_samples)['filename'].tolist()
        
        plot_paths = []
        
        # Plot best reconstructions
        for i, filename in enumerate(best_files):
            # Skip if file not in dictionary
            if filename not in signal_dict:
                logger.warning(f"[WARNING] Signal not found for {filename}, skipping plot")
                continue
                
            real_signal, fake_signal = signal_dict[filename]
            
            try:
                # Create figure
                plt.figure(figsize=(12, 6))
                
                # Reshape if needed (assuming 1D flattened signals)
                if len(real_signal.shape) == 1:
                    # Determine if we should reshape to 2D (e.g., time x channels)
                    # For now, keep as 1D for simplicity
                    pass
                
                # Plot signals
                plt.plot(real_signal.flatten(), label='Original', alpha=0.8)
                plt.plot(fake_signal.flatten(), label='Reconstructed', alpha=0.8)
                
                # Get metrics for title
                mse = metrics_df.loc[metrics_df['filename'] == filename, 'mse'].values[0]
                corr = metrics_df.loc[metrics_df['filename'] == filename, 'correlation'].values[0]
                
                # Add labels and title
                plt.xlabel('Sample Index', fontsize=12)
                plt.ylabel('Signal Amplitude', fontsize=12)
                plt.title(f'Best Reconstruction #{i+1}: {filename}\nMSE: {mse:.6f}, Correlation: {corr:.4f}', fontsize=14)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                output_path = output_dir / f"best_reconstruction_{i+1}_{filename}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(output_path))
            except Exception as e:
                logger.warning(f"[WARNING] Error creating plot for {filename}: {e}")
                continue
        
        # Plot worst reconstructions
        for i, filename in enumerate(worst_files):
            # Skip if file not in dictionary
            if filename not in signal_dict:
                logger.warning(f"[WARNING] Signal not found for {filename}, skipping plot")
                continue
                
            real_signal, fake_signal = signal_dict[filename]
            
            try:
                # Create figure
                plt.figure(figsize=(12, 6))
                
                # Plot signals
                plt.plot(real_signal.flatten(), label='Original', alpha=0.8)
                plt.plot(fake_signal.flatten(), label='Reconstructed', alpha=0.8)
                
                # Get metrics for title
                mse = metrics_df.loc[metrics_df['filename'] == filename, 'mse'].values[0]
                corr = metrics_df.loc[metrics_df['filename'] == filename, 'correlation'].values[0]
                
                # Add labels and title
                plt.xlabel('Sample Index', fontsize=12)
                plt.ylabel('Signal Amplitude', fontsize=12)
                plt.title(f'Worst Reconstruction #{i+1}: {filename}\nMSE: {mse:.6f}, Correlation: {corr:.4f}', fontsize=14)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                output_path = output_dir / f"worst_reconstruction_{i+1}_{filename}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(output_path))
            except Exception as e:
                logger.warning(f"[WARNING] Error creating plot for {filename}: {e}")
                continue
        
        # Plot overlay of best and worst reconstruction
        try:
            if best_files and worst_files and best_files[0] in signal_dict and worst_files[-1] in signal_dict:
                plt.figure(figsize=(15, 10))
                
                # Create 2x2 subplot grid
                plt.subplot(2, 2, 1)
                best_filename = best_files[0]
                best_real, best_fake = signal_dict[best_filename]
                plt.plot(best_real.flatten(), label='Original', alpha=0.8)
                plt.plot(best_fake.flatten(), label='Reconstructed', alpha=0.8)
                best_mse = metrics_df.loc[metrics_df['filename'] == best_filename, 'mse'].values[0]
                plt.title(f'Best Reconstruction: {best_filename}\nMSE: {best_mse:.6f}', fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend()
                
                plt.subplot(2, 2, 2)
                worst_filename = worst_files[-1]
                worst_real, worst_fake = signal_dict[worst_filename]
                plt.plot(worst_real.flatten(), label='Original', alpha=0.8)
                plt.plot(worst_fake.flatten(), label='Reconstructed', alpha=0.8)
                worst_mse = metrics_df.loc[metrics_df['filename'] == worst_filename, 'mse'].values[0]
                plt.title(f'Worst Reconstruction: {worst_filename}\nMSE: {worst_mse:.6f}', fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend()
                
                # Add a zoomed in portion of the best reconstruction
                plt.subplot(2, 2, 3)
                segment_length = min(500, len(best_real.flatten()))
                plt.plot(best_real.flatten()[:segment_length], label='Original', alpha=0.8)
                plt.plot(best_fake.flatten()[:segment_length], label='Reconstructed', alpha=0.8)
                plt.title(f'Best Reconstruction (Zoomed)', fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend()
                
                # Add a zoomed in portion of the worst reconstruction
                plt.subplot(2, 2, 4)
                plt.plot(worst_real.flatten()[:segment_length], label='Original', alpha=0.8)
                plt.plot(worst_fake.flatten()[:segment_length], label='Reconstructed', alpha=0.8)
                plt.title(f'Worst Reconstruction (Zoomed)', fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save overlay figure
                overlay_path = output_dir / f"reconstruction_comparison_overlay_{best_filename}_{worst_filename}.png"
                plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(overlay_path))
        except Exception as e:
            logger.warning(f"[WARNING] Error creating overlay plot: {e}")
        
        logger.info(f"[INFO] Created {len(plot_paths)} comparison plots")
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"[ERROR] Error plotting signal comparison: {e}")
        logger.error(traceback.format_exc())
        return []


def plot_metrics_distribution(metrics_df, output_dir):
    """
    Create and save plots of metrics distributions.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with reconstruction error metrics
        output_dir (Path): Directory to save plots
        
    Returns:
        list: Paths to saved plot files
    """
    try:
        if metrics_df is None or metrics_df.empty:
            logger.error("[ERROR] Cannot create distribution plots without valid metrics")
            return []
            
        plot_paths = []
        
        # Create histogram of MSE
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics_df['mse'], kde=True, bins=30, color='skyblue')
            plt.axvline(x=metrics_df['mse'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {metrics_df["mse"].mean():.6f}')
            plt.axvline(x=metrics_df['mse'].median(), color='green', linestyle='-.', 
                       label=f'Median: {metrics_df["mse"].median():.6f}')
            plt.xlabel('Mean Squared Error (MSE)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of Reconstruction MSE', fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            mse_path = output_dir / f"mse_distribution_{timestamp}.png"
            plt.savefig(mse_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(mse_path))
        except Exception as e:
            logger.warning(f"[WARNING] Error creating MSE distribution plot: {e}")
        
        # Create histogram of Correlation
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics_df['correlation'], kde=True, bins=30, color='lightgreen')
            plt.axvline(x=metrics_df['correlation'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {metrics_df["correlation"].mean():.4f}')
            plt.axvline(x=metrics_df['correlation'].median(), color='green', linestyle='-.', 
                       label=f'Median: {metrics_df["correlation"].median():.4f}')
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of Reconstruction Correlation', fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            corr_path = output_dir / f"correlation_distribution_{timestamp}.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(corr_path))
        except Exception as e:
            logger.warning(f"[WARNING] Error creating correlation distribution plot: {e}")
        
        # Create scatter plot of MSE vs Correlation
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='mse', y='correlation', data=metrics_df, alpha=0.7)
            plt.xlabel('Mean Squared Error (MSE)', fontsize=12)
            plt.ylabel('Correlation Coefficient', fontsize=12)
            plt.title('Relationship Between MSE and Correlation', fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            scatter_path = output_dir / f"mse_vs_correlation_{timestamp}.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(scatter_path))
        except Exception as e:
            logger.warning(f"[WARNING] Error creating scatter plot: {e}")
        
        # Create box plot of metrics
        try:
            plt.figure(figsize=(12, 8))
            metrics_melted = pd.melt(metrics_df, id_vars=['filename'], 
                                     value_vars=['mse', 'mae', 'rmse', 'nrmse'])
            sns.boxplot(x='variable', y='value', data=metrics_melted)
            plt.yscale('log')  # Use log scale for better visualization
            plt.xlabel('Metric', fontsize=12)
            plt.ylabel('Value (log scale)', fontsize=12)
            plt.title('Distribution of Reconstruction Error Metrics', fontsize=14)
            plt.xticks(rotation=0)
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save figure
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            box_path = output_dir / f"error_metrics_boxplot_{timestamp}.png"
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(box_path))
        except Exception as e:
            logger.warning(f"[WARNING] Error creating boxplot: {e}")
        
        logger.info(f"[INFO] Created {len(plot_paths)} distribution plots")
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"[ERROR] Error plotting metrics distribution: {e}")
        logger.error(traceback.format_exc())
        return []


def save_metrics_report(metrics_df, plot_paths, output_file):
    """
    Save metrics report to text file.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with reconstruction error metrics
        plot_paths (list): List of paths to saved plot files
        output_file (Path): Path to output report file
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if metrics_df is None or metrics_df.empty:
            logger.error("[ERROR] Cannot create report without valid metrics")
            return False
            
        # Create report content
        report_lines = [
            "======================================================",
            "            AUTOENCODER EVALUATION REPORT             ",
            "======================================================",
            ""
        ]
        
        # Add summary statistics
        report_lines.extend([
            "1. SUMMARY STATISTICS",
            "---------------------"
        ])
        
        # Add statistics for each metric
        metrics_to_report = ['mse', 'mae', 'rmse', 'nrmse', 'correlation']
        for metric in metrics_to_report:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                
                report_lines.extend([
                    f"\n{metric.upper()} Statistics:",
                    f"  Mean: {values.mean():.8f}",
                    f"  Median: {values.median():.8f}",
                    f"  Std Dev: {values.std():.8f}",
                    f"  Min: {values.min():.8f}",
                    f"  Max: {values.max():.8f}"
                ])
        
        report_lines.append("")
        
        # Add best and worst reconstructions
        report_lines.extend([
            "2. BEST RECONSTRUCTIONS (Lowest MSE)",
            "-----------------------------------"
        ])
        
        for i, row in metrics_df.head(5).iterrows():
            report_lines.append(
                f"  {row['filename']}: MSE={row['mse']:.8f}, Correlation={row['correlation']:.4f}"
            )
        
        report_lines.extend([
            "",
            "3. WORST RECONSTRUCTIONS (Highest MSE)",
            "-------------------------------------"
        ])
        
        for i, row in metrics_df.tail(5).iterrows():
            report_lines.append(
                f"  {row['filename']}: MSE={row['mse']:.8f}, Correlation={row['correlation']:.4f}"
            )
        
        report_lines.append("")
        
        # Add generated plots
        if plot_paths:
            report_lines.extend([
                "4. GENERATED PLOTS",
                "-----------------"
            ])
            
            for path in plot_paths:
                if path:
                    report_lines.append(f"  - {path}")
            
            report_lines.append("")
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.extend([
            f"Report generated on: {timestamp}",
            "======================================================"
        ])
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[INFO] Saved evaluation report to {output_file}")
        
        # Also save metrics DataFrame to CSV
        csv_path = output_file.parent / "reconstruction_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"[INFO] Saved metrics to {csv_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving metrics report: {e}")
        logger.error(traceback.format_exc())
        return False


def generate_predictions_with_model(model, signal_pairs, output_dir):
    """
    Generate reconstructions using the model for verification purposes.
    
    Args:
        model (nn.Module): Loaded autoencoder model
        signal_pairs (list): List of signal pairs (original, reconstructed, filename)
        output_dir (Path): Directory to save plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Select a sample for visualization
        if not signal_pairs:
            logger.warning("[WARNING] No signal pairs available for model prediction")
            return False
        
        # Choose the middle sample for demonstration
        sample_idx = len(signal_pairs) // 2
        real_signal, saved_recon, filename = signal_pairs[sample_idx]
        
        # Prepare input tensor
        input_tensor = torch.tensor(real_signal.flatten(), dtype=torch.float32).to(device)
        
        # Ensure input has batch dimension
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Generate reconstruction with the model
        with torch.no_grad():
            model_recon, _ = model(input_tensor)
            model_recon_np = model_recon.cpu().numpy().flatten()
        
        # Plot original and both reconstructions
        plt.figure(figsize=(12, 6))
        plt.plot(real_signal.flatten(), label='Original', alpha=0.8)
        plt.plot(saved_recon.flatten(), label='Pre-saved Reconstruction', alpha=0.7)
        plt.plot(model_recon_np, label='Model-generated Reconstruction', alpha=0.7, linestyle='--')
        
        # Add labels and title
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Signal Amplitude', fontsize=12)
        plt.title(f'Model Verification: Original vs Reconstructions for {filename}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"model_verification_{filename}_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[INFO] Created model verification plot: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error in model prediction: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """
    Main function to evaluate the autoencoder on signal reconstructions.
    """
    try:
        logger.info("[INFO] Starting autoencoder evaluation")
        
        # Load model
        model_path = MODELS_DIR / "autoencoder.pt"
        model = load_autoencoder_model(model_path, device)
        if model is None:
            logger.error("[ERROR] Failed to load model, aborting")
            return
        
        # Load signal pairs (original and reconstructed)
        signal_pairs = load_signals(REAL_SIGNALS_DIR, FAKE_SIGNALS_DIR, RAW_DATA_DIR)
        if signal_pairs is None or len(signal_pairs) == 0:
            logger.error("[ERROR] No signal pairs found, aborting")
            return
        
        # Compute metrics
        metrics_df = compute_metrics(signal_pairs)
        if metrics_df is None:
            logger.error("[ERROR] Failed to compute metrics, aborting")
            return
        
        # Plot signal comparisons
        comparison_paths = plot_signal_comparison(signal_pairs, metrics_df, PLOTS_DIR)
        
        # Plot metrics distributions
        distribution_paths = plot_metrics_distribution(metrics_df, PLOTS_DIR)
        
        # Optional: Generate reconstructions with the model for verification
        generate_predictions_with_model(model, signal_pairs, PLOTS_DIR)
        
        # Combine all plot paths
        all_plot_paths = []
        if comparison_paths:
            all_plot_paths.extend(comparison_paths)
        if distribution_paths:
            all_plot_paths.extend(distribution_paths)
        
        # Save metrics report
        report_path = REPORTS_DIR / "eval_autoencoder.txt"
        save_metrics_report(metrics_df, all_plot_paths, report_path)
        
        logger.info("[INFO] Autoencoder evaluation completed")
        
    except Exception as e:
        logger.error(f"[ERROR] Error in autoencoder evaluation: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()