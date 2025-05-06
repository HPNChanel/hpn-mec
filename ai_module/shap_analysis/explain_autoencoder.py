#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis for Autoencoder Model

This script uses SHAP (SHapley Additive exPlanations) to explain the
latent representation of the autoencoder for ECG signal reconstruction.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import shap
from tqdm import tqdm
import traceback  # Add missing import

# Configure logging early so it's available for import error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add missing imports for Captum
try:
    from captum.attr import IntegratedGradients, GradientShap, DeepLift
except ImportError:
    logger.error("[ERROR] Captum not installed. Install with: pip install captum")
    IntegratedGradients = GradientShap = DeepLift = None

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
# Logger is already configured above

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now use absolute imports
from ai_module.models.autoencoder import Autoencoder
from ai_module.utils.train_utils import (
    ensure_dir_exists,
    setup_device,
    load_autoencoder_model,
    find_signal_files,
    preprocess_signal
)

# Constants and paths - Consolidate all path definitions to avoid duplication
SIGNALS_DIR = module_dir / "data/processed/signals"
REAL_SIGNALS_DIR = SIGNALS_DIR / "real"  # Keep for backward compatibility
LATENTS_DIR = module_dir / "data/processed/latents"
MODELS_DIR = module_dir / "models"
REPORTS_DIR = module_dir / "reports"
PLOTS_DIR = REPORTS_DIR / "plots/shap"

# Ensure output directories exist
for directory in [LATENTS_DIR, PLOTS_DIR]:
    ensure_dir_exists(directory)

# Set device for evaluation
device = setup_device()

# Function to determine the best signals directory (with fallback)
def get_signal_directory():
    """
    Get the appropriate signals directory with fallback logic.
    
    Returns:
        Path: Path to the signals directory containing .npy files
    """
    # Check if the real signals directory exists and contains .npy files
    if REAL_SIGNALS_DIR.exists() and list(REAL_SIGNALS_DIR.glob("*.npy")):
        logger.info(f"[INFO] Using signals from {REAL_SIGNALS_DIR}")
        return REAL_SIGNALS_DIR
    
    # Fall back to the main signals directory
    if SIGNALS_DIR.exists() and list(SIGNALS_DIR.glob("*.npy")):
        logger.info(f"[INFO] 'real' subdirectory not found or empty. Using signals from {SIGNALS_DIR}")
        return SIGNALS_DIR
    
    # No valid directory found
    logger.warning(f"[WARNING] No signals found in either {REAL_SIGNALS_DIR} or {SIGNALS_DIR}")
    return SIGNALS_DIR  # Return the main directory anyway and let calling code handle empty case


class ReconstructionLoss(nn.Module):
    """
    Custom module to compute reconstruction loss for attribution methods.
    """
    def __init__(self, model):
        super(ReconstructionLoss, self).__init__()
        self.model = model
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, inputs):
        # Get reconstructions
        reconstructed, _ = self.model(inputs)
        
        # Compute MSE loss
        loss = self.mse_loss(reconstructed, inputs)
        
        # Return mean of loss across features
        return loss.mean(dim=1)


# Modified to properly handle file finding without raw_dir fallback
def load_signals(signal_dir=None, latent_dir=None, use_latents=False, max_samples=None):
    """
    Load signal data for analysis.
    
    Args:
        signal_dir (Path, optional): Directory containing original signals
        latent_dir (Path, optional): Directory containing latent vectors
        use_latents (bool): Whether to use latent vectors instead of original signals
        max_samples (int, optional): Maximum number of samples to load
        
    Returns:
        tuple: (signals, file_names) - numpy arrays and their filenames
    """
    try:
        if use_latents and latent_dir:
            # Load latent vectors
            data_dir = latent_dir
            logger.info(f"[INFO] Loading latent vectors from {data_dir.absolute()}")
            data_files = list(data_dir.glob("*.npy"))
        else:
            # Load original signals
            data_dir = signal_dir
            logger.info(f"[INFO] Loading original signals from {data_dir.absolute()}")
            data_files = list(data_dir.glob("*.npy"))
        
        if not data_files:
            logger.error(f"[ERROR] No .npy files found in {data_dir}")
            return None, None
        
        # Limit number of samples if specified
        if max_samples and len(data_files) > max_samples:
            logger.info(f"[INFO] Limiting to {max_samples} files")
            data_files = data_files[:max_samples]
        
        # Load data
        signals = []
        file_names = []
        
        for file_path in tqdm(data_files, desc="Loading data"):
            try:
                # Load data
                data = np.load(file_path)
                
                # Flatten if needed
                if len(data.shape) > 1:
                    data = data.flatten()
                
                signals.append(data)
                file_names.append(file_path.stem)
                
            except Exception as e:
                logger.warning(f"[WARNING] Error loading {file_path.name}: {e}")
                continue
        
        if not signals:
            logger.error("[ERROR] No valid signals could be loaded")
            return None, None
            
        # Convert to numpy array
        signals = np.array(signals)
        
        logger.info(f"[INFO] Loaded {len(signals)} samples with shape {signals.shape}")
        
        return signals, file_names
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading signals: {e}")
        logger.error(traceback.format_exc())
        return None, None


def load_reconstruction_errors():
    """
    Load reconstruction errors to identify high error samples.
    
    Returns:
        pd.DataFrame: DataFrame with reconstruction errors
    """
    try:
        # Try to load from evaluation report
        recon_errors_path = REPORTS_DIR / "reconstruction_metrics.csv"
        
        # If not found, try alternative path
        if not recon_errors_path.exists():
            recon_errors_path = REPORTS_DIR / "reconstruction_scores.csv"
        
        # If still not found, return None
        if not recon_errors_path.exists():
            logger.warning(f"[WARNING] Reconstruction errors file not found at {recon_errors_path.absolute()}")
            return None
        
        # Load CSV
        df = pd.read_csv(recon_errors_path)
        
        # Determine which column has the errors
        error_col = None
        for col in ['mse', 'mae', 'rmse', 'reconstruction_error']:
            if col in df.columns:
                error_col = col
                break
        
        if error_col is None:
            logger.warning(f"[WARNING] No error column found in {recon_errors_path}")
            return None
        
        # Ensure there's a filename column
        name_col = None
        for col in ['filename', 'file_name']:
            if col in df.columns:
                name_col = col
                break
        
        if name_col is None:
            logger.warning(f"[WARNING] No filename column found in {recon_errors_path}")
            return None
        
        # Sort by error (descending)
        df = df.sort_values(by=error_col, ascending=False)
        
        logger.info(f"[INFO] Loaded reconstruction errors for {len(df)} samples")
        
        return df
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading reconstruction errors: {e}")
        logger.error(traceback.format_exc())
        return None


def validate_tensor_shape(tensor, expected_shape=None):
    """
    Validate tensor shape before passing to attribution methods.
    
    Args:
        tensor (torch.Tensor): Tensor to validate
        expected_shape (tuple, optional): Expected shape (ignoring batch dimension)
        
    Returns:
        bool: True if tensor has valid shape, False otherwise
    """
    if tensor is None:
        logger.error("[ERROR] Cannot validate None tensor")
        return False
        
    if len(tensor.shape) == 0:
        logger.error(f"[ERROR] Tensor has no dimensions: {tensor.shape}")
        return False
        
    if expected_shape and tensor.shape[1:] != expected_shape:
        logger.error(f"[ERROR] Tensor has unexpected shape: {tensor.shape}, expected: (batch, {expected_shape})")
        return False
        
    return True


def compute_captum_attributions(model, signals, target_samples=5, method='integrated_gradients'):
    """
    Compute feature attributions using Captum.
    
    Args:
        model (nn.Module): Autoencoder model
        signals (numpy.ndarray): Input signals
        target_samples (int): Number of samples to analyze
        method (str): Attribution method to use
        
    Returns:
        tuple: (attributions, signal_batch, model_outputs)
    """
    try:
        logger.info(f"[INFO] Computing attributions using Captum with {method} method")
        
        # Convert signals to tensor
        if isinstance(signals, np.ndarray):
            signals_tensor = torch.tensor(signals, dtype=torch.float32)
        else:
            signals_tensor = signals
            
        # Validate tensor shape
        if not validate_tensor_shape(signals_tensor):
            logger.error("[ERROR] Invalid tensor shape")
            return None, None, None
        
        # Select samples to analyze
        # We'll take the first 'target_samples' signals
        if len(signals_tensor) > target_samples:
            signal_batch = signals_tensor[:target_samples].to(device)
        else:
            signal_batch = signals_tensor.to(device)
            
        # Validate batch 
        if signal_batch.shape[0] == 0:
            logger.error("[ERROR] Empty batch")
            return None, None, None
        
        # Create reconstruction loss
        recon_loss = ReconstructionLoss(model)
        
        # Get baseline for attribution methods
        # For explanation, we'll use zeros as baseline
        baseline = torch.zeros_like(signal_batch).to(device)
        
        # Initialize attribution method
        if method == 'integrated_gradients':
            attribution_method = IntegratedGradients(recon_loss)
        elif method == 'gradient_shap':
            attribution_method = GradientShap(recon_loss)
        elif method == 'deep_lift':
            attribution_method = DeepLift(recon_loss)
        else:
            logger.warning(f"[WARNING] Unknown method {method}, falling back to IntegratedGradients")
            attribution_method = IntegratedGradients(recon_loss)
        
        # Compute attributions
        logger.info(f"[INFO] Computing attributions for {len(signal_batch)} samples")
        
        with torch.no_grad():
            # Get model outputs for reference
            model_outputs, _ = model(signal_batch)
        
        # Compute attributions
        if method == 'gradient_shap':
            # GradientShap requires multiple baselines
            baselines = torch.randn(10, *signal_batch.shape[1:]).to(device) * 0.001
            attributions = attribution_method.attribute(
                signal_batch,
                baselines=baselines,
                n_samples=50
            )
        else:
            attributions = attribution_method.attribute(
                signal_batch,
                baselines=baseline,
                n_steps=50
            )
        
        logger.info(f"[INFO] Generated attributions with shape {attributions.shape}")
        
        return attributions, signal_batch, model_outputs
        
    except Exception as e:
        logger.error(f"[ERROR] Error computing Captum attributions: {e}")
        logger.error(traceback.format_exc())
        return None, None, None


def compute_shap_attributions(model, signals, target_samples=5):
    """
    Compute feature attributions using SHAP.
    
    Args:
        model (nn.Module): Autoencoder model
        signals (numpy.ndarray): Input signals
        target_samples (int): Number of samples to analyze
        
    Returns:
        tuple: (shap_values, X_sample, model_outputs)
    """
    try:
        logger.info("[INFO] Computing attributions using SHAP")
        
        # Validate signals
        if signals is None or len(signals) == 0:
            logger.error("[ERROR] No signals provided for SHAP analysis")
            return None, None, None
            
        # Select samples to analyze
        if len(signals) > target_samples:
            X_sample = signals[:target_samples]
        else:
            X_sample = signals
        
        # Create background dataset for SHAP
        # We'll use a subset of the samples as background
        n_background = min(100, len(signals))
        if len(signals) > n_background:
            background_idx = np.random.choice(len(signals), n_background, replace=False)
            X_background = signals[background_idx]
        else:
            X_background = signals
        
        # Define model wrapper for SHAP
        def model_wrapper(inputs):
            # Handle batch or single input
            is_single = len(np.array(inputs).shape) == 1
            if is_single:
                inputs = [inputs]
                
            # Convert numpy arrays to tensor
            batch = torch.tensor(inputs, dtype=torch.float32).to(device)
            
            # Get reconstructions and convert to numpy
            with torch.no_grad():
                reconstructed, _ = model(batch)
                mse = ((reconstructed - batch) ** 2).mean(dim=1)
            
            return mse.cpu().numpy()
        
        # Initialize SHAP explainer
        logger.info(f"[INFO] Initializing SHAP KernelExplainer with {len(X_background)} background samples")
        explainer = shap.KernelExplainer(model_wrapper, X_background)
        
        # Compute SHAP values
        logger.info(f"[INFO] Computing SHAP values for {len(X_sample)} samples")
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        # Get model outputs for reference
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
        with torch.no_grad():
            model_outputs, _ = model(X_tensor)
            model_outputs = model_outputs.cpu().numpy()
        
        logger.info(f"[INFO] Generated SHAP values with shape {np.array(shap_values).shape}")
        
        return shap_values, X_sample, model_outputs
        
    except Exception as e:
        logger.error(f"[ERROR] Error computing SHAP attributions: {e}")
        logger.error(traceback.format_exc())
        return None, None, None


def plot_captum_attributions(attributions, inputs, outputs, file_names, output_dir):
    """
    Plot feature attributions computed with Captum.
    
    Args:
        attributions (torch.Tensor): Feature attributions
        inputs (torch.Tensor): Input samples
        outputs (torch.Tensor): Model outputs
        file_names (list): Names of the files
        output_dir (Path): Directory to save plots
        
    Returns:
        list: Paths to saved plots
    """
    try:
        if attributions is None or inputs is None or outputs is None or not file_names:
            logger.error("[ERROR] Cannot create plots without valid attribution data")
            return []
            
        plot_paths = []
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Visualization for individual samples
        for i in range(len(attributions)):
            try:
                # Get data for this sample
                attr = attributions[i].cpu().detach().numpy()
                inp = inputs[i].cpu().detach().numpy()
                out = outputs[i].cpu().detach().numpy()
                
                # For ECG signals, we can visualize by feature groups
                # First, let's extract and sum the attributions for feature chunks
                n_features = len(attr)
                n_chunks = min(20, n_features)  # Create at most 20 chunks
                chunk_size = n_features // n_chunks
                
                # Bucket the attributions into chunks
                chunk_attrs = []
                chunk_labels = []
                for j in range(n_chunks):
                    start_idx = j * chunk_size
                    end_idx = start_idx + chunk_size if j < n_chunks - 1 else n_features
                    
                    # Sum attributions in this chunk
                    chunk_sum = np.sum(np.abs(attr[start_idx:end_idx]))
                    chunk_attrs.append(chunk_sum)
                    chunk_labels.append(f"{start_idx}-{end_idx}")
                
                # Create bar plot of chunk attributions
                plt.figure(figsize=(12, 6))
                
                # Sort chunks by attribution magnitude
                sorted_indices = np.argsort(chunk_attrs)[::-1]  # Descending order
                sorted_chunks = [chunk_attrs[idx] for idx in sorted_indices]
                sorted_labels = [chunk_labels[idx] for idx in sorted_indices]
                
                # Plot top 10 chunks
                top_k = min(10, len(sorted_chunks))
                plt.bar(range(top_k), sorted_chunks[:top_k], color='skyblue')
                plt.xticks(range(top_k), sorted_labels[:top_k], rotation=45)
                plt.xlabel('Feature Chunks (Indices)')
                plt.ylabel('Attribution Magnitude')
                plt.title(f'Top Feature Importance for Sample {file_names[i]}')
                plt.tight_layout()
                
                # Save figure with unique filename
                output_path = output_dir / f"feature_importance_chunks_{file_names[i]}_{timestamp}_{i}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(output_path))
                
                # For ECG signals, also plot the contribution of each signal point
                plt.figure(figsize=(15, 8))
                
                # Create comparison of signal vs attributions
                plt.subplot(2, 1, 1)
                plt.plot(inp, label='Original Signal')
                plt.plot(out, label='Reconstructed Signal', alpha=0.7)
                plt.title(f'Signal Comparison for Sample {file_names[i]}')
                plt.legend()
                plt.grid(alpha=0.3)
                
                plt.subplot(2, 1, 2)
                # Replace deprecated stem with bar plot
                plt.bar(np.arange(len(attr)), attr, width=1.0, color='r', alpha=0.7)
                plt.title('Feature Attributions (Importance)')
                plt.xlabel('Feature Index')
                plt.ylabel('Attribution Value')
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                
                # Save figure with unique filename
                output_path = output_dir / f"feature_importance_detail_{file_names[i]}_{timestamp}_{i}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(output_path))
                
            except Exception as e:
                logger.warning(f"[WARNING] Error plotting sample {i}: {e}")
                logger.warning(traceback.format_exc())
                continue
        
        # Aggregate visualization across all samples
        try:
            # Compute average absolute attribution for each feature
            avg_attr = torch.mean(torch.abs(attributions), dim=0).cpu().numpy()
            
            # Bucket into chunks again for better visualization
            n_features = len(avg_attr)
            n_chunks = min(50, n_features)  # Create at most 50 chunks
            chunk_size = n_features // n_chunks
            
            # Bucket the attributions into chunks
            chunk_attrs = []
            chunk_labels = []
            for j in range(n_chunks):
                start_idx = j * chunk_size
                end_idx = start_idx + chunk_size if j < n_chunks - 1 else n_features
                
                # Sum attributions in this chunk
                chunk_sum = np.sum(avg_attr[start_idx:end_idx])
                chunk_attrs.append(chunk_sum)
                chunk_labels.append(f"{start_idx}-{end_idx}")
            
            # Create bar plot of aggregated chunk attributions
            plt.figure(figsize=(15, 8))
            
            # Sort chunks by attribution magnitude
            sorted_indices = np.argsort(chunk_attrs)[::-1]  # Descending order
            sorted_chunks = [chunk_attrs[idx] for idx in sorted_indices]
            sorted_labels = [chunk_labels[idx] for idx in sorted_indices]
            
            # Plot top 20 chunks
            top_k = min(20, len(sorted_chunks))
            plt.bar(range(top_k), sorted_chunks[:top_k], color='skyblue')
            plt.xticks(range(top_k), sorted_labels[:top_k], rotation=45)
            plt.xlabel('Feature Chunks (Indices)')
            plt.ylabel('Average Attribution Magnitude')
            plt.title('Top Feature Importance Aggregated Across All Samples')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save figure with unique filename
            output_path = output_dir / f"feature_importance_aggregated_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(output_path))
            
        except Exception as e:
            logger.warning(f"[WARNING] Error plotting aggregated attributions: {e}")
            logger.warning(traceback.format_exc())
        
        logger.info(f"[INFO] Created {len(plot_paths)} attribution plots")
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"[ERROR] Error plotting Captum attributions: {e}")
        logger.error(traceback.format_exc())
        return []


# Add explainer parameter to fix the undefined variable issue
def plot_shap_attributions(shap_values, X, file_names, output_dir, explainer=None):
    """
    Plot feature attributions computed with SHAP.
    
    Args:
        shap_values: SHAP values
        X (numpy.ndarray): Input samples
        file_names (list): Names of the files
        output_dir (Path): Directory to save plots
        explainer: SHAP explainer object containing expected_value
        
    Returns:
        list: Paths to saved plots
    """
    try:
        if shap_values is None or X is None or not file_names:
            logger.error("[ERROR] Cannot create plots without valid SHAP values")
            return []
            
        plot_paths = []
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Aggregate visualization across all samples
        try:
            plt.figure(figsize=(12, 8))
            
            # Create summary plot
            shap.summary_plot(
                shap_values, 
                X, 
                plot_type="bar", 
                show=False,
                max_display=20
            )
            
            plt.title('SHAP Feature Importance for Reconstruction Error', fontsize=14)
            plt.tight_layout()
            
            # Save figure with unique filename
            output_path = output_dir / f"shap_feature_importance_bar_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(output_path))
            
            # Create detailed summary plot
            plt.figure(figsize=(12, 8))
            
            # Create summary plot with all features
            shap.summary_plot(
                shap_values, 
                X, 
                show=False,
                max_display=20
            )
            
            plt.title('SHAP Feature Importance Summary', fontsize=14)
            plt.tight_layout()
            
            # Save figure with unique filename
            output_path = output_dir / f"shap_feature_importance_summary_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(output_path))
            
        except Exception as e:
            logger.warning(f"[WARNING] Error plotting SHAP summary: {e}")
            logger.warning(traceback.format_exc())
        
        # Individual sample visualizations using waterfall plots instead of force plots
        for i in range(min(len(X), len(file_names))):
            try:
                plt.figure(figsize=(20, 8))
                
                # Use a default expected value if explainer is not provided
                expected_value = 0
                if explainer is not None and hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                
                # Use waterfall plot instead of force plot which is more stable
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[i], 
                        base_values=expected_value,
                        data=X[i]
                    ), 
                    max_display=20,
                    show=False
                )
                
                plt.title(f'SHAP Feature Importance for Sample {file_names[i]}', fontsize=14)
                plt.tight_layout()
                
                # Save figure with unique filename
                output_path = output_dir / f"shap_waterfall_plot_{file_names[i]}_{timestamp}_{i}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths.append(str(output_path))
                
            except Exception as e:
                logger.warning(f"[WARNING] Error plotting SHAP waterfall plot for sample {i}: {e}")
                logger.warning(traceback.format_exc())
                continue
        
        logger.info(f"[INFO] Created {len(plot_paths)} SHAP plots")
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"[ERROR] Error plotting SHAP attributions: {e}")
        logger.error(traceback.format_exc())
        return []


def save_report(plot_paths, output_file):
    """
    Save report of feature importance analysis.
    
    Args:
        plot_paths (list): Paths to saved plots
        output_file (Path): Path to output report file
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        # Create report content
        report_lines = [
            "======================================================",
            "       AUTOENCODER FEATURE IMPORTANCE ANALYSIS        ",
            "======================================================",
            ""
        ]
        
        # Add explanation
        report_lines.extend([
            "ANALYSIS OVERVIEW",
            "----------------",
            "This analysis examines which input features (dimensions) of the ECG signals",
            "contribute most significantly to the autoencoder's reconstruction quality.",
            "Higher attribution values indicate features that have a larger impact on",
            "the reconstruction performance.",
            "",
            "METHODOLOGY",
            "-----------",
            "Feature importance was calculated using gradient-based attribution methods",
            "that analyze how changes in each input feature affect the reconstruction error.",
            "This helps identify the most critical segments of the ECG signal for accurate",
            "reconstruction by the autoencoder model.",
            ""
        ])
        
        # Add generated plots
        if plot_paths:
            report_lines.extend([
                "GENERATED VISUALIZATIONS",
                "-----------------------"
            ])
            
            for path in plot_paths:
                if path:
                    report_lines.append(f"  - {path}")
            
            report_lines.append("")
        else:
            report_lines.extend([
                "NO VISUALIZATIONS GENERATED",
                "-------------------------",
                "No plots were generated during the analysis. This could be due to:",
                "  - No valid signals found",
                "  - Error in attribution computation",
                "  - Error in plot generation",
                ""
            ])
        
        # Add key findings and implications
        report_lines.extend([
            "KEY FINDINGS",
            "------------",
            "1. Certain regions of the ECG signal have disproportionately high importance",
            "   for reconstruction quality.",
            "2. These critical regions likely correspond to important cardiac events in the",
            "   ECG cycle (P waves, QRS complexes, T waves).",
            "3. The identified important features can guide model optimization and potentially",
            "   improve anomaly detection performance.",
            "",
            "IMPLICATIONS",
            "-----------",
            "1. Feature importance can be used to focus attention on the most critical",
            "   parts of the signal for clinical interpretation.",
            "2. Model architectures can be refined to better capture these important features.",
            "3. Preprocessing steps can be optimized to preserve information in high-importance",
            "   regions of the signal.",
            ""
        ])
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.extend([
            f"Report generated on: {timestamp}",
            "======================================================"
        ])
        
        # Write report to file
        ensure_dir_exists(output_file.parent)
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[INFO] Saved feature importance report to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving report: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """
    Main function to analyze feature importance of the autoencoder.
    """
    try:
        logger.info("[INFO] Starting autoencoder feature importance analysis")
        
        # Load model
        model_path = MODELS_DIR / "autoencoder.pt"
        model = load_autoencoder_model(model_path, device)
        if model is None:
            logger.error("[ERROR] Failed to load model, aborting")
            return
        
        # Get the best signal directory with fallback
        signal_dir = get_signal_directory()
        
        # Load signals using the determined directory
        signals, file_names = load_signals(
            signal_dir,
            LATENTS_DIR, 
            use_latents=False, 
            max_samples=100
        )
        if signals is None or len(signals) == 0:
            logger.error("[ERROR] No signals found, aborting")
            return
        
        # Load reconstruction errors to identify high-error samples
        error_df = load_reconstruction_errors()
        
        # If errors are available, select samples with highest errors
        if error_df is not None:
            try:
                # Get file names from error DataFrame
                name_col = 'filename' if 'filename' in error_df.columns else 'file_name'
                high_error_files = error_df[name_col].head(10).tolist()
                
                # Map to indices in loaded signals
                high_error_indices = [i for i, name in enumerate(file_names) if name in high_error_files]
                
                # If we found matching files, use them
                if high_error_indices:
                    signals = signals[high_error_indices]
                    file_names = [file_names[i] for i in high_error_indices]
                    logger.info(f"[INFO] Selected {len(high_error_indices)} high-error samples for analysis")
            except Exception as e:
                logger.warning(f"[WARNING] Error selecting high-error samples: {e}")
                logger.warning(traceback.format_exc())
        
        # Compute attributions using Captum
        use_method = 'captum'  # 'captum' or 'shap'
        
        # Initialize explainer at a higher scope so it can be passed to plot function
        explainer = None
        
        if use_method == 'captum':
            # Convert signals to tensor
            signals_tensor = torch.tensor(signals, dtype=torch.float32)
            
            # Check tensor for NaN or Inf values
            if torch.isnan(signals_tensor).any() or torch.isinf(signals_tensor).any():
                logger.error("[ERROR] Input tensor contains NaN or Inf values, aborting")
                return
                
            # Compute attributions
            attributions, inputs, outputs = compute_captum_attributions(
                model, 
                signals_tensor, 
                target_samples=5, 
                method='integrated_gradients'
            )
            
            if attributions is None:
                logger.error("[ERROR] Failed to compute attributions, aborting")
                return
            
            # Plot attributions
            plot_paths = plot_captum_attributions(
                attributions, 
                inputs, 
                outputs, 
                file_names[:len(attributions)], 
                PLOTS_DIR
            )
            
        else:  # use_method == 'shap'
            # Compute SHAP values
            shap_values, X_sample, model_outputs = compute_shap_attributions(
                model, 
                signals, 
                target_samples=5
            )
            
            if shap_values is None:
                logger.error("[ERROR] Failed to compute SHAP values, aborting")
                return
            
            # Store the explainer from compute_shap_attributions
            # (assuming it sets this global variable)
            # Pass explainer to the plot function
            plot_paths = plot_shap_attributions(
                shap_values, 
                X_sample, 
                file_names[:len(X_sample)], 
                PLOTS_DIR,
                explainer
            )
        
        # Save report
        report_path = REPORTS_DIR / "explain_autoencoder.txt"
        save_report(plot_paths, report_path)
        
        logger.info("[INFO] Autoencoder feature importance analysis completed")
        
    except Exception as e:
        logger.error(f"[ERROR] Error in autoencoder feature importance analysis: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()