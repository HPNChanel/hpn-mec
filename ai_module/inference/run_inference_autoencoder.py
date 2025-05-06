#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder Inference Script for ECG Signals

This script loads a trained autoencoder model and runs inference on ECG signals,
generating reconstructions, latent representations, and computing reconstruction errors.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import traceback
from datetime import datetime
from ai_module.utils.train_utils import ensure_dir_exists, setup_device

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

# Define base directories relative to module_dir
PROCESSED_DIR = module_dir / "data" / "processed"
SIGNALS_DIR = PROCESSED_DIR / "signals"
SIGNALS_FAKE_DIR = PROCESSED_DIR / "signals" / "fake"
SIGNALS_REAL_DIR = PROCESSED_DIR / "signals" / "real"
LATENTS_DIR = PROCESSED_DIR / "latents"
MODELS_DIR = module_dir / "models"
REPORTS_DIR = module_dir / "reports"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "reconstruction_scores.csv"
# Ensure directories exist
for directory in [SIGNALS_DIR, SIGNALS_FAKE_DIR, SIGNALS_REAL_DIR, LATENTS_DIR, REPORTS_DIR]:
    ensure_dir_exists(directory)
    ensure_dir_exists(directory)

# Set device for inference
device = setup_device()

# Now use absolute imports
from ai_module.utils.train_utils import (
    ensure_dir_exists, 
    setup_device, 
    load_autoencoder_model, 
    preprocess_signal
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def batch_signals(signal_files, batch_size=32):
    """
    Group signal files into batches for more efficient processing.
    
    Args:
        signal_files (list): List of paths to signal files
        batch_size (int): Size of each batch
        
    Returns:
        list: List of batches, each containing signal paths
    """
    batches = []
    for i in range(0, len(signal_files), batch_size):
        batches.append(signal_files[i:i + batch_size])
    return batches


def process_batch(model, signal_batch):
    """
    Process a batch of signals with the autoencoder model.
    
    Args:
        model: Trained autoencoder model
        signal_batch (list): List of signal paths to process
        
    Returns:
        list: List of result dictionaries
    """
    batch_results = []
    valid_tensors = []
    file_stems = []
    original_signals = []
    
    # Preprocess all signals in the batch
    for signal_path in signal_batch:
        original_signal, signal_tensor, file_stem = preprocess_signal(signal_path)
        
        if signal_tensor is None:
            logger.warning(f"[WARNING] Skipping {signal_path.name} due to preprocessing error")
            continue
        
        valid_tensors.append(signal_tensor)
        file_stems.append(file_stem)
        original_signals.append(original_signal)
    
    if not valid_tensors:
        return []
    
    # Stack tensors into a batch
    batch_tensor = torch.stack(valid_tensors).to(device)
    
    # Run inference
    with torch.no_grad():
        reconstructed, latent = model(batch_tensor)
        
        # Calculate reconstruction errors
        mse = torch.mean((reconstructed - batch_tensor) ** 2, dim=1).cpu().numpy()
        mae = torch.mean(torch.abs(reconstructed - batch_tensor), dim=1).cpu().numpy()
    
    # Process results for each sample in the batch
    for i in range(len(file_stems)):
        # Get results for this sample
        reconstructed_np = reconstructed[i].cpu().numpy()
        latent_np = latent[i].cpu().numpy()
        
        # Save reconstructed signal
        reconstructed_path = SIGNALS_FAKE_DIR / f"{file_stems[i]}.npy"
        np.save(reconstructed_path, reconstructed_np)
        
        # Save latent vector
        latent_path = LATENTS_DIR / f"{file_stems[i]}.npy"
        np.save(latent_path, latent_np)
        
        # Add result to list
        batch_results.append({
            'file_name': f"{file_stems[i]}.npy",
            'reconstruction_error': mse[i],
            'mae': mae[i],
            'latent_dim': latent_np.shape[0],
            'latent_vector_path': str(latent_path),
            'original_shape': str(original_signals[i].shape) if original_signals[i] is not None else None
        })
    
    return batch_results


def run_inference(model, signal_files, batch_size=32):
    """
    Run inference using the autoencoder model on signal files.
    
    Args:
        model: Trained autoencoder model
        signal_files (list): List of paths to signal files
        batch_size (int): Size of each batch for processing
        
    Returns:
        pd.DataFrame: DataFrame with inference results or None if inference fails
    """
    try:
        if model is None or not signal_files:
            logger.error("[ERROR] Cannot run inference with None inputs")
            return None
        
        logger.info("[INFO] Running inference with autoencoder model")
        
        # Group signals into batches
        batches = batch_signals(signal_files, batch_size)
        logger.info(f"[INFO] Processing {len(signal_files)} signals in {len(batches)} batches")
        
        # Initialize list to store all results
        all_results = []
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            results = process_batch(model, batch)
            all_results.extend(results)
            
            # Log progress periodically
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                logger.info(f"[INFO] Processed {batch_idx + 1}/{len(batches)} batches ({len(all_results)}/{len(signal_files)} signals)")
        
        if not all_results:
            logger.error("[ERROR] No valid results generated")
            return None
        
        # Create DataFrame with results
        results_df = pd.DataFrame(all_results)
        
        # Calculate additional statistics
        reconstruction_errors = results_df['reconstruction_error']
        percentile_75 = np.percentile(reconstruction_errors, 75)
        percentile_90 = np.percentile(reconstruction_errors, 90)
        results_df['is_anomaly'] = (reconstruction_errors > percentile_90).astype(int)
        results_df['anomaly_score'] = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min())
        
        # Sort by reconstruction error (descending)
        results_df = results_df.sort_values(by='reconstruction_error', ascending=False)
        
        logger.info(f"[INFO] Inference completed for {len(results_df)} samples")
        logger.info(f"[INFO] 75th percentile threshold: {percentile_75:.6f}")
        logger.info(f"[INFO] 90th percentile threshold (anomaly): {percentile_90:.6f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"[ERROR] Error running inference: {e}")
        logger.error(traceback.format_exc())
        return None


def save_results(results_df, output_path):
    """
    Save inference results to CSV file.
    
    Args:
        results_df (pd.DataFrame): DataFrame with inference results
        output_path (Path): Path to save results
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        if results_df is None or results_df.empty:
            logger.error("[ERROR] Cannot save empty or None results")
            return False
        
        # Ensure output directory exists
        ensure_dir_exists(output_path.parent)
        
        # Add timestamp to column
        results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        # Also save a timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = output_path.parent / f"reconstruction_scores_{timestamp}.csv"
        results_df.to_csv(timestamped_path, index=False)
        
        logger.info(f"[INFO] Inference results saved to {output_path}")
        logger.info(f"[INFO] Timestamped copy saved to {timestamped_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving results: {e}")
        logger.error(traceback.format_exc())
        return False


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained autoencoder model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "autoencoder.pt"),
        help="Path to trained autoencoder model (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save inference results CSV (default: %(default)s)"
    )
    
    # Define RAW_DATA_DIR at the global level
    RAW_DATA_DIR = module_dir / "data" / "raw"
    ensure_dir_exists(RAW_DATA_DIR)
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to raw data directory (default: %(default)s)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(SIGNALS_REAL_DIR),
        help="Path to processed real signals directory (default: %(default)s)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: %(default)s)"
    )
    
    return parser.parse_args()


def find_signal_files(signals_dir):
    """
    Find all .npy signal files in the specified directory.
    
    Args:
        signals_dir (Path): Directory to search for signal files
        
    Returns:
        list: List of paths to signal files or None if no files found
    """
    try:
        logger.info(f"[INFO] Looking for signal files in {signals_dir}")
        
        # Check if directory exists
        if not signals_dir.exists():
            logger.error(f"[ERROR] Directory does not exist: {signals_dir}")
            return None
            
        # Find all .npy files
        signal_files = list(signals_dir.glob("*.npy"))
        
        if not signal_files:
            logger.error(f"[ERROR] No .npy signal files found in {signals_dir}")
            return None
            
        logger.info(f"[INFO] Found {len(signal_files)} signal files")
        return signal_files
        
    except Exception as e:
        logger.error(f"[ERROR] Error finding signal files: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main function to run inference with autoencoder model.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Convert paths
        model_path = Path(args.model)
        output_path = Path(args.output)
        batch_size = args.batch_size
        
        # Record start time
        start_time = datetime.now()
        logger.info(f"[INFO] Starting autoencoder inference at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load model
        model = load_autoencoder_model(model_path, device)
        if model is None:
            logger.error("[ERROR] Failed to load model, aborting")
            return
        
        # Find signal files
        signal_files = find_signal_files(SIGNALS_DIR)
        if signal_files is None:
            logger.error("[ERROR] No signal files found for inference, aborting")
            return
        
        # Run inference
        results_df = run_inference(model, signal_files, batch_size)
        if results_df is None:
            logger.error("[ERROR] Failed to run inference, aborting")
            return
        
        # Save results
        save_results(results_df, output_path)
        
        # Print summary statistics
        logger.info("[INFO] Inference summary statistics:")
        logger.info(f"[INFO]   Total samples processed: {len(results_df)}")
        logger.info(f"[INFO]   Mean reconstruction error (MSE): {results_df['reconstruction_error'].mean():.6f}")
        logger.info(f"[INFO]   Mean absolute error (MAE): {results_df['mae'].mean():.6f}")
        logger.info(f"[INFO]   Min reconstruction error: {results_df['reconstruction_error'].min():.6f}")
        logger.info(f"[INFO]   Max reconstruction error: {results_df['reconstruction_error'].max():.6f}")
        logger.info(f"[INFO]   Potential anomalies (90th percentile): {results_df['is_anomaly'].sum()}")
        
        # Identify potential anomalies (highest reconstruction errors)
        top_anomalies = results_df.head(5)  # Already sorted by error (descending)
        logger.info("[INFO] Top 5 potential anomalies (highest reconstruction errors):")
        for idx, row in top_anomalies.iterrows():
            logger.info(f"[INFO]   {row['file_name']}: {row['reconstruction_error']:.6f}")
        
        # Record end time and calculate duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"[INFO] Autoencoder inference completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"[INFO] Total execution time: {duration}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error in autoencoder inference: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()