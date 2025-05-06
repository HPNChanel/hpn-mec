# TODO: Implement this module

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB-XL ECG Data Preprocessing Module

This module processes ECG data from the PTB-XL dataset, extracting metadata from .hea files,
reading signal data from .dat files, normalizing the signals, and saving processed
data for AI model training.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import json
import logging
import numpy as np
import wfdb
import datetime
from tqdm import tqdm
from pathlib import Path

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
RAW_DATA_DIR = module_dir / "data/raw"
PROCESSED_DATA_DIR = module_dir / "data/processed"
PROCESSED_SIGNALS_DIR = PROCESSED_DATA_DIR / "signals"
PROCESSED_METADATA_DIR = PROCESSED_DATA_DIR / "metadata"

# Ensure output directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_SIGNALS_DIR, PROCESSED_METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_header(file_path):
    """
    Load and parse header information from .hea files.
    
    Args:
        file_path (str or Path): Path to the .hea file
        
    Returns:
        dict: Dictionary containing metadata extracted from the header
              or None if an error occurs
    """
    try:
        # Convert to Path object if it's a string
        file_path = Path(file_path).resolve()
        
        # Remove extension (if any) and convert to POSIX format for WFDB
        record_path = file_path.with_suffix('').as_posix()
            
        # Read header information
        record_info = wfdb.rdheader(record_path)
        
        # Extract relevant metadata
        metadata = {
            'record_name': record_info.record_name,
            'n_sig': record_info.n_sig,  # Number of signals (channels)
            'fs': record_info.fs,  # Sample frequency (Hz)
            'sig_len': record_info.sig_len,  # Signal length (number of samples)
            'units': [unit for unit in record_info.units] if hasattr(record_info, 'units') else None,
            'adc_gain': [gain for gain in record_info.adc_gain] if hasattr(record_info, 'adc_gain') else None,
            'baseline': [baseline for baseline in record_info.baseline] if hasattr(record_info, 'baseline') else None,
            'lead_names': [name for name in record_info.sig_name] if hasattr(record_info, 'sig_name') else None,
        }
        
        # Add patient ID from filename if available
        if '_' in record_info.record_name:
            metadata['patient_id'] = record_info.record_name.split('_')[0]
        
        # Try to extract comments if available
        if hasattr(record_info, 'comments') and record_info.comments:
            metadata['comments'] = record_info.comments
            
            # Parse comments for demographic info if present
            for comment in record_info.comments:
                if 'Age:' in comment:
                    metadata['age'] = comment.split('Age:')[1].strip()
                if 'Sex:' in comment:
                    metadata['sex'] = comment.split('Sex:')[1].strip()
                if 'Dx:' in comment:
                    metadata['diagnosis'] = comment.split('Dx:')[1].strip()
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading header from {file_path}: {e}")
        return None


def load_signal(file_path):
    """
    Load ECG signal data from .dat files.
    
    Args:
        file_path (str or Path): Path to the record (with or without extension)
        
    Returns:
        numpy.ndarray: Multi-channel ECG signal data with shape (n_samples, n_channels)
                      or None if an error occurs
    """
    try:
        # Convert to Path object if it's a string and resolve to absolute path
        file_path = Path(file_path).resolve()
        
        # Remove extension (if any) and convert to POSIX format for WFDB
        record_path = file_path.with_suffix('').as_posix()
            
        # Read signal
        record = wfdb.rdrecord(record_path)
        
        # p_signal contains the physical signal (with units)
        signals = record.p_signal
        
        return signals
    
    except Exception as e:
        logger.error(f"Error loading signal from {file_path}: {e}")
        return None


def normalize_signal(signal, method='z-score'):
    """
    Normalize ECG signal data to standardize ranges for AI model training.
    
    Args:
        signal (numpy.ndarray): Multi-channel ECG signal data with shape (n_samples, n_channels)
        method (str): Normalization method, either 'z-score' or 'min-max'
        
    Returns:
        numpy.ndarray: Normalized signal data with the same shape as input
                      or None if an error occurs
    
    Notes:
        - z-score normalization: (x - mean) / std
          * Centers data around 0 with std=1
          * Better for algorithms sensitive to feature scales (e.g., neural networks)
        
        - min-max normalization: (x - min) / (max - min)
          * Scales data to [0, 1] range
          * Preserves relationships between original values
    """
    if signal is None:
        return None
    
    try:
        # Create a copy to avoid modifying the original
        normalized = np.copy(signal)
        
        # Apply normalization for each channel separately
        for i in range(signal.shape[1]):
            channel = signal[:, i]
            
            if method == 'z-score':
                # Z-score normalization: (x - mean) / std
                mean = np.mean(channel)
                std = np.std(channel)
                # Check for constant signals (std = 0)
                if std != 0:
                    normalized[:, i] = (channel - mean) / std
                else:
                    # If channel is constant, set to zeros
                    logger.warning(f"Channel {i} has zero standard deviation. Setting to zeros.")
                    normalized[:, i] = np.zeros_like(channel)
            
            elif method == 'min-max':
                # Min-max normalization: (x - min) / (max - min)
                min_val = np.min(channel)
                max_val = np.max(channel)
                # Check for constant signals (max = min)
                if max_val > min_val:
                    normalized[:, i] = (channel - min_val) / (max_val - min_val)
                else:
                    # If channel is constant, set to zeros
                    logger.warning(f"Channel {i} has constant value. Setting to zeros.")
                    normalized[:, i] = np.zeros_like(channel)
            
            else:
                logger.warning(f"Unknown normalization method '{method}'. Using original signal.")
                return signal
        
        return normalized
    
    except Exception as e:
        logger.error(f"Error normalizing signal: {e}")
        return signal  # Return original signal if normalization fails


def save_output(signal, metadata, base_filename, signal_dir=None, metadata_dir=None):
    """
    Save processed signal and metadata to output directories.
    
    Args:
        signal (numpy.ndarray): Normalized multi-channel ECG signal
        metadata (dict): Extracted metadata
        base_filename (str): Base filename (without extension)
        signal_dir (Path, optional): Directory to save signal data, defaults to PROCESSED_SIGNALS_DIR
        metadata_dir (Path, optional): Directory to save metadata, defaults to PROCESSED_METADATA_DIR
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    if signal is None or metadata is None:
        return False
    
    # Use default directories if not specified
    signal_dir = signal_dir or PROCESSED_SIGNALS_DIR
    metadata_dir = metadata_dir or PROCESSED_METADATA_DIR
    
    try:
        # Save normalized signal as numpy array
        signal_output_path = signal_dir / f"{base_filename}.npy"
        np.save(signal_output_path, signal)
        
        # Save metadata as JSON
        metadata_output_path = metadata_dir / f"{base_filename}.json"
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        return True
    
    except Exception as e:
        logger.error(f"Error saving output for {base_filename}: {e}")
        return False


def process_ecg_file(file_path, signal_dir=None, metadata_dir=None):
    """
    Process a single ECG file (both header and signal).
    
    Args:
        file_path (str or Path): Path to the .hea file
        signal_dir (Path, optional): Directory to save signal data, defaults to PROCESSED_SIGNALS_DIR
        metadata_dir (Path, optional): Directory to save metadata, defaults to PROCESSED_METADATA_DIR
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Convert to Path object if it's a string
        file_path = Path(file_path)
        base_filename = file_path.stem
        
        # Load header and extract metadata
        metadata = load_header(file_path)
        if metadata is None:
            logger.error(f"Failed to load header for {file_path}")
            return False
        
        # Load signal data
        signal = load_signal(file_path)
        if signal is None:
            logger.error(f"Failed to load signal for {file_path}")
            return False
        
        # Normalize signal
        normalized_signal = normalize_signal(signal, method='z-score')
        if normalized_signal is None:
            logger.error(f"Failed to normalize signal for {file_path}")
            return False
        
        # Save processed data
        success = save_output(normalized_signal, metadata, base_filename, 
                              signal_dir=signal_dir, metadata_dir=metadata_dir)
        if not success:
            logger.error(f"Failed to save output for {file_path}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False


def main():
    """
    Main function to process all ECG files in the raw data directory.
    Scans for all .hea and .dat file pairs and converts valid recordings to .npy format.
    """
    # Find all .hea files recursively under the raw data directory
    hea_files = list(RAW_DATA_DIR.glob('**/*.hea'))
    
    total_files = len(hea_files)
    logger.info(f"Found {total_files} header files to process")
    
    if total_files == 0:
        logger.warning(f"No .hea files found in {RAW_DATA_DIR}")
        return
    
    # Track processing statistics
    processed_count = 0
    skipped_count = 0
    error_count = 0
    skipped_files = []
    
    # Process each file with progress bar
    for file_path in tqdm(hea_files, desc="Processing ECG recordings"):
        # Check if corresponding .dat file exists
        dat_path = file_path.with_suffix('.dat')
        
        if not dat_path.exists():
            logger.warning(f"Missing .dat file for {file_path.name} - skipping")
            skipped_count += 1
            skipped_files.append(file_path.name)
            continue
        
        # Process the file pair
        try:
            if process_ecg_file(file_path):
                processed_count += 1
            else:
                logger.warning(f"Failed to process {file_path.name} - skipping")
                skipped_count += 1
                skipped_files.append(file_path.name)
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")
            error_count += 1
            skipped_files.append(file_path.name)
    
    # Print processing summary
    logger.info(f"\n----- Processing Summary -----")
    logger.info(f"Total .hea files found: {total_files}")
    logger.info(f"Successfully processed: {processed_count} pairs into .npy")
    logger.info(f"Skipped files: {skipped_count}")
    logger.info(f"Error encountered: {error_count}")
    
    # Log skipped files if any
    if skipped_files:
        if len(skipped_files) <= 10:
            logger.info(f"Skipped files: {', '.join(skipped_files)}")
        else:
            logger.info(f"Skipped {len(skipped_files)} files (first 10): {', '.join(skipped_files[:10])}")
    
    # Save a detailed summary
    summary = {
        'total_files_found': total_files,
        'successfully_processed': processed_count,
        'skipped_files': skipped_count,
        'error_count': error_count,
        'skipped_file_list': skipped_files,
        'normalization_method': 'z-score',
        'processing_date': str(datetime.datetime.now())
    }
    
    summary_path = PROCESSED_DATA_DIR / "preprocessing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed processing summary saved to {summary_path}")
    
    # Final message
    if processed_count > 0:
        logger.info(f"Successfully processed {processed_count} ECG recording pairs into .npy format")
    else:
        logger.error("No ECG recordings were successfully processed")


if __name__ == "__main__":
    logger.info("Starting ECG data preprocessing")
    main()
    logger.info("ECG data preprocessing completed")