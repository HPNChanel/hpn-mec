#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions for the HPN Medicare AI module.

This module provides shared functionality for model loading, directory management,
device setup, and data handling across the AI module scripts.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import traceback
import torch.nn as nn

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
from ai_module.models.autoencoder import Autoencoder

# Get logger for this module
logger = logging.getLogger(__name__)

def ensure_dir_exists(path):
    """
    Check if directory exists and create it if it doesn't.
    
    Args:
        path (str or Path): Path to check/create
        
    Returns:
        Path: Path object of the directory
    """
    path = Path(path) if not isinstance(path, Path) else path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"[INFO] Created missing directory at: {path.absolute()}")
    return path

def setup_device():
    """
    Set up and return the appropriate device (CUDA or CPU).
    
    Returns:
        torch.device: Device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Print additional GPU info for debugging
        logging.info(f"[INFO] CUDA version: {torch.version.cuda}")
        logging.info(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"[INFO] GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        logging.warning(f"[WARNING] CUDA not available. Using CPU instead. This may slow down processing significantly.")
    
    return device

def load_autoencoder_model(model_path, device):
    """
    Load the trained autoencoder model from its weights file.
    
    Args:
        model_path (Path): Path to the model weights file
        device (torch.device): Device to load the model on
    
    Returns:
        nn.Module: Loaded autoencoder model or None if loading fails
    """
    try:
        if not model_path.exists():
            logger.error(f"[ERROR] Model file not found: {model_path}")
            return None
        
        # Initialize model architecture
        logger.info(f"[INFO] Initializing autoencoder model")
        model = Autoencoder(input_dim=12000, latent_dim=64, use_sigmoid=True)
        
        # Load model weights - use map_location to ensure proper device mapping
        logger.info(f"[INFO] Loading model weights from {model_path} to {device}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set model to evaluation mode
        model.eval()
        
        # Move model to device
        model = model.to(device)
        logger.info(f"[INFO] Model moved to {device}")
        
        # Test with random data to verify model is working
        with torch.no_grad():
            test_input = torch.randn(1, 12000, device=device)  # Create tensor directly on target device
            test_output, _ = model(test_input)
            
            # Check if model is identity mapping (which would indicate issues)
            if torch.allclose(test_input, test_output, rtol=1e-3, atol=1e-3):
                logger.warning("[WARNING] Model behaves like identity function!")
                
            # Check for NaN or Inf in output
            if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                logger.warning("[WARNING] Model produces NaN or Inf values!")
        
        logger.info(f"[INFO] Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return None

def find_signal_files(primary_dir, fallback_dir=None):
    """
    Find signal files to process, first checking primary directory,
    then falling back to fallback directory if specified.
    
    Args:
        primary_dir (Path): Primary path to check for signal files
        fallback_dir (Path, optional): Fallback path to check if no files in primary
        
    Returns:
        tuple: (list of file paths, source directory) or (None, None) if no files found
    """
    # Ensure directories exist
    ensure_dir_exists(primary_dir)
    if fallback_dir:
        ensure_dir_exists(fallback_dir)
    
    # Check primary directory
    logging.info(f"[INFO] Looking for input signals in: {primary_dir.absolute()}")
    signal_files = list(primary_dir.glob("*.npy"))
    
    if signal_files:
        logging.info(f"[INFO] Found {len(signal_files)} files in {primary_dir}")
        return signal_files, primary_dir
    
    # If no files and fallback provided, check fallback
    if fallback_dir:
        logging.warning(f"[WARNING] Found 0 files. Trying fallback: {fallback_dir.absolute()}")
        signal_files = list(fallback_dir.glob("*.npy"))
        
        if signal_files:
            logging.info(f"[INFO] Found {len(signal_files)} files in {fallback_dir}")
            return signal_files, fallback_dir
    
    # No files found in any location
    fallback_msg = f" or {fallback_dir.absolute()}" if fallback_dir else ""
    logging.error(f"[ERROR] No files found in any location: {primary_dir.absolute()}{fallback_msg}")
    return None, None

def preprocess_signal(signal_path):
    """
    Load and preprocess a signal file for inference.
    
    Args:
        signal_path (Path): Path to the signal file
        
    Returns:
        tuple: (original_signal, normalized_tensor, file_stem) or (None, None, None) if preprocessing fails
    """
    try:
        # Load signal from .npy file
        signal = np.load(signal_path)
        
        # Get file stem (filename without extension)
        file_stem = signal_path.stem
        
        # Flatten signal if needed (from 2D to 1D)
        if len(signal.shape) > 1:
            flattened = signal.flatten()
        else:
            flattened = signal
        
        # Ensure the signal has the expected input dimension (12000)
        if len(flattened) != 12000:
            logging.warning(
                f"[WARNING] Signal dimension mismatch for {signal_path.name}: "
                f"got {len(flattened)}, expected 12000"
            )
            # For now, we'll skip signals with incorrect dimensions
            return None, None, None
        
        # Min-max normalization to [0, 1]
        min_val = flattened.min()
        max_val = flattened.max()
        
        # Handle constant signals to avoid division by zero
        if max_val > min_val:
            normalized = (flattened - min_val) / (max_val - min_val)
        else:
            logging.warning(f"[WARNING] Found constant signal in {signal_path.name}")
            normalized = np.zeros_like(flattened)
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(normalized, dtype=torch.float32)
        
        return signal, tensor, file_stem
        
    except Exception as e:
        logging.error(f"[ERROR] Error preprocessing signal {signal_path}: {e}")
        return None, None, None

def ensure_module_dir():
    """
    Find the ai_module directory from the current file location.
    
    Returns:
        Path: Path to the ai_module directory
    """
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if parent.name == "ai_module":
            return parent
    raise ImportError("Cannot find ai_module directory in path hierarchy")

def get_project_paths():
    """
    Get standard project paths relative to the ai_module directory.
    
    Returns:
        dict: Dictionary containing key project paths
    """
    module_dir = ensure_module_dir()
    
    paths = {
        "module_dir": module_dir,
        "models_dir": module_dir / "models",
        "data_dir": module_dir / "data",
        "processed_dir": module_dir / "data/processed",
        "signals_dir": module_dir / "data/processed/signals",
        "real_signals_dir": module_dir / "data/processed/signals/real",
        "fake_signals_dir": module_dir / "data/processed/signals/fake",
        "latents_dir": module_dir / "data/processed/latents",
        "reports_dir": module_dir / "reports",
        "plots_dir": module_dir / "reports/plots"
    }
    
    # Ensure all directories exist
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    
    return paths

# Create global paths dictionary for other modules to use
PROJECT_PATHS = get_project_paths()