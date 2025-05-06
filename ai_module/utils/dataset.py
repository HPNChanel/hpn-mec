#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Dataset Utility Module

This module contains the ECGDataset class for loading and preprocessing ECG signals.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ECGDataset(Dataset):
    """
    Dataset class for loading and preprocessing ECG signals for autoencoder training.
    """
    
    def __init__(self, signals_dir, min_files_warning=50):
        """
        Initialize the dataset.
        
        Args:
            signals_dir (Path): Directory containing .npy signal files
            min_files_warning (int): Minimum number of files below which to issue warning
        """
        self.signals_dir = signals_dir
        
        # Search recursively for .npy files
        all_npy_files = list(signals_dir.glob("**/*.npy"))
        
        # Initialize list for valid files
        self.file_list = []
        skipped_files = []
        
        # Validate each file
        for file_path in all_npy_files:
            try:
                # Try to load file to validate it
                signal = np.load(file_path)
                
                # Check if shape is valid (assuming we expect ECG signals)
                if len(signal.shape) < 1 or signal.size == 0:
                    logger.warning(f"Skipping file with invalid shape: {file_path.name}, shape: {signal.shape}")
                    skipped_files.append(file_path.name)
                    continue
                
                # If all checks pass, add to file list
                self.file_list.append(file_path)
                
            except Exception as e:
                logger.error(f"Error loading file {file_path.name}: {str(e)}")
                skipped_files.append(file_path.name)
        
        # Check if we have any valid files
        if not self.file_list:
            raise ValueError(f"No valid .npy files found in {signals_dir}")
        
        # Log number of files
        file_count = len(self.file_list)
        logger.info(f"Found {file_count} valid signal files")
        
        # Log skipped files
        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} invalid files")
        
        # Warning if fewer than min_files_warning
        if file_count < min_files_warning:
            file_list_str = ", ".join([f.name for f in self.file_list])
            logger.warning(f"Only {file_count} files found (fewer than {min_files_warning}). Files: {file_list_str}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Load, preprocess, and return a single ECG signal.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            torch.Tensor: Preprocessed ECG signal
        """
        # Load signal from .npy file
        file_path = self.file_list[idx]
        signal = np.load(file_path)
        
        # Flatten signal if needed
        flattened = signal.flatten()
        
        # Min-max normalization to [0, 1]
        min_val = flattened.min()
        max_val = flattened.max()
        
        # Handle constant signals to avoid division by zero
        if max_val > min_val:
            normalized = (flattened - min_val) / (max_val - min_val)
        else:
            logger.warning(f"Found constant signal in {file_path.name}")
            normalized = np.zeros_like(flattened)
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(normalized, dtype=torch.float32)
        
        return tensor
