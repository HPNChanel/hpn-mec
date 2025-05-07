#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Vector Loader Utility Module

This module provides functions for loading latent vectors from .npy files,
creating train/test splits, and merging with reconstruction error data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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

# Define paths relative to the ai_module directory
LATENTS_DIR = module_dir / "data/processed/latents"
REPORTS_DIR = module_dir / "reports"


def load_latents(test_size=0.3, random_state=42, include_recon_errors=True):
    """
    Load all latent vectors from .npy files, create train/test split,
    and optionally merge with reconstruction errors.
    
    Args:
        test_size (float): Proportion of data to use for test set (default: 0.3)
        random_state (int): Random seed for reproducible splitting (default: 42)
        include_recon_errors (bool): Whether to include reconstruction errors (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with file_name, split, reconstruction_error (if available),
                      and latent vector features
    """
    try:
        logger.info(f"Loading latent vectors from {LATENTS_DIR}")
        
        # Check if directory exists
        if not LATENTS_DIR.exists():
            logger.error(f"Latent vectors directory not found: {LATENTS_DIR}")
            return None
        
        # Find all .npy files
        latent_files = list(LATENTS_DIR.glob("*.npy"))
        
        if not latent_files:
            logger.error(f"No .npy files found in {LATENTS_DIR}")
            return None
        
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
        
        if not features:
            logger.error("No valid latent vectors could be loaded")
            return None
        
        # Convert list of features to 2D array
        X = np.vstack(features)
        
        logger.info(f"Loaded feature matrix with shape: {X.shape}")
        
        # Create train/test split indices
        indices = np.arange(len(file_names))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # Create split labels
        split_labels = np.array(['train'] * len(file_names), dtype=object)
        split_labels[test_indices] = 'test'
        
        logger.info(f"Created split: {len(train_indices)} train, {len(test_indices)} test")
        
        # Create base DataFrame
        df = pd.DataFrame({
            'file_name': file_names,
            'split': split_labels
        })
        
        # Try to load reconstruction errors if requested
        if include_recon_errors:
            recon_errors = load_reconstruction_errors()
            if recon_errors is not None:
                # Merge with reconstruction errors
                df = merge_with_reconstruction_errors(df, recon_errors)
                logger.info(f"Merged with reconstruction errors")
        
        # Add latent features as columns
        for i in range(X.shape[1]):
            df[f'feature_{i+1}'] = X[:, i]
        
        # Store the feature matrix as a property of the DataFrame
        df.attrs['X'] = X
        
        logger.info(f"Final DataFrame has shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading latent vectors: {e}")
        logger.error(traceback.format_exc())
        return None


def load_reconstruction_errors():
    """
    Load reconstruction errors from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with reconstruction errors or None if loading fails
    """
    try:
        # Try multiple possible filenames for reconstruction errors
        possible_paths = [
            REPORTS_DIR / "reconstruction_scores.csv",
            REPORTS_DIR / "reconstruction_metrics.csv",
            REPORTS_DIR / "autoencoder_results.csv"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found reconstruction errors at: {path}")
                df = pd.read_csv(path)
                return df
        
        logger.warning("No reconstruction error files found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading reconstruction errors: {e}")
        logger.error(traceback.format_exc())
        return None


def merge_with_reconstruction_errors(latents_df, recon_errors_df):
    """
    Merge latent vectors DataFrame with reconstruction errors.
    
    Args:
        latents_df (pd.DataFrame): DataFrame with latent vectors
        recon_errors_df (pd.DataFrame): DataFrame with reconstruction errors
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        if latents_df is None or recon_errors_df is None:
            logger.error("Cannot merge with None DataFrames")
            return latents_df
        
        # Determine which column has the file names in reconstruction errors DataFrame
        file_col = None
        for col_name in ['file_name', 'filename', 'name', 'id']:
            if col_name in recon_errors_df.columns:
                file_col = col_name
                break
        
        if file_col is None:
            logger.error("Could not find file name column in reconstruction errors")
            return latents_df
        
        # Determine which column has the reconstruction errors
        error_col = None
        for col_name in ['reconstruction_error', 'mse', 'error', 'loss']:
            if col_name in recon_errors_df.columns:
                error_col = col_name
                break
        
        if error_col is None:
            logger.error("Could not find reconstruction error column")
            return latents_df
        
        # Extract file stems from recon_errors_df if needed
        if '.' in recon_errors_df[file_col].iloc[0]:
            recon_errors_df['file_name'] = recon_errors_df[file_col].apply(
                lambda x: Path(x).stem
            )
        else:
            recon_errors_df['file_name'] = recon_errors_df[file_col]
        
        # Merge DataFrames
        merged_df = pd.merge(
            latents_df,
            recon_errors_df[['file_name', error_col]],
            on='file_name',
            how='left'
        )
        
        # Rename the error column to a standardized name if it's not already
        if error_col != 'reconstruction_error':
            merged_df.rename(columns={error_col: 'reconstruction_error'}, inplace=True)
        
        logger.info(f"Merged DataFrame has {len(merged_df)} rows")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging with reconstruction errors: {e}")
        logger.error(traceback.format_exc())
        return latents_df


def main():
    """
    Main function for testing the module.
    """
    try:
        logger.info("Testing latent vector loading")
        
        # Load latent vectors
        df = load_latents()
        
        if df is not None:
            logger.info(f"Successfully loaded {len(df)} latent vectors")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Feature matrix shape: {df.attrs['X'].shape}")
            
            # Print sample
            logger.info("\nSample data:")
            logger.info(df.head().to_string())
            
            # Print train/test distribution
            train_count = (df['split'] == 'train').sum()
            test_count = (df['split'] == 'test').sum()
            logger.info(f"\nSplit distribution: {train_count} train, {test_count} test")
            
            # Check if reconstruction errors are available
            if 'reconstruction_error' in df.columns:
                logger.info("\nReconstruction error stats:")
                logger.info(f"Mean: {df['reconstruction_error'].mean():.6f}")
                logger.info(f"Min: {df['reconstruction_error'].min():.6f}")
                logger.info(f"Max: {df['reconstruction_error'].max():.6f}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()