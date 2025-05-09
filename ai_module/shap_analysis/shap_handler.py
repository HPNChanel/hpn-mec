#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis Handler for HPN Medicare AI System

This module provides utilities for generating and interpreting SHAP values
for both AutoEncoder and Isolation Forest models.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import numpy as np
import pandas as pd
import torch
import json
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import project modules
from ai_module.models import AutoEncoder, AUTOENCODER_PATH, ISOLATION_FOREST_PATH, AUTOENCODER_META_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShapHandler:
    """
    Handler for generating and analyzing SHAP values for AI models.
    
    This class supports both AutoEncoder and Isolation Forest models and
    provides methods to generate, visualize, and interpret SHAP values.
    """

    def __init__(
        self, 
        model_type: str,
        model_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the SHAP handler.
        
        Args:
            model_type: Type of model ('autoencoder' or 'isolation_forest')
            model_path: Path to the model file (optional, will use default if None)
            feature_names: List of feature names (optional)
            background_data: Background data for SHAP explainer (optional)
            device: Device to run computation on ('cpu' or 'cuda')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.explainer = None
        self.feature_names = feature_names
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        
        if self.model_type not in ['autoencoder', 'isolation_forest']:
            raise ValueError(f"Unsupported model type: {model_type}. Must be 'autoencoder' or 'isolation_forest'")
        
        # Set default model path if none provided
        if model_path is None:
            if self.model_type == 'autoencoder':
                model_path = AUTOENCODER_PATH
            else:  # isolation_forest
                model_path = ISOLATION_FOREST_PATH
        
        self.model_path = Path(model_path)
        
        # Load model
        self._load_model()
        
        # Load model metadata for autoencoder
        if self.model_type == 'autoencoder':
            self._load_model_metadata()
        
        # Set up SHAP explainer
        if background_data is not None:
            self._setup_explainer(background_data)
            
        logger.info(f"Initialized SHAP Handler for {self.model_type}")
    
    def _load_model(self):
        """Load the model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            if self.model_type == 'autoencoder':
                # Load autoencoder model using PyTorch
                self.model = torch.load(self.model_path, map_location=self.device)
                # Set to evaluation mode
                self.model.eval()
                logger.info(f"Loaded AutoEncoder model from {self.model_path}")
            elif self.model_type == 'isolation_forest':
                # Load isolation forest model using pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded Isolation Forest model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
    
    def _load_model_metadata(self):
        """Load model metadata for autoencoder."""
        meta_path = Path(str(self.model_path).replace('.pt', '_meta.json'))
        if not meta_path.exists():
            meta_path = AUTOENCODER_META_PATH
            
        if not meta_path.exists():
            logger.warning(f"Model metadata file not found: {meta_path}")
            self.input_dim = None
            self.latent_dim = None
            return
            
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            self.input_dim = metadata.get('input_dim')
            self.latent_dim = metadata.get('latent_dim')
            logger.info(f"Loaded model metadata: input_dim={self.input_dim}, latent_dim={self.latent_dim}")
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            self.input_dim = None
            self.latent_dim = None
    
    def _setup_explainer(self, background_data: np.ndarray):
        """Set up the SHAP explainer based on model type."""
        try:
            if self.model_type == 'autoencoder':
                # For autoencoder, we use DeepExplainer or GradientExplainer
                background_tensor = torch.FloatTensor(background_data).to(self.device)
                
                # Define a function to compute reconstruction error
                def reconstruction_error(x):
                    tensor_x = torch.FloatTensor(x).to(self.device)
                    reconstructed, _ = self.model(tensor_x)
                    errors = torch.mean((reconstructed - tensor_x) ** 2, dim=1)
                    return errors.detach().cpu().numpy()
                
                # Use KernelExplainer since we're explaining a custom function
                self.explainer = shap.KernelExplainer(reconstruction_error, background_data)
                
            elif self.model_type == 'isolation_forest':
                # For isolation forest, we can use TreeExplainer
                self.explainer = shap.TreeExplainer(self.model)
            
            logger.info(f"Set up SHAP explainer for {self.model_type}")
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {e}")
            raise
    
    def generate_shap_values(
        self, 
        X: np.ndarray, 
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate SHAP values for input data.
        
        Args:
            X: Input data (samples x features)
            sample_indices: Indices of specific samples to explain (optional)
            max_samples: Maximum number of samples to explain
            
        Returns:
            Tuple of (shap_values, data_subset)
        """
        # If explainer not set up yet, initialize with this data
        if self.explainer is None:
            # Use a small subset of data for background
            background_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
            background_data = X[background_indices]
            self._setup_explainer(background_data)
        
        # Select samples to explain
        if sample_indices is not None:
            # Use provided indices
            data_subset = X[sample_indices]
        else:
            # Randomly select samples
            n_samples = min(max_samples, len(X))
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            data_subset = X[sample_indices]
        
        # Generate SHAP values
        try:
            shap_values = self.explainer.shap_values(data_subset)
            logger.info(f"Generated SHAP values for {len(data_subset)} samples")
            return shap_values, data_subset
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise
    
    def save_shap_values(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray, 
        output_dir: Union[str, Path] = "data/processed/shap",
        prefix: str = ""
    ) -> Path:
        """
        Save SHAP values and corresponding data.
        
        Args:
            shap_values: SHAP values to save
            data: Corresponding data samples
            output_dir: Directory to save files
            prefix: Prefix for filenames
            
        Returns:
            Path to the directory where files were saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filenames with model type and optional prefix
        if prefix:
            prefix = f"{prefix}_"
        shap_file = output_dir / f"{prefix}{self.model_type}_shap_values.npy"
        data_file = output_dir / f"{prefix}{self.model_type}_shap_data.npy"
        
        # Save arrays
        np.save(shap_file, shap_values)
        np.save(data_file, data)
        
        logger.info(f"Saved SHAP values and data to {output_dir}")
        return output_dir
    
    def plot_shap_summary(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        output_file: Optional[Union[str, Path]] = None,
        max_display: int = 20,
        show: bool = False
    ) -> None:
        """
        Create and optionally save a SHAP summary plot.
        
        Args:
            shap_values: SHAP values to plot
            data: Corresponding data samples
            output_file: Path to save the plot (optional)
            max_display: Maximum number of features to display
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create plot with feature names if available
        shap.summary_plot(
            shap_values, 
            data, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False  # Don't show yet, we'll save first
        )
        
        # Save if output file specified
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Saved SHAP summary plot to {output_file}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_shap_waterfall(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        sample_index: int = 0,
        output_file: Optional[Union[str, Path]] = None,
        max_display: int = 20,
        show: bool = False
    ) -> None:
        """
        Create and optionally save a SHAP waterfall plot for a specific sample.
        
        Args:
            shap_values: SHAP values to plot
            data: Corresponding data samples
            sample_index: Index of sample to explain
            output_file: Path to save the plot (optional)
            max_display: Maximum number of features to display
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Select sample
        sample_shap = shap_values[sample_index]
        sample_data = data[sample_index]
        
        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=sample_shap,
                data=sample_data,
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        
        # Save if output file specified
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Saved SHAP waterfall plot to {output_file}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close() 