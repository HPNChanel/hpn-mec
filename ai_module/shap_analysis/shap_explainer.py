#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified SHAP Explainer for AI Module

This module provides a centralized implementation for generating SHAP explanations
for all model types in the system. It supports PyTorch models (AutoEncoder) and
scikit-learn models (Isolation Forest) through a consistent interface.

Author: Huỳnh Phúc Nguyên
Created: June 2025
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
from datetime import datetime

# Import project modules
from ai_module.models.base_model import BaseModelWrapper, PyTorchModelWrapper, SKLearnModelWrapper
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("shap_analysis.shap_explainer")

# Get configuration
config = get_config()

class SHAPExplainer:
    """
    Unified SHAP explainer for all model types.
    
    This class provides a standardized interface for generating SHAP explanations
    for different types of models (PyTorch, scikit-learn) through a consistent API.
    """
    
    def __init__(
        self, 
        model: BaseModelWrapper,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
        max_display: int = 20
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Model wrapper (must implement BaseModelWrapper)
            feature_names: List of feature names (optional)
            background_data: Background data for SHAP explainer (optional)
            output_dir: Directory to save SHAP outputs (optional)
            max_display: Maximum number of features to display in plots
        """
        self.model = model
        self.model_type = model.model_type
        self.feature_names = feature_names
        self.explainer = None
        self.max_display = max_display
        
        # Set output directory
        self.output_dir = Path(output_dir) if output_dir else Path(config.get("paths.shap_dir", "reports/shap"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model metadata for more context
        self.metadata = model.get_metadata()
        
        # Set up SHAP explainer if background data is provided
        if background_data is not None:
            self.setup_explainer(background_data)
            
        logger.info(f"Initialized SHAP Explainer for {self.model_type}")
    
    def setup_explainer(self, background_data: np.ndarray) -> None:
        """
        Set up the SHAP explainer based on model type.
        
        Args:
            background_data: Background data for SHAP explainer
        
        Raises:
            TypeError: If model type is not supported
        """
        try:
            if isinstance(self.model, PyTorchModelWrapper):
                self._setup_pytorch_explainer(background_data)
            elif isinstance(self.model, SKLearnModelWrapper):
                self._setup_sklearn_explainer(background_data)
            else:
                raise TypeError(f"Unsupported model type: {type(self.model).__name__}")
                
            logger.info(f"Set up SHAP explainer for {self.model_type}")
            
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {e}")
            raise
    
    def _setup_pytorch_explainer(self, background_data: np.ndarray) -> None:
        """
        Set up SHAP explainer for PyTorch models.
        
        Args:
            background_data: Background data for SHAP explainer
        """
        # For PyTorch models, we have two approaches depending on the model architecture
        
        # For autoencoder, we want to explain the reconstruction error
        if self.model_type.lower() == 'autoencoder':
            # Define a function to compute reconstruction error
            def reconstruction_error(x):
                # Convert to tensor if numpy
                if isinstance(x, np.ndarray):
                    x_tensor = torch.FloatTensor(x)
                    if torch.cuda.is_available():
                        x_tensor = x_tensor.cuda()
                else:
                    x_tensor = x
                
                with torch.no_grad():
                    reconstructed = self.model.model(x_tensor)[0]
                    error = torch.mean((reconstructed - x_tensor) ** 2, dim=1)
                
                return error.cpu().numpy()
            
            # Use KernelExplainer since we're explaining a custom function
            self.explainer = shap.KernelExplainer(reconstruction_error, background_data)
            
        # For other PyTorch models, use DeepExplainer or GradientExplainer if possible
        else:
            # Check if model has predict_proba method (PyTorch classifier)
            if hasattr(self.model, 'predict_proba'):
                # Define a function to get probabilities
                def predict_proba_fn(x):
                    if isinstance(x, np.ndarray):
                        x_tensor = torch.FloatTensor(x)
                        if torch.cuda.is_available():
                            x_tensor = x_tensor.cuda()
                    else:
                        x_tensor = x
                    
                    with torch.no_grad():
                        probas = self.model.predict_proba(x_tensor)
                    
                    return probas
                
                # Use KernelExplainer for the probabilities
                self.explainer = shap.KernelExplainer(predict_proba_fn, background_data)
            else:
                # Try to use DeepExplainer if possible
                try:
                    # Prepare background data as tensor
                    background_tensor = torch.FloatTensor(background_data)
                    if torch.cuda.is_available():
                        background_tensor = background_tensor.cuda()
                    
                    # Use DeepExplainer
                    self.explainer = shap.DeepExplainer(self.model.model, background_tensor)
                except Exception as e:
                    logger.warning(f"Could not initialize DeepExplainer: {e}")
                    logger.warning("Falling back to KernelExplainer")
                    
                    # Fall back to KernelExplainer
                    def predict_fn(x):
                        if isinstance(x, np.ndarray):
                            x_tensor = torch.FloatTensor(x)
                            if torch.cuda.is_available():
                                x_tensor = x_tensor.cuda()
                        else:
                            x_tensor = x
                        
                        with torch.no_grad():
                            preds = self.model.predict(x_tensor)
                        
                        return preds
                    
                    self.explainer = shap.KernelExplainer(predict_fn, background_data)
    
    def _setup_sklearn_explainer(self, background_data: np.ndarray) -> None:
        """
        Set up SHAP explainer for scikit-learn models.
        
        Args:
            background_data: Background data for SHAP explainer
        """
        # Get the underlying scikit-learn model
        sklearn_model = self.model.model
        
        # For tree-based models, use the TreeExplainer
        if self.model_type.lower() in ['isolation_forest', 'randomforest', 'xgboost', 'lightgbm']:
            self.explainer = shap.TreeExplainer(sklearn_model)
            
        # For linear models, use LinearExplainer
        elif self.model_type.lower() in ['linearregression', 'logisticregression', 'ridge', 'lasso']:
            self.explainer = shap.LinearExplainer(sklearn_model, background_data)
            
        # For other models, use KernelExplainer
        else:
            if hasattr(sklearn_model, 'predict_proba'):
                # For classifiers, use predict_proba
                self.explainer = shap.KernelExplainer(sklearn_model.predict_proba, background_data)
            else:
                # For other models, use plain predict
                self.explainer = shap.KernelExplainer(sklearn_model.predict, background_data)
    
    def explain(
        self, 
        X: np.ndarray, 
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 100,
        batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate SHAP values for input data.
        
        Args:
            X: Input data (samples x features)
            sample_indices: Indices of specific samples to explain (optional)
            max_samples: Maximum number of samples to explain
            batch_size: Batch size for processing large inputs (optional)
            
        Returns:
            Tuple of (shap_values, data_subset)
        """
        # If explainer not set up yet, initialize with this data
        if self.explainer is None:
            # Use a small subset of data for background
            n_background = min(100, len(X))
            background_indices = np.random.choice(len(X), n_background, replace=False)
            background_data = X[background_indices]
            self.setup_explainer(background_data)
        
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
            n_samples = len(data_subset)
            logger.info(f"Generating SHAP values for {n_samples} samples")
            
            # Process in batches if batch_size is provided and data is large enough
            if batch_size is not None and n_samples > batch_size:
                all_shap_values = []
                
                # Process in batches
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batch = data_subset[i:end_idx]
                    
                    logger.info(f"Processing batch {i//batch_size + 1}: samples {i} to {end_idx}")
                    
                    # Compute SHAP values for batch
                    batch_values = self.explainer.shap_values(batch)
                    
                    # Handle different return types from SHAP
                    if isinstance(batch_values, list) and len(batch_values) == 1:
                        batch_values = batch_values[0]
                    
                    if isinstance(batch_values, np.ndarray):
                        all_shap_values.append(batch_values)
                    elif isinstance(batch_values, list):
                        # For multi-output models, take the first output
                        all_shap_values.append(batch_values[0])
                
                # Combine batches
                shap_values = np.vstack(all_shap_values)
                
            else:
                # For small datasets, process all at once
                shap_values = self.explainer.shap_values(data_subset)
                
                # Handle different return types from SHAP
                if isinstance(shap_values, list) and len(shap_values) == 1:
                    shap_values = shap_values[0]
                elif isinstance(shap_values, list) and len(shap_values) > 1:
                    # For multi-output models, take the first output
                    shap_values = shap_values[0]
            
            logger.info(f"Generated SHAP values with shape {shap_values.shape}")
            
            return shap_values, data_subset
            
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise
    
    def plot_summary(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        output_file: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> None:
        """
        Plot SHAP summary (beeswarm plot).
        
        Args:
            shap_values: SHAP values from explain()
            data: Input data corresponding to the SHAP values
            output_file: Path to save the plot (optional)
            show: Whether to display the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create summary plot
            feature_names = self.feature_names if self.feature_names else None
            shap.summary_plot(
                shap_values, 
                data, 
                feature_names=feature_names,
                max_display=self.max_display,
                show=False
            )
            
            # Add title
            plt.title(f"SHAP Summary for {self.model_type}")
            
            # Save if output file specified
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP summary plot to {output_file}")
            else:
                # Generate automatic filename
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{self.model_type}_shap_summary_{timestamp}.png"
                path = self.output_dir / filename
                plt.savefig(path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP summary plot to {path}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            raise
    
    def plot_waterfall(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        sample_index: int = 0,
        output_file: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> None:
        """
        Plot SHAP waterfall for a single sample.
        
        Args:
            shap_values: SHAP values from explain()
            data: Input data corresponding to the SHAP values
            sample_index: Index of the sample to explain
            output_file: Path to save the plot (optional)
            show: Whether to display the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Extract the sample and its SHAP values
            sample_data = data[sample_index:sample_index+1]
            sample_shap = shap_values[sample_index]
            
            # Create waterfall plot
            feature_names = self.feature_names if self.feature_names else None
            shap.plots.waterfall(
                shap.Explanation(
                    values=sample_shap,
                    base_values=self.explainer.expected_value 
                        if hasattr(self.explainer, 'expected_value') else 0,
                    data=sample_data[0],
                    feature_names=feature_names
                ),
                max_display=self.max_display,
                show=False
            )
            
            # Add title
            plt.title(f"SHAP Waterfall for {self.model_type} - Sample {sample_index}")
            
            # Save if output file specified
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP waterfall plot to {output_file}")
            else:
                # Generate automatic filename
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{self.model_type}_shap_waterfall_sample{sample_index}_{timestamp}.png"
                path = self.output_dir / filename
                plt.savefig(path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP waterfall plot to {path}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            raise
    
    def plot_force(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        sample_index: int = 0,
        output_file: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> None:
        """
        Plot SHAP force plot for a single sample.
        
        Args:
            shap_values: SHAP values from explain()
            data: Input data corresponding to the SHAP values
            sample_index: Index of the sample to explain
            output_file: Path to save the plot (optional)
            show: Whether to display the plot
        """
        try:
            # Extract the sample and its SHAP values
            sample_data = data[sample_index:sample_index+1]
            sample_shap = shap_values[sample_index:sample_index+1]
            
            # Create force plot
            feature_names = self.feature_names if self.feature_names else None
            force_plot = shap.force_plot(
                base_value=self.explainer.expected_value 
                    if hasattr(self.explainer, 'expected_value') else 0,
                shap_values=sample_shap,
                features=sample_data,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Save if output file specified
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP force plot to {output_file}")
            else:
                # Generate automatic filename
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{self.model_type}_shap_force_sample{sample_index}_{timestamp}.png"
                path = self.output_dir / filename
                plt.savefig(path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP force plot to {path}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP force plot: {e}")
            raise
    
    def plot_dependence(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray,
        feature_idx: int,
        interaction_idx: Optional[int] = None,
        output_file: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values from explain()
            data: Input data corresponding to the SHAP values
            feature_idx: Index of the feature to plot
            interaction_idx: Index of the interaction feature (optional)
            output_file: Path to save the plot (optional)
            show: Whether to display the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Get feature name
            feature_names = self.feature_names if self.feature_names else None
            feature_name = feature_names[feature_idx] if feature_names else f"feature {feature_idx}"
            
            # Create dependence plot
            if interaction_idx is not None:
                interaction_feature = feature_names[interaction_idx] if feature_names else f"feature {interaction_idx}"
                shap.dependence_plot(
                    feature_idx, 
                    shap_values, 
                    data,
                    interaction_index=interaction_idx,
                    feature_names=feature_names,
                    show=False
                )
                title = f"SHAP Dependence: {feature_name} with {interaction_feature} interaction"
            else:
                shap.dependence_plot(
                    feature_idx, 
                    shap_values, 
                    data,
                    feature_names=feature_names,
                    show=False
                )
                title = f"SHAP Dependence: {feature_name}"
            
            # Add title
            plt.title(title)
            
            # Save if output file specified
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP dependence plot to {output_file}")
            else:
                # Generate automatic filename
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                feature_str = f"{feature_idx}"
                interaction_str = f"_interaction{interaction_idx}" if interaction_idx is not None else ""
                filename = f"{self.model_type}_shap_dependence_f{feature_str}{interaction_str}_{timestamp}.png"
                path = self.output_dir / filename
                plt.savefig(path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved SHAP dependence plot to {path}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            raise
    
    def get_top_features(
        self, 
        shap_values: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get the top-k most important features based on mean absolute SHAP value.
        
        Args:
            shap_values: SHAP values from explain()
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, importance_value) tuples
        """
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create feature importance tuples
        importance_tuples = [(feature_names[i], feature_importance[i]) for i in range(len(feature_importance))]
        
        # Sort by importance (descending)
        sorted_importance = sorted(importance_tuples, key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return sorted_importance[:top_k]
    
    def explain_sample(
        self, 
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        generate_plots: bool = True,
        save_values: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive explanation for a single sample.
        
        Args:
            X: Input sample (can be a single sample or a batch with multiple samples)
            feature_names: List of feature names (optional)
            output_dir: Directory to save outputs (optional)
            generate_plots: Whether to generate plots
            save_values: Whether to save SHAP values
            
        Returns:
            Dictionary of explanation results
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Set feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Set output directory if provided
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate SHAP values
        shap_values, data = self.explain(X)
        
        # Get top features
        top_features = self.get_top_features(shap_values)
        
        results = {
            "model_type": self.model_type,
            "shap_values": shap_values,
            "data": data,
            "top_features": top_features,
            "plots": {}
        }
        
        # Generate plots if requested
        if generate_plots:
            # Summary plot
            summary_path = self.output_dir / f"{self.model_type}_summary.png"
            self.plot_summary(shap_values, data, summary_path)
            results["plots"]["summary"] = str(summary_path)
            
            # Waterfall plot for the first sample
            waterfall_path = self.output_dir / f"{self.model_type}_waterfall.png"
            self.plot_waterfall(shap_values, data, 0, waterfall_path)
            results["plots"]["waterfall"] = str(waterfall_path)
            
            # Force plot for the first sample
            force_path = self.output_dir / f"{self.model_type}_force.png"
            self.plot_force(shap_values, data, 0, force_path)
            results["plots"]["force"] = str(force_path)
            
            # Dependence plot for the top feature
            if top_features:
                top_feature_idx = list(self.feature_names).index(top_features[0][0])
                dependence_path = self.output_dir / f"{self.model_type}_dependence.png"
                self.plot_dependence(shap_values, data, top_feature_idx, None, dependence_path)
                results["plots"]["dependence"] = str(dependence_path)
        
        # Save SHAP values if requested
        if save_values:
            # Save SHAP values
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            shap_path = self.output_dir / f"{self.model_type}_shap_values_{timestamp}.npy"
            np.save(shap_path, shap_values)
            results["shap_values_path"] = str(shap_path)
            
            # Save top features to text file
            top_features_path = self.output_dir / f"{self.model_type}_top_features_{timestamp}.txt"
            with open(top_features_path, "w") as f:
                for feature, importance in top_features:
                    f.write(f"{feature}: {importance:.6f}\n")
            results["top_features_path"] = str(top_features_path)
        
        return results
    
    def save_shap_values(
        self, 
        shap_values: np.ndarray, 
        data: np.ndarray, 
        output_file: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save SHAP values and data to disk.
        
        Args:
            shap_values: SHAP values from explain()
            data: Input data corresponding to the SHAP values
            output_file: Path to save the values (optional)
            
        Returns:
            Path where values were saved
        """
        # Generate output file path if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = self.output_dir / f"{self.model_type}_shap_values_{timestamp}.npz"
        else:
            output_file = Path(output_file)
        
        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as npz file
        np.savez(
            output_file,
            shap_values=shap_values,
            data=data,
            model_type=self.model_type,
            feature_names=self.feature_names
        )
        
        logger.info(f"Saved SHAP values to {output_file}")
        
        return output_file
    
    @classmethod
    def load_shap_values(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load saved SHAP values from disk.
        
        Args:
            path: Path to the saved npz file
            
        Returns:
            Dictionary with loaded values
        """
        # Load npz file
        with np.load(path) as data:
            result = {key: data[key] for key in data.files}
            
        logger.info(f"Loaded SHAP values from {path}")
        
        return result 