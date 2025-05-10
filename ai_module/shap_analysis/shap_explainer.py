#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified SHAP Explainer Module

This module provides functionality to generate and interpret SHAP explanations
for different model types used in the healthcare anomaly detection system.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import json
import tempfile

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")

# Import from project
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger
from ai_module.utils.visualization import plot_shap_summary, figure_to_base64
from ai_module.models.base_model import BaseModelWrapper, PyTorchModelWrapper, SKLearnModelWrapper

# Initialize logger
logger = get_logger("shap_explainer")

# Get configuration
config = get_config()


class SHAPNotAvailableError(Exception):
    """Exception raised when SHAP is not available."""
    pass


class ModelNotSupportedError(Exception):
    """Exception raised when a model type is not supported for SHAP explanations."""
    pass


class SHAPExplainer:
    """
    A unified interface for generating SHAP explanations for different model types.
    
    This class handles the creation of appropriate SHAP explainers based on the
    model type and provides methods to generate and visualize SHAP values.
    """
    
    def __init__(
        self,
        model: Union[BaseModelWrapper, Any],
        model_type: Optional[str] = None,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Model to explain (either a BaseModelWrapper or the raw model)
            model_type: Type of model (required if raw model provided)
            background_data: Background data for SHAP explainer (required for some explainers)
            feature_names: Names of input features
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError("SHAP package is required but not installed.")
        
        # Store model
        self.model_wrapper = None
        if isinstance(model, BaseModelWrapper):
            self.model_wrapper = model
            self.model = model.model
            self.model_type = model.model_type
        else:
            self.model = model
            self.model_type = model_type
            
            if model_type is None:
                raise ValueError("model_type must be provided when model is not a BaseModelWrapper")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Create explainer based on model type
        self.explainer = self._create_explainer(background_data)
        
        # Store background data
        self.background_data = background_data
        
        logger.info(f"Initialized SHAP explainer for {self.model_type} model")
    
    def _create_explainer(self, background_data: Optional[np.ndarray] = None) -> Any:
        """
        Create the appropriate SHAP explainer based on model type.
        
        Args:
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        # Check if we have PyTorch model
        if isinstance(self.model, torch.nn.Module):
            return self._create_pytorch_explainer(background_data)
        
        # Check for scikit-learn models
        if self.model_type == "isolation_forest":
            return self._create_isolation_forest_explainer(background_data)
        
        # For other scikit-learn models
        try:
            # Try to determine if model is a tree-based model
            if hasattr(self.model, "estimators_") or hasattr(self.model, "tree_"):
                return self._create_tree_explainer(background_data)
            
            # For other models, use KernelExplainer as a fallback
            return self._create_kernel_explainer(background_data)
        except Exception as e:
            logger.error(f"Error creating explainer: {e}")
            # Default to KernelExplainer
            return self._create_kernel_explainer(background_data)
    
    def _create_pytorch_explainer(self, background_data: Optional[np.ndarray] = None) -> Any:
        """
        Create a SHAP explainer for PyTorch models.
        
        Args:
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        # Check if we have AutoEncoder
        if self.model_type == "autoencoder":
            # For AutoEncoder, we use the reconstruction error as the output
            def predict_fn(x):
                if isinstance(x, pd.DataFrame):
                    x = x.values
                
                # Convert to PyTorch tensor
                x_tensor = torch.tensor(x, dtype=torch.float32)
                
                # Get device
                device = next(self.model.parameters()).device
                x_tensor = x_tensor.to(device)
                
                # Set model to eval mode
                self.model.eval()
                
                # Forward pass
                with torch.no_grad():
                    recon, _ = self.model(x_tensor)
                    
                # Calculate reconstruction error (MSE)
                recon_error = torch.mean(torch.pow(x_tensor - recon, 2), dim=1)
                
                return recon_error.cpu().numpy()
            
            # Create explainer using background data or a subset
            if background_data is not None:
                if len(background_data) > 100:
                    # Use a subset to reduce computation
                    indices = np.random.choice(len(background_data), 100, replace=False)
                    background_subset = background_data[indices]
                else:
                    background_subset = background_data
                
                return shap.KernelExplainer(predict_fn, background_subset)
            else:
                # Create a zero background
                logger.warning("No background data provided for SHAP explainer. Using zero background.")
                if hasattr(self.model, "input_dim"):
                    input_dim = self.model.input_dim
                else:
                    # Try to infer input dimension from first layer
                    for module in self.model.modules():
                        if isinstance(module, torch.nn.Linear):
                            input_dim = module.in_features
                            break
                    else:
                        raise ValueError("Could not determine input dimension for model")
                
                zero_background = np.zeros((1, input_dim))
                return shap.KernelExplainer(predict_fn, zero_background)
        
        # For other PyTorch models
        def predict_fn(x):
            if isinstance(x, pd.DataFrame):
                x = x.values
            
            # Convert to PyTorch tensor
            x_tensor = torch.tensor(x, dtype=torch.float32)
            
            # Get device
            device = next(self.model.parameters()).device
            x_tensor = x_tensor.to(device)
            
            # Set model to eval mode
            self.model.eval()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(x_tensor)
                
                # Handle tuple returns
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                return outputs.cpu().numpy()
        
        # Create explainer using background data or a subset
        if background_data is not None:
            if len(background_data) > 100:
                # Use a subset to reduce computation
                indices = np.random.choice(len(background_data), 100, replace=False)
                background_subset = background_data[indices]
            else:
                background_subset = background_data
            
            return shap.KernelExplainer(predict_fn, background_subset)
        else:
            logger.warning("No background data provided for SHAP explainer.")
            return None
    
    def _create_isolation_forest_explainer(self, background_data: Optional[np.ndarray] = None) -> Any:
        """
        Create a SHAP explainer for Isolation Forest models.
        
        Args:
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        # For IsolationForest, use TreeExplainer
        try:
            return shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Error creating TreeExplainer for IsolationForest: {e}")
            
            # Fall back to KernelExplainer
            if background_data is not None:
                # Prepare prediction function
                def predict_fn(x):
                    if isinstance(x, pd.DataFrame):
                        x = x.values
                    
                    # Get decision function scores (lower is more anomalous)
                    scores = self.model.decision_function(x)
                    
                    # Convert to anomaly scores (higher is more anomalous)
                    return -scores
                
                # Use a subset of background data
                if len(background_data) > 100:
                    indices = np.random.choice(len(background_data), 100, replace=False)
                    background_subset = background_data[indices]
                else:
                    background_subset = background_data
                
                return shap.KernelExplainer(predict_fn, background_subset)
            else:
                logger.warning("No background data provided for SHAP KernelExplainer fallback.")
                return None
    
    def _create_tree_explainer(self, background_data: Optional[np.ndarray] = None) -> Any:
        """
        Create a SHAP explainer for tree-based models.
        
        Args:
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        try:
            if background_data is not None:
                return shap.TreeExplainer(self.model, data=background_data)
            else:
                return shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Error creating TreeExplainer: {e}. Falling back to KernelExplainer.")
            return self._create_kernel_explainer(background_data)
    
    def _create_kernel_explainer(self, background_data: Optional[np.ndarray] = None) -> Any:
        """
        Create a SHAP KernelExplainer for generic models.
        
        Args:
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        if background_data is None:
            logger.warning("Background data is required for KernelExplainer but not provided.")
            return None
        
        # Prepare prediction function
        def predict_fn(x):
            if isinstance(x, pd.DataFrame):
                x = x.values
            
            return self.model.predict(x)
        
        # Use a subset of background data to reduce computation
        if len(background_data) > 100:
            indices = np.random.choice(len(background_data), 100, replace=False)
            background_subset = background_data[indices]
        else:
            background_subset = background_data
        
        return shap.KernelExplainer(predict_fn, background_subset)
    
    def explain(
        self, 
        X: np.ndarray, 
        max_samples: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Generate SHAP values for the given samples.
        
        Args:
            X: Samples to explain
            max_samples: Maximum number of samples to explain (for performance reasons)
            **kwargs: Additional parameters to pass to the explainer
            
        Returns:
            SHAP values array
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError("SHAP package is required but not installed.")
        
        if self.explainer is None:
            raise ValueError("SHAP explainer was not properly initialized.")
        
        # Limit number of samples to explain
        if len(X) > max_samples:
            logger.info(f"Limiting SHAP explanation to {max_samples} samples (out of {len(X)})")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        # Generate SHAP values
        try:
            # For tree explainers
            if isinstance(self.explainer, shap.TreeExplainer):
                shap_values = self.explainer.shap_values(X_subset)
                
                # Handle multi-output models
                if isinstance(shap_values, list):
                    # For multi-class, use the mean of absolute values
                    if len(shap_values) > 1:
                        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
                    else:
                        shap_values = shap_values[0]
            
            # For kernel explainers
            elif isinstance(self.explainer, shap.KernelExplainer):
                shap_values = self.explainer.shap_values(X_subset, **kwargs)
                
                # Handle multi-output models
                if isinstance(shap_values, list):
                    # For multi-class, use the mean of absolute values
                    if len(shap_values) > 1:
                        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
                    else:
                        shap_values = shap_values[0]
            
            # For other explainers
            else:
                shap_values = self.explainer.shap_values(X_subset)
            
            logger.info(f"Generated SHAP values with shape {shap_values.shape} for {len(X_subset)} samples")
            return shap_values
            
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise
    
    def get_feature_importance(
        self, 
        shap_values: np.ndarray, 
        aggregate: str = "mean_abs"
    ) -> Dict[str, float]:
        """
        Calculate feature importance based on SHAP values.
        
        Args:
            shap_values: SHAP values array
            aggregate: Aggregation method ('mean_abs', 'mean', 'sum_abs', 'sum')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        else:
            feature_names = self.feature_names
        
        # Aggregate SHAP values
        if aggregate == "mean_abs":
            # Mean of absolute values (most common)
            importance = np.abs(shap_values).mean(axis=0)
        elif aggregate == "mean":
            # Mean of values
            importance = shap_values.mean(axis=0)
        elif aggregate == "sum_abs":
            # Sum of absolute values
            importance = np.abs(shap_values).sum(axis=0)
        elif aggregate == "sum":
            # Sum of values
            importance = shap_values.sum(axis=0)
        else:
            logger.warning(f"Unknown aggregation method: {aggregate}. Using mean_abs.")
            importance = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importance
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return importance_dict
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        plot_type: str = "bar",
        max_display: int = 20,
        title: str = "SHAP Feature Importance",
        show_plot: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        as_base64: bool = False
    ) -> Union[plt.Figure, str, None]:
        """
        Create and optionally save a SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            X: Input feature values
            plot_type: Type of plot ('bar', 'dot', or 'violin')
            max_display: Maximum number of features to display
            title: Plot title
            show_plot: Whether to show the plot
            save_path: Path to save the plot
            as_base64: Whether to return the plot as a base64 encoded string
            
        Returns:
            matplotlib Figure, base64 string, or None
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError("SHAP package is required but not installed.")
        
        # Get feature names
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
        
        # Create plot
        result = plot_shap_summary(
            shap_values=shap_values,
            features=X,
            feature_names=feature_names,
            title=title,
            plot_type=plot_type,
            max_display=max_display,
            save_path=save_path,
            as_base64=as_base64
        )
        
        # Show plot if requested
        if show_plot and not as_base64:
            plt.show()
        elif not show_plot and not as_base64:
            plt.close()
        
        return result
    
    def get_top_features(
        self, 
        shap_values: np.ndarray, 
        sample_idx: int = 0, 
        top_n: int = 5, 
        abs_values: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get the top contributing features for a specific sample.
        
        Args:
            shap_values: SHAP values array
            sample_idx: Index of the sample to explain
            top_n: Number of top features to return
            abs_values: Whether to use absolute values for ranking
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        else:
            feature_names = self.feature_names
        
        # Get SHAP values for the sample
        sample_shap = shap_values[sample_idx]
        
        # Create feature-value pairs
        feature_values = list(zip(feature_names, sample_shap))
        
        # Sort by importance
        if abs_values:
            feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            feature_values.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        top_features = feature_values[:top_n]
        
        return top_features
    
    def save_explainer(self, path: Union[str, Path]) -> str:
        """
        Save SHAP explainer to a file.
        
        Args:
            path: Path to save the explainer
            
        Returns:
            Path where the explainer was saved
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError("SHAP package is required but not installed.")
        
        # Convert path to Path object
        path = Path(path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Some explainers can't be directly saved, so we'll save only what we need
        metadata = {
            "model_type": self.model_type,
            "feature_names": self.feature_names
        }
        
        # Save metadata
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved SHAP explainer metadata to {path}")
        return str(path)
    
    @classmethod
    def load_explainer(
        cls,
        path: Union[str, Path],
        model: Union[BaseModelWrapper, Any],
        background_data: Optional[np.ndarray] = None
    ) -> "SHAPExplainer":
        """
        Load SHAP explainer from a file.
        
        Args:
            path: Path to the saved explainer
            model: Model to explain
            background_data: Background data for the explainer
            
        Returns:
            SHAPExplainer instance
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError("SHAP package is required but not installed.")
        
        # Convert path to Path object
        path = Path(path)
        
        # Load metadata
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        # Extract feature names
        feature_names = metadata.get("feature_names")
        
        # Create new explainer
        return cls(model, background_data=background_data, feature_names=feature_names)


# Factory function to create an explainer for a specific model
def get_explainer(
    model: Union[BaseModelWrapper, Any],
    model_type: Optional[str] = None,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> SHAPExplainer:
    """
    Create a SHAP explainer for the specified model.
    
    Args:
        model: Model to explain
        model_type: Type of model (required if model is not a BaseModelWrapper)
        background_data: Background data for the explainer
        feature_names: Names of input features
        
    Returns:
        Configured SHAPExplainer instance
    """
    if not SHAP_AVAILABLE:
        raise SHAPNotAvailableError("SHAP package is required but not installed.")
    
    return SHAPExplainer(model, model_type, background_data, feature_names) 