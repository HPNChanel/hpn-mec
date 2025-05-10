#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loading Utilities

This module provides functions to load trained models from standardized locations,
supporting both PyTorch and scikit-learn models.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import torch
import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

# Import project modules
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("model_loader")

# Get model directory from config
config = get_config()
DEFAULT_MODEL_DIR = Path(config.get("model.directory", "models"))


def find_latest_model(
    model_type: str,
    model_dir: Optional[Union[str, Path]] = None,
    extension: Optional[str] = None,
    prefix: Optional[str] = None
) -> Optional[Path]:
    """
    Find the latest model file for the specified model type.
    
    Args:
        model_type: Type of model to find (e.g., 'autoencoder', 'isolation_forest')
        model_dir: Directory to search in (default: from config)
        extension: File extension to filter by (default: infer from model_type)
        prefix: File prefix to filter by (default: model_type)
        
    Returns:
        Path to the latest model file or None if not found
    """
    # Get model directory
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = Path(model_dir)
    
    # Infer extension if not provided
    if extension is None:
        # Default extensions for common model types
        if model_type in ['autoencoder', 'lstm', 'vae', 'nn']:
            extension = '.pt'
        else:
            extension = '.joblib'
    
    # Set prefix if not provided
    if prefix is None:
        prefix = model_type
    
    # Ensure directory exists
    if not model_dir.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return None
    
    # Find all matching files
    pattern = f"{prefix}*{extension}"
    model_files = list(model_dir.glob(pattern))
    
    # Filter out checkpoint files (with "epoch" in name) if there are other options
    regular_files = [f for f in model_files if "epoch" not in f.stem]
    if regular_files:
        model_files = regular_files
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not model_files:
        logger.warning(f"No {model_type} model files found in {model_dir} matching {pattern}")
        return None
    
    logger.info(f"Found latest {model_type} model: {model_files[0]}")
    return model_files[0]


def load_pytorch_model(
    model_path: Union[str, Path],
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> torch.nn.Module:
    """
    Load a PyTorch model from a file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model onto (default: None, uses CUDA if available)
        eval_mode: Whether to set the model to evaluation mode
        
    Returns:
        Loaded PyTorch model
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert path to Path object
    model_path = Path(model_path)
    
    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load model
        model = torch.load(model_path, map_location=device)
        
        # Set to evaluation mode if requested
        if eval_mode:
            model.eval()
        
        logger.info(f"Loaded PyTorch model from {model_path} to {device}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading PyTorch model from {model_path}: {e}")
        raise


def load_sklearn_model(model_path: Union[str, Path]) -> Any:
    """
    Load a scikit-learn model from a file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded scikit-learn model
    """
    # Convert path to Path object
    model_path = Path(model_path)
    
    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Loaded scikit-learn model from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading scikit-learn model from {model_path}: {e}")
        raise


def load_model_metadata(
    model_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load metadata associated with a model.
    
    Args:
        model_path: Path to the model file
        metadata_path: Path to the metadata file (default: infer from model_path)
        
    Returns:
        Dictionary of metadata or empty dict if not found
    """
    # Convert paths to Path objects
    model_path = Path(model_path)
    
    # Infer metadata path if not provided
    if metadata_path is None:
        # Try different potential metadata paths
        potential_paths = [
            model_path.with_suffix('.json'),  # model.pt -> model.json
            model_path.parent / f"{model_path.stem}_metadata.json",  # model.pt -> model_metadata.json
            model_path.parent / "metadata" / f"{model_path.stem}.json"  # models/model.pt -> models/metadata/model.json
        ]
        
        # Find first existing path
        for path in potential_paths:
            if path.exists():
                metadata_path = path
                break
        
        if metadata_path is None:
            logger.warning(f"No metadata file found for model: {model_path}")
            return {}
    else:
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            logger.warning(f"Specified metadata file not found: {metadata_path}")
            return {}
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded model metadata from {metadata_path}")
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading model metadata from {metadata_path}: {e}")
        return {}


def load_model(
    model_type: str,
    model_path: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    with_metadata: bool = False
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Load a model of the specified type.
    
    This function automatically determines the appropriate loading function
    based on the model type or file extension.
    
    Args:
        model_type: Type of model to load (e.g., 'autoencoder', 'isolation_forest')
        model_path: Specific path to the model file (default: find latest)
        model_dir: Directory to search for model if model_path not provided
        device: Device to load PyTorch models onto
        with_metadata: Whether to return metadata along with the model
        
    Returns:
        If with_metadata=False: Loaded model
        If with_metadata=True: Tuple of (loaded_model, metadata_dict)
    """
    # If model_path is not provided, find latest model
    if model_path is None:
        model_path = find_latest_model(model_type, model_dir)
        if model_path is None:
            raise FileNotFoundError(f"No {model_type} model found")
    else:
        model_path = Path(model_path)
    
    # Choose loading function based on file extension
    if model_path.suffix.lower() == '.pt':
        model = load_pytorch_model(model_path, device)
    else:
        model = load_sklearn_model(model_path)
    
    # Load metadata if requested
    if with_metadata:
        metadata = load_model_metadata(model_path)
        return model, metadata
    else:
        return model


def list_available_models(
    model_dir: Optional[Union[str, Path]] = None,
    model_type: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all available models and their metadata.
    
    Args:
        model_dir: Directory to search in (default: from config)
        model_type: Type of model to filter by (default: all)
        
    Returns:
        Dictionary mapping model types to lists of model info dictionaries
    """
    # Get model directory
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = Path(model_dir)
    
    # Ensure directory exists
    if not model_dir.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return {}
    
    # Find all model files
    pytorch_models = list(model_dir.glob("*.pt"))
    sklearn_models = list(model_dir.glob("*.joblib"))
    all_models = pytorch_models + sklearn_models
    
    # Initialize result dictionary
    result = {}
    
    # Process each model
    for model_path in all_models:
        try:
            # Extract model type from filename
            model_name = model_path.stem
            
            # Skip checkpoint files if listing all models
            if model_type is None and "epoch" in model_name:
                continue
            
            # Determine model type
            detected_type = model_name.split('_')[0]
            
            # Skip if filtering by type and not matching
            if model_type is not None and model_type != detected_type:
                continue
            
            # Get model metadata
            metadata = load_model_metadata(model_path)
            
            # Add basic file info
            model_info = {
                "name": model_name,
                "path": str(model_path),
                "size": model_path.stat().st_size,
                "modified": model_path.stat().st_mtime,
                "metadata": metadata
            }
            
            # Add to result
            if detected_type not in result:
                result[detected_type] = []
            
            result[detected_type].append(model_info)
        
        except Exception as e:
            logger.warning(f"Error processing model {model_path}: {e}")
    
    # Sort model lists by modification time (newest first)
    for model_type in result:
        result[model_type].sort(key=lambda m: m["modified"], reverse=True)
    
    return result 