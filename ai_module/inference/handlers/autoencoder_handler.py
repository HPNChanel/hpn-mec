#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Handler for AutoEncoder Model

This module provides utilities for loading the trained AutoEncoder model
and running inference to encode structured health data into latent space
and reconstruct it.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import the AutoEncoder class if it's defined in a module
try:
    from ...models.autoencoder import Autoencoder
    logger.info("Imported AutoEncoder class from models.autoencoder")
except ImportError:
    # If not found, define the AutoEncoder class here
    logger.info("AutoEncoder class not found, defining locally")
    
    class AutoEncoder(nn.Module):
        """
        AutoEncoder neural network for dimensionality reduction and reconstruction.
        
        This model encodes the input into a lower-dimensional latent space
        and then decodes it back to the original space.
        """
        
        def __init__(self, input_dim, latent_dim=16, dropout=0.2, use_sigmoid=False):
            """
            Initialize autoencoder with configurable dimensions and dropout.
            
            Args:
                input_dim (int): Dimension of input features
                latent_dim (int): Dimension of latent space (bottleneck)
                dropout (float): Dropout rate for regularization
                use_sigmoid (bool): Whether to use sigmoid activation on output
            """
            super(AutoEncoder, self).__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.use_sigmoid = use_sigmoid
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, latent_dim),
                nn.ReLU()
            )
            
            # Decoder
            decoder_layers = [
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, input_dim)
            ]
            
            # Add sigmoid if requested (useful if input data is normalized to [0,1])
            if use_sigmoid:
                decoder_layers.append(nn.Sigmoid())
                
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            """
            Forward pass through the autoencoder.
            
            Args:
                x: Input tensor
                
            Returns:
                tuple: (reconstructed_output, latent_vector)
            """
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent

def get_device():
    """
    Get the appropriate device for inference (CPU or CUDA).
    
    Returns:
        torch.device: Device to use for inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    return device

def load_model(model_path: Optional[str] = None) -> nn.Module:
    """
    Load the trained AutoEncoder model.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        nn.Module: Loaded AutoEncoder model in evaluation mode
    """
    try:
        device = get_device()
        
        # Use Path for file path resolution
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "autoencoder.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            # Check if there are alternative model files
            model_dir = model_path.parent
            alternate_models = list(model_dir.glob("*autoencoder*.pt"))
            
            if alternate_models:
                # Use the first alternative model
                model_path = alternate_models[0]
                logger.warning(f"Using alternative model: {model_path}")
            else:
                raise FileNotFoundError(f"No model files found in {model_dir}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Try different loading methods, as the model might have been saved in different ways
        try:
            # Method 1: Try loading the whole model
            model = torch.load(model_path, map_location=device)
            logger.info("Model loaded successfully (entire model)")
        except:
            # Method 2: Try loading state dict into a new model
            # We need to determine input_dim and latent_dim from the state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Try to infer dimensions from state dict
            try:
                # Get dimensions from the first and last layer weights
                encoder_first_layer = [k for k in state_dict.keys() if 'encoder.0.weight' in k][0]
                decoder_last_layer = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k][-1]
                
                input_dim = state_dict[encoder_first_layer].shape[1]
                
                # Find the latent dimension (harder to determine)
                # Look for the last encoder layer
                encoder_layers = [k for k in state_dict.keys() if 'encoder' in k and 'weight' in k]
                decoder_layers = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
                
                if encoder_layers and decoder_layers:
                    encoder_last_layer = encoder_layers[-1]
                    decoder_first_layer = decoder_layers[0]
                    
                    latent_dim = state_dict[encoder_last_layer].shape[0]
                    
                    # Verify with decoder's first layer input dim
                    if state_dict[decoder_first_layer].shape[1] != latent_dim:
                        logger.warning("Inconsistent latent dimension, using encoder's output")
                else:
                    # Default dimensions if can't determine
                    logger.warning("Couldn't determine dimensions from state dict, using defaults")
                    input_dim = 10
                    latent_dim = 16
            except:
                # Default dimensions if anything fails
                logger.warning("Error determining dimensions from state dict, using defaults")
                input_dim = 10
                latent_dim = 16
            
            # Create a new model with the inferred dimensions
            model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
            
            # Load the state dict
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully (state dict) with input_dim={input_dim}, latent_dim={latent_dim}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Move model to the appropriate device
        model = model.to(device)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def encode_and_reconstruct(
    model: nn.Module, 
    X: np.ndarray, 
    batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode data to latent space and reconstruct it using the autoencoder.
    
    Args:
        model: Trained AutoEncoder model
        X: Input feature matrix (normalized)
        batch_size: Batch size for processing
        
    Returns:
        tuple: (latent_vectors, reconstructions, reconstruction_errors)
    """
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Get device
        device = next(model.parameters()).device
        
        # Convert to float32 if not already
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Calculate number of batches
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Initialize arrays to store results
        latent_vectors = []
        reconstructions = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch = X[start_idx:end_idx]
                
                # Convert to tensor and move to device
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                
                # Get reconstruction and latent vector
                recon_batch, latent_batch = model(batch_tensor)
                
                # Move back to CPU and convert to numpy
                latent_vectors.append(latent_batch.cpu().numpy())
                reconstructions.append(recon_batch.cpu().numpy())
        
        # Concatenate results
        latent_vectors = np.vstack(latent_vectors)
        reconstructions = np.vstack(reconstructions)
        
        # Calculate reconstruction errors (mean squared error per sample)
        reconstruction_errors = np.mean((X - reconstructions) ** 2, axis=1)
        
        logger.info(f"Processed {n_samples} samples in {n_batches} batches")
        logger.info(f"Latent vectors shape: {latent_vectors.shape}")
        logger.info(f"Reconstructions shape: {reconstructions.shape}")
        logger.info(f"Reconstruction errors shape: {reconstruction_errors.shape}")
        
        return latent_vectors, reconstructions, reconstruction_errors
        
    except Exception as e:
        logger.error(f"Error in encode_and_reconstruct: {e}")
        raise

def encode(
    model: nn.Module, 
    X: np.ndarray, 
    batch_size: int = 128
) -> np.ndarray:
    """
    Encode data to latent space only (without reconstruction).
    
    Args:
        model: Trained AutoEncoder model
        X: Input feature matrix (normalized)
        batch_size: Batch size for processing
        
    Returns:
        np.ndarray: Latent vectors
    """
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Get device
        device = next(model.parameters()).device
        
        # Convert to float32 if not already
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Calculate number of batches
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Initialize array to store results
        latent_vectors = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch = X[start_idx:end_idx]
                
                # Convert to tensor and move to device
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                
                # Get latent vector only
                _, latent_batch = model(batch_tensor)
                
                # Move back to CPU and convert to numpy
                latent_vectors.append(latent_batch.cpu().numpy())
        
        # Concatenate results
        latent_vectors = np.vstack(latent_vectors)
        
        logger.info(f"Encoded {n_samples} samples to shape {latent_vectors.shape}")
        
        return latent_vectors
        
    except Exception as e:
        logger.error(f"Error in encode: {e}")
        raise

def reconstruct(
    model: nn.Module, 
    latent_vectors: np.ndarray, 
    batch_size: int = 128
) -> np.ndarray:
    """
    Reconstruct data from latent vectors.
    
    Args:
        model: Trained AutoEncoder model
        latent_vectors: Latent vectors to reconstruct from
        batch_size: Batch size for processing
        
    Returns:
        np.ndarray: Reconstructed data
    """
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Get device
        device = next(model.parameters()).device
        
        # Convert to float32 if not already
        if latent_vectors.dtype != np.float32:
            latent_vectors = latent_vectors.astype(np.float32)
        
        # Calculate number of batches
        n_samples = latent_vectors.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Initialize array to store results
        reconstructions = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch = latent_vectors[start_idx:end_idx]
                
                # Convert to tensor and move to device
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                
                # Get reconstruction
                recon_batch = model.decoder(batch_tensor)
                
                # Move back to CPU and convert to numpy
                reconstructions.append(recon_batch.cpu().numpy())
        
        # Concatenate results
        reconstructions = np.vstack(reconstructions)
        
        logger.info(f"Reconstructed {n_samples} samples to shape {reconstructions.shape}")
        
        return reconstructions
        
    except Exception as e:
        logger.error(f"Error in reconstruct: {e}")
        raise

# Simple test function for the handler
def test_handler():
    """Test the AutoEncoder handler with sample data."""
    try:
        # Load model
        model = load_model()
        
        # Create sample data
        input_dim = model.input_dim if hasattr(model, 'input_dim') else 10
        X = np.random.rand(10, input_dim).astype(np.float32)
        
        # Encode and reconstruct
        latent_vectors, reconstructions, errors = encode_and_reconstruct(model, X)
        
        logger.info("Test successful!")
        logger.info(f"Sample reconstruction error: {errors[0]:.6f}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Add parent directories to path if running as script
    if __package__ is None:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    # Run test if executed directly
    test_handler()