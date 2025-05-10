#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder Training for Health Data Anomaly Detection

This script trains an autoencoder neural network on preprocessed
health data features, with the goal of detecting anomalies via
reconstruction error.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader, random_split
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from project modules
from ai_module.models.autoencoder import AutoEncoder
from ai_module.data_loader import DataLoader

# Default paths
MODELS_DIR = Path(__file__).parent.parent / "models"
LATENTS_DIR = Path("data/processed/latents")


def build_autoencoder(input_dim: int, latent_dim: int = 16, dropout: float = 0.2, use_sigmoid: bool = False) -> AutoEncoder:
    """
    Build an autoencoder model with the specified dimensions
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        dropout: Dropout rate for regularization
        use_sigmoid: Whether to use sigmoid activation on output
        
    Returns:
        AutoEncoder: Initialized model
    """
    model = AutoEncoder(input_dim, latent_dim, dropout, use_sigmoid)
    return model


def train_model(
    model: nn.Module, 
    train_loader: TorchDataLoader, 
    val_loader: Optional[TorchDataLoader] = None, 
    epochs: int = 50, 
    lr: float = 0.001, 
    device: str = 'cpu', 
    save_interval: int = 10, 
    model_path: Union[str, Path] = 'model.pt',
    max_grad_norm: float = 1.0,
    weight_decay: float = 1e-5
) -> Tuple[nn.Module, List[float]]:
    """
    Train the autoencoder model
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        save_interval: Save checkpoint every N epochs (0 to disable)
        model_path: Path to save the model
        max_grad_norm: Maximum norm for gradient clipping
        weight_decay: L2 regularization factor
        
    Returns:
        tuple: (trained_model, losses)
    """
    # Move model to device
    model.to(device)
    
    # Initialize model weights properly to prevent vanishing/exploding gradients
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Lists to store losses
    train_losses = []
    
    # Create directory for model if it doesn't exist
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        nan_detected = False
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data (autoencoder's input is same as target)
            inputs = batch[0].to(device)
            targets = inputs
            
            # Skip batch if it contains NaN values
            if torch.isnan(inputs).any():
                logger.warning(f"NaN values detected in input batch {batch_idx}, skipping")
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss).item():
                logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                nan_detected = True
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimize
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Check if NaN was detected
        if nan_detected:
            logger.warning(f"NaN values detected during epoch {epoch+1}. Consider reducing learning rate.")
        
        # Calculate average loss for the epoch
        if len(train_loader.dataset) > 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
        else:
            logger.warning("Empty training dataset or all batches contained NaN")
            epoch_loss = float('nan')
            train_losses.append(epoch_loss)
        
        # Validation if loader provided
        val_msg = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_nan_detected = False
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    targets = inputs
                    
                    # Skip batch if it contains NaN
                    if torch.isnan(inputs).any():
                        continue
                    
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        val_nan_detected = True
                        continue
                    
                    val_loss += loss.item() * inputs.size(0)
            
            if len(val_loader.dataset) > 0 and not val_nan_detected:
                val_loss = val_loss / len(val_loader.dataset)
                val_msg = f", Validation Loss: {val_loss:.6f}"
            else:
                val_msg = ", Validation Loss: NaN (detected NaNs)"
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}{val_msg}")
        
        # Save intermediate checkpoint
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = Path(str(model_path).replace('.pt', f'_epoch_{epoch+1}.pt'))
            torch.save(model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    torch.save(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    return model, train_losses


def save_latents(
    model: nn.Module, 
    dataloader: TorchDataLoader, 
    device: torch.device, 
    latent_dir: Union[str, Path], 
    save_errors: bool = True, 
    timestamp: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate and save latent representations for data
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader with input data
        device: Device to run model on
        latent_dir: Directory to save latent representations
        save_errors: Whether to save reconstruction errors
        timestamp: Whether to add timestamp to filenames
        
    Returns:
        tuple: (latent_vectors, reconstruction_errors or None)
    """
    model.eval()
    
    # Convert latent_dir to Path and create if doesn't exist
    latent_dir = Path(latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare filename suffix if timestamp requested
    suffix = ""
    if timestamp:
        from datetime import datetime
        suffix = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Collect latent vectors and reconstruction errors
    all_latents = []
    all_errors = [] if save_errors else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating latent vectors"):
            inputs = batch[0].to(device)
            
            # Skip batch if it contains NaN values
            if torch.isnan(inputs).any():
                logger.warning("NaN values detected in input batch, skipping")
                continue
            
            # Get reconstructed outputs and latent vectors
            outputs, latents = model(inputs)
            
            # Store latent vectors
            all_latents.append(latents.cpu().numpy())
            
            # Calculate and store reconstruction errors if requested
            if save_errors:
                errors = torch.mean((outputs - inputs) ** 2, dim=1)
                all_errors.append(errors.cpu().numpy())
    
    # Concatenate batches
    latent_vectors = np.vstack(all_latents) if all_latents else np.array([])
    
    if save_errors and all_errors:
        reconstruction_errors = np.concatenate(all_errors)
    else:
        reconstruction_errors = None
    
    # Save latent vectors
    latent_path = latent_dir / f"latents{suffix}.npy"
    np.save(latent_path, latent_vectors)
    logger.info(f"Saved latent vectors with shape {latent_vectors.shape} to {latent_path}")
    
    # Save reconstruction errors if available
    if reconstruction_errors is not None:
        error_path = latent_dir / f"reconstruction_errors{suffix}.npy"
        np.save(error_path, reconstruction_errors)
        logger.info(f"Saved reconstruction errors with shape {reconstruction_errors.shape} to {error_path}")
    
    return latent_vectors, reconstruction_errors


def save_model_metadata(model: AutoEncoder, metadata_path: Union[str, Path]) -> None:
    """
    Save model metadata to JSON file
    
    Args:
        model: Trained autoencoder model
        metadata_path: Path to save metadata
    """
    metadata = {
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
        "use_sigmoid": model.use_sigmoid,
        "creation_time": str(datetime.datetime.now()),
        "model_type": "autoencoder"
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Saved model metadata to {metadata_path}")


def setup_device() -> torch.device:
    """
    Set up device for model training (CPU or CUDA)
    
    Returns:
        torch.device: Selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    return device


def main():
    """
    Main entry point for training autoencoder
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train autoencoder for anomaly detection")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent space dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--save-interval", type=int, default=10, help="Save model every N epochs (0 to disable)")
    parser.add_argument("--normalize", action="store_true", help="Normalize input features")
    parser.add_argument("--use-sigmoid", action="store_true", help="Use sigmoid activation on output")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to saved files")
    args = parser.parse_args()
    
    # Setup device (CPU or CUDA)
    device = setup_device()
    
    # Load data using DataLoader
    logger.info("Loading data...")
    data_loader = DataLoader()
    X = data_loader.load_data(supervised=False, normalize=args.normalize)
    
    if X is None or len(X) == 0:
        logger.error("Failed to load data or empty dataset")
        return
    
    logger.info(f"Loaded data with shape: {X.shape}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    
    # Split into train/validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = TorchDataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Build model
    input_dim = X.shape[1]
    model = build_autoencoder(
        input_dim=input_dim, 
        latent_dim=args.latent_dim, 
        dropout=args.dropout,
        use_sigmoid=args.use_sigmoid
    )
    logger.info(f"Created autoencoder with input_dim={input_dim}, latent_dim={args.latent_dim}")
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "autoencoder.pt"
    model, losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_interval=args.save_interval,
        model_path=model_path
    )
    
    # Save model metadata
    metadata_path = MODELS_DIR / "autoencoder_model_meta.json"
    save_model_metadata(model, metadata_path)
    
    # Generate and save latent vectors
    logger.info("Generating latent vectors...")
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    full_loader = TorchDataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    latent_vectors, reconstruction_errors = save_latents(
        model=model,
        dataloader=full_loader,
        device=device,
        latent_dir=LATENTS_DIR,
        save_errors=True,
        timestamp=args.timestamp
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    import datetime
    main()