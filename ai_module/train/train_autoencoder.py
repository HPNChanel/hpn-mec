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

import os
import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the AutoEncoder class from the models module
try:
    from ..models.autoencoder import AutoEncoder
    logger.info("Imported AutoEncoder class from models module")
except ImportError:
    # Define the class locally if import fails
    logger.info("Defining AutoEncoder class locally")

    class AutoEncoder(nn.Module):
        """
        Autoencoder neural network for unsupervised anomaly detection
        """
        
        def __init__(self, input_dim, latent_dim=16, dropout=0.2, use_sigmoid=False):
            """
            Initialize autoencoder with configurable dimensions and dropout
            
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
            Forward pass through the autoencoder
            
            Args:
                x: Input tensor
                
            Returns:
                tuple: (reconstructed_output, latent_vector)
            """
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent

def build_autoencoder(input_dim, latent_dim=16, dropout=0.2, use_sigmoid=False):
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

def train_model(model, train_loader, val_loader=None, epochs=50, lr=0.001, 
                device='cpu', save_interval=10, model_path='model.pt'):
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
        
    Returns:
        tuple: (trained_model, losses)
    """
    # Move model to device
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data (autoencoder's input is same as target)
            inputs = batch[0].to(device)
            targets = inputs
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation if loader provided
        val_msg = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    targets = inputs
                    
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_msg = f", Validation Loss: {val_loss:.6f}"
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.6f}{val_msg}")
        
        # Save checkpoint if requested
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = f"{str(model_path).split('.')[0]}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    return model, train_losses

def save_latents(model, dataloader, device, latent_dir, save_errors=True, timestamp=False):
    """
    Extract and save latent representations and reconstruction errors
    
    Args:
        model: Trained autoencoder model
        dataloader: DataLoader with input data
        device: Device to run inference on
        latent_dir: Directory to save latent vectors
        save_errors: Whether to save reconstruction errors
        timestamp: Whether to add timestamp to filenames
        
    Returns:
        tuple: (latent_vectors, reconstruction_errors)
    """
    model.eval()
    all_latents = []
    all_errors = []
    
    # Generate timestamp suffix if requested
    suffix = ""
    if timestamp:
        from datetime import datetime
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directory if it doesn't exist
    latent_dir = Path(latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract latent vectors and errors
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent vectors"):
            inputs = batch[0].to(device)
            
            # Get reconstructions and latent vectors
            reconstructions, latents = model(inputs)
            
            # Calculate reconstruction error for each sample
            errors = torch.sum((reconstructions - inputs) ** 2, dim=1)
            
            # Store results
            all_latents.append(latents.cpu().numpy())
            all_errors.append(errors.cpu().numpy())
    
    # Concatenate results
    latent_vectors = np.concatenate(all_latents)
    errors = np.concatenate(all_errors)
    
    # Save latent vectors
    latents_path = latent_dir / f"latents{suffix}.npy"
    np.save(latents_path, latent_vectors)
    logger.info(f"Saved latent vectors with shape {latent_vectors.shape} to {latents_path}")
    
    # Save errors if requested
    if save_errors:
        errors_path = latent_dir / f"errors{suffix}.npy"
        np.save(errors_path, errors)
        logger.info(f"Saved reconstruction errors with shape {errors.shape} to {errors_path}")
    
    return latent_vectors, errors

def setup_device():
    """
    Set up and return the appropriate device (CUDA or CPU)
    
    Returns:
        torch.device: Device to use for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Print CUDA memory stats if available
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        logger.info("CUDA not available, using CPU")
    
    return device

def main():
    """
    Main function to load data, build model, train, and save results
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train autoencoder for health data anomaly detection")
    
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Dimension of latent space (default: 16)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--save-latents", action="store_true",
                        help="Save latent vectors and errors")
    parser.add_argument("--use-sigmoid", action="store_true",
                        help="Use sigmoid activation on output")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add timestamp to saved files")
    parser.add_argument("--save-interval", type=int, default=0,
                        help="Save model checkpoint every N epochs (default: 0, disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to input data file (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = setup_device()
    
    # Use Path for file path handling
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define paths
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Try several possible data paths
        possible_data_paths = [
            script_dir.parent / "data" / "processed" / "features" / "X_train.npy",
            project_root / "data" / "processed" / "features" / "X_train.npy",
            project_root / "ai_module" / "data" / "processed" / "features" / "X_train.npy"
        ]
        
        data_path = None
        for path in possible_data_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            logger.error("Could not find input data file. Please specify with --data-path")
            sys.exit(1)
    
    # Define output directories with pathlib
    latent_dir = script_dir.parent / "data" / "processed" / "latents"
    model_dir = script_dir.parent / "models"
    
    # Create directories if they don't exist
    for directory in [latent_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    logger.info(f"Loading data from {data_path}")
    
    try:
        X = np.load(data_path)
        logger.info(f"Loaded data with shape {X.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Convert to PyTorch tensor and create dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    
    # Split into train and validation sets
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    # For encoding all data
    full_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    # Build model
    input_dim = X.shape[1]
    model = build_autoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        use_sigmoid=args.use_sigmoid
    )
    
    logger.info(f"Created autoencoder model with:")
    logger.info(f"  Input dimension: {input_dim}")
    logger.info(f"  Latent dimension: {args.latent_dim}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model_path = model_dir / "autoencoder.pt"
    logger.info(f"Training model for {args.epochs} epochs (batch size: {args.batch_size})")
    
    model, losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        save_interval=args.save_interval,
        model_path=model_path
    )
    
    # Save latent vectors and errors
    if args.save_latents:
        logger.info("Extracting and saving latent vectors")
        save_latents(
            model=model,
            dataloader=full_loader,
            device=device,
            latent_dir=latent_dir,
            save_errors=True,
            timestamp=args.timestamp
        )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()