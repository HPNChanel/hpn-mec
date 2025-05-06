#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Model Trainer for AI Models

This module provides functions for training various AI models in a consistent way,
with support for progress tracking, checkpointing, and metrics logging.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, dataloader, config, device=None):
    """
    Generic model training function that can be used for various model types.
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloader (DataLoader): DataLoader for training data
        config (dict): Configuration parameters including:
                      - num_epochs: Number of training epochs
                      - lr: Learning rate
                      - model_path: Path to save model
                      - save_interval: Save checkpoint every N epochs (0 to disable)
                      - optimizer: Optimizer type ('adam', 'sgd', etc.)
                      - loss_function: Loss function ('mse', 'bce', 'ce', etc.)
        device (torch.device): Device to use for training (default: None, auto-detect)
        
    Returns:
        tuple: (model, losses) - Trained model and list of losses per epoch
    """
    # Extract config values with defaults
    num_epochs = config.get('num_epochs', 50)
    lr = config.get('lr', 0.001)
    model_path = config.get('model_path', 'model.pt')
    save_interval = config.get('save_interval', 0)
    optimizer_type = config.get('optimizer', 'adam')
    loss_type = config.get('loss_function', 'mse')
    
    # Resolve device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Auto-selected device: {device}")
    
    # Ensure model is on the correct device
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    # Set up loss function - ensure criterion is on the same device as model
    if loss_type.lower() == 'mse':
        criterion = nn.MSELoss().to(device)
    elif loss_type.lower() == 'bce':
        criterion = nn.BCELoss().to(device)
    elif loss_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        logger.warning(f"Unknown loss type: {loss_type}, defaulting to MSE")
        criterion = nn.MSELoss().to(device)
    
    # Set up optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}, defaulting to Adam")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store training metrics
    losses = []
    epoch_times = []
    start_time = time.time()
    
    # Create DataFrame to store losses
    loss_df = pd.DataFrame(columns=['epoch', 'loss', 'time_seconds'])
    
    # Ensure reports directory exists for saving loss CSV
    if 'reports_dir' in config:
        reports_dir = Path(config['reports_dir'])
        reports_dir.mkdir(parents=True, exist_ok=True)
        losses_path = reports_dir / f"losses_{Path(model_path).stem}.csv"
    else:
        # Default to parent directory of model_path
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        losses_path = model_dir / f"losses_{Path(model_path).stem}.csv"
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        # Wrap dataloader with tqdm for progress tracking
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
            ncols=100
        )
        
        # Process batches
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            inputs = None
            targets = None
            
            # Case 1: batch is a tensor (autoencoder case)
            if isinstance(batch, torch.Tensor):
                inputs = batch.to(device, non_blocking=True)
                targets = inputs  # For autoencoder, input = target
            
            # Case 2: batch is a tuple/list (input, target)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)
            
            # Error case
            if inputs is None:
                logger.error(f"Could not determine inputs from batch: {type(batch)}")
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - handle different model return types
            outputs = model(inputs)
            
            # Extract outputs for loss calculation
            if isinstance(outputs, tuple):
                # Handle models that return multiple outputs (like autoencoder returning reconstruction and latent)
                loss_inputs = outputs[0]
            else:
                loss_inputs = outputs
            
            # Calculate loss
            loss = criterion(loss_inputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{running_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Calculate epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch_time * (num_epochs - epoch - 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        
        # Log progress
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s, ETA: {eta_str}")
        
        # Append to DataFrame
        loss_df = pd.concat([
            loss_df,
            pd.DataFrame({
                'epoch': [epoch + 1],
                'loss': [avg_loss],
                'time_seconds': [epoch_time]
            })
        ], ignore_index=True)
        
        # Save loss data every epoch
        loss_df.to_csv(losses_path, index=False)
        
        # Save model checkpoint if requested
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = Path(model_path).parent / f"{Path(model_path).stem}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'device': str(device),  # Save device information for loading
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    
    # Save the final trained model
    logger.info(f"Saving trained model to {model_path}")
    torch.save(model.state_dict(), model_path)
    
    return model, losses
