#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Model Training Utilities

This module provides a centralized training loop and utilities that can be
used with different model types, supporting both PyTorch and scikit-learn models.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator
import importlib.util
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import project modules
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("trainer")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting during training.
    
    This class tracks validation metrics and signals when to stop training
    if no improvement is observed for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        restore_best_weights: bool = True
    ):
        """
        Initialize the early stopping handler.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change in monitored value to qualify as improvement
            restore_best_weights: Whether to restore the best weights when stopped
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, model: nn.Module, val_score: float, epoch: int) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            model: The model being trained
            val_score: The current validation score (lower is better)
            epoch: The current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = val_score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.save_checkpoint(model)
        elif val_score > self.best_score - self.min_delta:
            # No improvement
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best epoch was {self.best_epoch+1}")
                self.early_stop = True
        else:
            # Improvement
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.save_checkpoint(model)
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """
        Save model weights.
        
        Args:
            model: The model to save weights from
        """
        # For PyTorch models, save state_dict
        if isinstance(model, nn.Module):
            self.best_weights = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    
    def restore_checkpoint(self, model: nn.Module) -> None:
        """
        Restore best model weights.
        
        Args:
            model: The model to restore weights to
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored model weights from epoch {self.best_epoch+1}")


class GPUMemoryManager:
    """
    Utility class to manage GPU memory during training.
    
    This class provides functions to clear GPU cache periodically,
    monitor GPU memory usage, and optimize memory allocation.
    """
    
    def __init__(self, threshold_usage: float = 0.9, clear_interval: int = 5):
        """
        Initialize the GPU memory manager.
        
        Args:
            threshold_usage: Memory usage threshold (0.0-1.0) to trigger clearing
            clear_interval: Clear cache every N epochs regardless of usage
        """
        self.threshold_usage = threshold_usage
        self.clear_interval = clear_interval
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    
    def clear_cache(self, force: bool = False) -> None:
        """
        Clear GPU cache if available.
        
        Args:
            force: Whether to force clearing regardless of usage
        """
        if not self.cuda_available:
            return
            
        if force or self.get_memory_usage() > self.threshold_usage:
            # Manually trigger garbage collection first
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            logger.debug(f"Cleared GPU memory cache, usage now: {self.get_memory_usage():.2f}")
    
    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage as a fraction.
        
        Returns:
            Memory usage as a fraction (0.0-1.0)
        """
        if not self.cuda_available:
            return 0.0
            
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        
        # Calculate usage as a fraction of total memory
        usage = memory_reserved / memory_total
        
        return usage
    
    def log_memory_stats(self) -> None:
        """Log detailed GPU memory statistics."""
        if not self.cuda_available:
            return
            
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)    # MB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
        
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Total: {total:.2f} MB")
    
    def should_clear_on_epoch(self, epoch: int) -> bool:
        """
        Check if memory should be cleared based on epoch number.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            True if memory should be cleared
        """
        if not self.cuda_available:
            return False
            
        # Clear on specified interval
        return epoch % self.clear_interval == 0


def create_dataloaders(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoader objects from numpy arrays.
    
    Args:
        X: Features array
        y: Labels array (optional)
        batch_size: Size of mini-batches
        val_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create dataset based on whether labels are provided
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)
    
    # Split into train and validation sets
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=DEVICE.type == "cuda"
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=DEVICE.type == "cuda"
        )
        
        return train_loader, val_loader
    else:
        # Create only training loader
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=DEVICE.type == "cuda"
        )
        
        return train_loader, None


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 15,
    max_grad_norm: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    save_interval: int = 0,
    callbacks: Optional[List[Callable]] = None,
    return_best: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Generic training loop for PyTorch models.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        criterion: Loss function (if None, uses MSELoss)
        optimizer: Optimizer (if None, uses Adam)
        epochs: Number of training epochs
        lr: Learning rate (if optimizer not provided)
        weight_decay: L2 regularization factor (if optimizer not provided)
        early_stopping_patience: Patience for early stopping
        max_grad_norm: Maximum gradient norm for clipping
        save_path: Path to save the model
        save_interval: Save checkpoint every N epochs (0 to disable)
        callbacks: List of callback functions called at the end of each epoch
        return_best: Whether to return the best model based on validation loss
        
    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Move model to device
    model.to(DEVICE)
    
    # Initialize model weights properly
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Set up loss function and optimizer if not provided
    if criterion is None:
        criterion = nn.MSELoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Initialize GPU memory manager
    memory_manager = GPUMemoryManager()
    
    # Create history dictionary to track metrics
    history = {
        "train_loss": [],
        "val_loss": [] if val_loader else None,
        "epoch_times": []
    }
    
    # Create directory for model if needed
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Clear GPU memory if needed
        if memory_manager.should_clear_on_epoch(epoch):
            memory_manager.clear_cache(force=True)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats (with or without labels)
            if len(batch) > 1:
                inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
            else:
                inputs = batch[0].to(DEVICE)
                targets = inputs  # Autoencoder-style (input is also target)
            
            # Skip batch if it contains NaN values
            if torch.isnan(inputs).any():
                logger.warning(f"NaN values detected in input batch {batch_idx}, skipping")
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, nn.Module):
                # Handle models with multiple returns (e.g., encoders that return latent)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first output if multiple returned
            else:
                # Fallback for non-standard models
                outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss).item():
                logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimize
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else float('nan')
        history["train_loss"].append(epoch_loss)
        
        # Validation phase
        val_loss = None
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Handle different batch formats
                    if len(batch) > 1:
                        inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    else:
                        inputs = batch[0].to(DEVICE)
                        targets = inputs
                    
                    # Skip batch if it contains NaN values
                    if torch.isnan(inputs).any():
                        continue
                    
                    # Forward pass
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                        
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
            
            # Calculate average validation loss
            val_loss = val_running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')
            history["val_loss"].append(val_loss)
            
            # Check early stopping
            if early_stopping(model, val_loss, epoch):
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        
        # Log progress
        if val_loss:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
        
        # Save intermediate checkpoint
        if save_path and save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = save_path.with_name(f"{save_path.stem}_epoch_{epoch+1}{save_path.suffix}")
            torch.save(model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Run callbacks if provided
        if callbacks:
            for callback in callbacks:
                callback(model, epoch, history)
    
    # Restore best model if using early stopping and validation set provided
    if val_loader and return_best and early_stopping.best_weights is not None:
        early_stopping.restore_checkpoint(model)
    
    # Final memory cleanup
    memory_manager.clear_cache(force=True)
    
    # Save final model
    if save_path:
        torch.save(model, save_path)
        logger.info(f"Saved final model to {save_path}")
    
    return model, history


def train_sklearn_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    callbacks: Optional[List[Callable]] = None
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a scikit-learn model.
    
    Args:
        model: Scikit-learn model to train
        X_train: Training features
        y_train: Training labels (for supervised learning)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        save_path: Path to save the model
        callbacks: List of callback functions called after training
        
    Returns:
        Tuple of (trained_model, metadata)
    """
    import joblib
    
    start_time = time.time()
    logger.info(f"Training {model.__class__.__name__} model")
    
    # Store metadata
    metadata = {
        "model_type": model.__class__.__name__,
        "training_samples": X_train.shape[0],
        "feature_dimension": X_train.shape[1]
    }
    
    # Train the model
    if y_train is not None:
        # Supervised learning
        model.fit(X_train, y_train)
        metadata["supervised"] = True
        
        # Calculate training score
        train_score = model.score(X_train, y_train)
        metadata["train_score"] = train_score
        logger.info(f"Training score: {train_score:.4f}")
        
        # Calculate validation score if provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            metadata["val_score"] = val_score
            logger.info(f"Validation score: {val_score:.4f}")
    else:
        # Unsupervised learning
        model.fit(X_train)
        metadata["supervised"] = False
        
        # Some models like IsolationForest have a score function for unsupervised
        if hasattr(model, "score_samples"):
            train_score = model.score_samples(X_train).mean()
            metadata["train_score"] = train_score
            logger.info(f"Average training score: {train_score:.4f}")
            
            if X_val is not None:
                val_score = model.score_samples(X_val).mean()
                metadata["val_score"] = val_score
                logger.info(f"Average validation score: {val_score:.4f}")
    
    # Record training time
    training_time = time.time() - start_time
    metadata["training_time"] = training_time
    logger.info(f"Model trained in {training_time:.2f} seconds")
    
    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        logger.info(f"Saved model to {save_path}")
    
    # Run callbacks if provided
    if callbacks:
        for callback in callbacks:
            callback(model, 0, metadata)
    
    return model, metadata


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary of training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot epoch times
    if "epoch_times" in history:
        plt.subplot(1, 2, 2)
        plt.plot(history["epoch_times"])
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title("Epoch Training Time")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.close()


class ModelTrainer:
    """
    Unified model trainer class that can handle different model types.
    
    This class provides a consistent interface for training both PyTorch
    and scikit-learn models with various options.
    """
    
    def __init__(
        self,
        model: Union[nn.Module, BaseEstimator],
        save_dir: Union[str, Path] = "models",
        model_name: Optional[str] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            model: The model to train (PyTorch or scikit-learn)
            save_dir: Directory to save models and results
            model_name: Name for the model (used in filenames)
        """
        self.model = model
        self.save_dir = Path(save_dir)
        
        # Determine model type
        self.is_pytorch = isinstance(model, nn.Module)
        
        # Set model name if not provided
        if model_name is None:
            self.model_name = model.__class__.__name__.lower()
        else:
            self.model_name = model_name
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Full save path for model
        ext = ".pt" if self.is_pytorch else ".joblib"
        self.save_path = self.save_dir / f"{self.model_name}{ext}"
        
        # Initialize logger
        self.logger = get_logger(f"trainer.{self.model_name}")
        
        # Get configuration
        self.config = get_config()
    
    def train(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        val_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        early_stopping_patience: int = 15,
        save_checkpoints: bool = False,
        plot_history: bool = True,
        **kwargs
    ) -> Union[nn.Module, BaseEstimator]:
        """
        Train the model on provided data.
        
        This method automatically chooses the appropriate training function
        based on the model type.
        
        Args:
            X: Features array
            y: Labels array (optional)
            val_split: Fraction of data to use for validation
            batch_size: Batch size for PyTorch models
            epochs: Number of epochs for PyTorch models
            lr: Learning rate for PyTorch models
            early_stopping_patience: Patience for early stopping
            save_checkpoints: Whether to save intermediate checkpoints
            plot_history: Whether to plot training history
            **kwargs: Additional arguments passed to the training function
            
        Returns:
            Trained model
        """
        self.logger.info(f"Training {self.model_name} model on {X.shape[0]} samples")
        
        # Check for NaN values
        if np.isnan(X).any():
            self.logger.warning("Input data contains NaN values. These may cause issues during training.")
        
        if y is not None and np.isnan(y).any():
            self.logger.warning("Target data contains NaN values. These may cause issues during training.")
        
        # Create timestamp for saving
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_save_path = self.save_dir / f"{self.model_name}_history_{timestamp}.png"
        
        # Train based on model type
        if self.is_pytorch:
            # Create data loaders
            train_loader, val_loader = create_dataloaders(
                X, y, batch_size=batch_size, val_split=val_split,
                shuffle=True, num_workers=kwargs.get('num_workers', 4)
            )
            
            # Determine whether to save checkpoints
            save_interval = 5 if save_checkpoints else 0
            
            # Train model
            self.model, history = train_pytorch_model(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                early_stopping_patience=early_stopping_patience,
                save_path=self.save_path,
                save_interval=save_interval,
                **kwargs
            )
            
            # Plot training history
            if plot_history:
                plot_training_history(history, plot_save_path)
        else:
            # Split data for validation if requested
            if val_split > 0 and X.shape[0] > 10:  # Ensure enough samples
                split_idx = int(X.shape[0] * (1 - val_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                
                if y is not None:
                    y_train, y_val = y[:split_idx], y[split_idx:]
                else:
                    y_train, y_val = None, None
            else:
                X_train, X_val = X, None
                y_train, y_val = y, None
            
            # Train model
            self.model, metadata = train_sklearn_model(
                model=self.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                save_path=self.save_path,
                **kwargs
            )
            
            # Save metadata
            metadata_path = self.save_dir / f"{self.model_name}_metadata_{timestamp}.json"
            try:
                import json
                with open(metadata_path, 'w') as f:
                    # Convert numpy values to Python types
                    metadata_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                    for k, v in metadata.items()}
                    json.dump(metadata_json, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Could not save metadata: {e}")
        
        self.logger.info(f"Training complete, model saved to {self.save_path}")
        return self.model


# Factory function to create a trainer for a specific model
def get_trainer(
    model: Union[nn.Module, BaseEstimator],
    save_dir: Union[str, Path] = "models",
    model_name: Optional[str] = None
) -> ModelTrainer:
    """
    Create a trainer for the specified model.
    
    Args:
        model: The model to train
        save_dir: Directory to save models and results
        model_name: Name for the model (used in filenames)
        
    Returns:
        Configured ModelTrainer instance
    """
    return ModelTrainer(model, save_dir, model_name)
