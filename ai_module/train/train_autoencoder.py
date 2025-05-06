#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder Training for ECG Signal Anomaly Detection

This script trains an autoencoder neural network on multi-channel ECG signal data.
The trained model can be used for anomaly detection by comparing reconstruction error.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

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

# Get project root (parent of ai_module)
PROJECT_ROOT = module_dir.parent

# Constants and paths relative to the ai_module directory
MODELS_DIR = module_dir / "models"
DATA_DIR = module_dir / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SIGNALS_DIR = PROCESSED_DIR / "signals"
REPORTS_DIR = module_dir / "reports"

# Ensure directories exist
for directory in [MODELS_DIR, PROCESSED_DIR, SIGNALS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default model path
DEFAULT_MODEL_PATH = MODELS_DIR / "autoencoder.pt"

# Add project root to path for reliable imports
sys.path.insert(0, str(PROJECT_ROOT))

# Now use absolute imports
from ai_module.models.autoencoder import Autoencoder
from ai_module.utils.dataset import ECGDataset
from ai_module.utils.trainer import train_model
from ai_module.utils.train_utils import setup_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device for training - use the improved setup_device function
device = setup_device()
logger.info(f"Using device: {device}")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
logger.info(f"Set random seed to {SEED} for reproducibility")

# Add config loading function
def load_config():
    try:
        # Luôn tìm thư mục 'ai_module' làm gốc
        current_file = Path(__file__).resolve()
        ai_module_dir = current_file
        while ai_module_dir.name != "ai_module":
            ai_module_dir = ai_module_dir.parent

        config_path = ai_module_dir / "config" / "config.yaml"
        if not config_path.exists():
            logging.error(f"Configuration file not found: {config_path}")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logging.info(f"Loaded config from: {config_path}")
        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return {}

# Load configuration from YAML file
config = load_config()

if not config:
    logger.error("Failed to load configuration, using default values")
    # Default values
    input_dim = 12000
    latent_dim = 64
    use_sigmoid = True
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
else:
    # Load values from config
    input_dim = config.get('input_dim', 12000)
    latent_dim = config.get('latent_dim', 64)
    use_sigmoid = config.get('use_sigmoid', True)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)

# Set paths with improved defaults and cross-platform compatibility
signal_dir = Path(config.get("signal_dir", "ai_module/data/processed/signals/"))
model_path = Path(config.get("model_path", "ai_module/models/autoencoder.pt"))

# Update paths based on config
SIGNALS_DIR = PROJECT_ROOT / signal_dir
MODELS_DIR = PROJECT_ROOT / model_path.parent
model_path = str(PROJECT_ROOT / model_path)

# Create a training config dictionary
training_config = {
    'num_epochs': num_epochs,
    'lr': learning_rate,
    'model_path': model_path,
    'save_interval': 10,  # Save checkpoint every 10 epochs
    'optimizer': 'adam',
    'loss_function': 'mse',
    'reports_dir': str(REPORTS_DIR)
}

def main():
    """
    Main function to load data, create model, and train.
    """
    try:
        # Create dataset and dataloader
        logger.info("Creating ECG dataset")
        dataset = ECGDataset(SIGNALS_DIR)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),  # Enable pin_memory for faster data transfer to GPU
            persistent_workers=torch.cuda.is_available() and batch_size > 1  # Keep workers alive between batches
        )
        
        # Create autoencoder model
        logger.info("Initializing autoencoder model")
        model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, use_sigmoid=use_sigmoid)
        
        # Print model summary
        logger.info(f"Autoencoder architecture:\n"
                   f"Input dimension: {model.input_dim}\n"
                   f"Latent dimension: {model.latent_dim}\n"
                   f"Using sigmoid: {model.use_sigmoid}")
        
        # Move model to device
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        
        # Log GPU memory usage before training (if using CUDA)
        if device.type == 'cuda':
            logger.info(f"GPU memory allocated before training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"GPU memory reserved before training: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Train the model using the generic trainer
        logger.info("Training autoencoder")
        trained_model, losses = train_model(model, dataloader, training_config, device)
        
        # Log GPU memory usage after training (if using CUDA)
        if device.type == 'cuda':
            logger.info(f"GPU memory allocated after training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"GPU memory reserved after training: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            # Empty CUDA cache to free up memory
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        logger.info(f"Training completed. Final loss: {losses[-1]:.6f}")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Loss history saved to {REPORTS_DIR / 'losses_autoencoder.csv'}")
        
    except Exception as e:
        logger.error(f"Error in autoencoder training: {e}")
        # If there was an error, try to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after error")
        raise
    

if __name__ == "__main__":
    main()