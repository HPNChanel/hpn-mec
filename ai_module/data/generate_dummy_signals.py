#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Synthetic ECG-like Signals

This script generates synthetic ECG-like signals for testing the AI training pipeline.
It creates 20 .npy files containing numpy arrays of shape (1000, 12) that mimic 
multichannel ECG signals with various patterns and noise levels.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
PROCESSED_SIGNALS_DIR = PROJECT_ROOT / "data/processed/signals"

# Ensure output directory exists
PROCESSED_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

def generate_base_signal(n_samples=1000, n_channels=12):
    """
    Generate base synthetic ECG-like signal using sine waves.
    
    Args:
        n_samples (int): Number of time samples
        n_channels (int): Number of channels (leads)
        
    Returns:
        numpy.ndarray: Synthetic signal of shape (n_samples, n_channels)
    """
    # Create time vector (normalized from 0 to 2π)
    t = np.linspace(0, 2*np.pi, n_samples)
    
    # Initialize signal array
    signal = np.zeros((n_samples, n_channels))
    
    # Generate different patterns for each channel
    for i in range(n_channels):
        # Base frequency varies slightly per channel
        base_freq = 1.0 + i * 0.1
        
        # Create ECG-like pattern with multiple sine waves
        # Main wave (like QRS complex)
        signal[:, i] = 1.0 * np.sin(base_freq * t)
        
        # Add smaller waves (like P and T waves)
        signal[:, i] += 0.3 * np.sin(base_freq * 2 * t + np.pi/4)
        signal[:, i] += 0.2 * np.sin(base_freq * 3 * t - np.pi/3)
        
        # Add sharp peak (similar to R peak)
        peak_position = int(n_samples * 0.4) + i * 20
        peak_width = int(n_samples * 0.05)
        peak_start = max(0, peak_position - peak_width)
        peak_end = min(n_samples, peak_position + peak_width)
        signal[peak_start:peak_end, i] += 2.0 * np.sin(np.linspace(0, np.pi, peak_end - peak_start))
    
    return signal

def add_gaussian_noise(signal, noise_level=0.1):
    """
    Add Gaussian noise to signal.
    
    Args:
        signal (numpy.ndarray): Input signal
        noise_level (float): Standard deviation of the noise
        
    Returns:
        numpy.ndarray: Signal with added noise
    """
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def add_outlier_noise(signal, outlier_intensity=5.0, outlier_percentage=0.02):
    """
    Add outlier noise to signal (for anomaly detection testing).
    
    Args:
        signal (numpy.ndarray): Input signal
        outlier_intensity (float): Intensity multiplier for outliers
        outlier_percentage (float): Percentage of samples to make outliers
        
    Returns:
        numpy.ndarray: Signal with added outliers
    """
    # Create a copy of the signal
    noisy_signal = np.copy(signal)
    
    # Get dimensions
    n_samples, n_channels = signal.shape
    
    # Calculate number of outlier points
    n_outliers = int(n_samples * n_channels * outlier_percentage)
    
    # Randomly select points to make outliers
    for _ in range(n_outliers):
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, n_channels)
        
        # Add large spike (positive or negative)
        sign = 1 if np.random.random() > 0.5 else -1
        noisy_signal[i, j] += sign * outlier_intensity * np.random.random()
    
    # Add some channel-wide outliers (simulating electrode issues)
    if np.random.random() > 0.7:  # 30% chance of channel-wide issue
        problem_channel = np.random.randint(0, n_channels)
        problem_start = np.random.randint(0, n_samples//2)
        problem_duration = np.random.randint(n_samples//10, n_samples//5)
        problem_end = min(n_samples, problem_start + problem_duration)
        
        # Apply channel-wide distortion
        distortion_type = np.random.choice(['flatline', 'noise', 'shift'])
        
        if distortion_type == 'flatline':
            # Simulate flatline
            noisy_signal[problem_start:problem_end, problem_channel] = 0
        elif distortion_type == 'noise':
            # Extreme noise
            noisy_signal[problem_start:problem_end, problem_channel] += np.random.normal(
                0, outlier_intensity, problem_end - problem_start)
        else:  # shift
            # Baseline shift
            noisy_signal[problem_start:problem_end, problem_channel] += outlier_intensity/2
    
    return noisy_signal

def generate_and_save_signal(file_index, add_outliers=False):
    """
    Generate synthetic ECG-like signal and save to .npy file.
    
    Args:
        file_index (int): File index (1-20)
        add_outliers (bool): Whether to add outlier noise
    
    Returns:
        bool: Success status
    """
    try:
        # Generate base signal
        signal = generate_base_signal(n_samples=1000, n_channels=12)
        
        # Add normal Gaussian noise
        signal = add_gaussian_noise(signal, noise_level=0.1)
        
        # Add outliers if specified
        if add_outliers:
            signal = add_outlier_noise(signal, outlier_intensity=5.0, outlier_percentage=0.02)
            logger.info(f"Generated signal {file_index:05d} with outliers")
        else:
            logger.info(f"Generated normal signal {file_index:05d}")
        
        # Save to .npy file
        filename = f"{file_index:05d}_fake.npy"
        output_path = PROCESSED_SIGNALS_DIR / filename
        np.save(output_path, signal)
        
        logger.info(f"Saved signal to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating signal {file_index}: {e}")
        return False

def main():
    """
    Main function to generate all synthetic ECG signals.
    """
    logger.info("Starting synthetic ECG signal generation")
    
    success_count = 0
    
    # Generate 20 files
    for i in range(1, 21):
        # Add outliers to every 5th file (5, 10, 15, 20)
        add_outliers = (i % 5 == 0)
        
        if generate_and_save_signal(i, add_outliers):
            success_count += 1
    
    logger.info(f"Successfully generated {success_count} out of 20 synthetic signals")
    logger.info("Synthetic ECG signal generation completed")

if __name__ == "__main__":
    main()