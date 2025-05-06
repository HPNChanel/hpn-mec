# TODO: Implement this module

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for PTB-XL ECG Data Preprocessing Module

This module contains unit tests for the functions in preprocess_ptbxl_data.py,
testing functionality for loading, processing, and saving ECG data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import unittest
from unittest.mock import patch
import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
import tempfile
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Add the parent directory to path to import the module being tested
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ai_module.preprocess.preprocess_ptbxl_data import (
    load_header, load_signal, normalize_signal, 
    save_output, process_ecg_file, PROCESSED_SIGNALS_DIR, 
    PROCESSED_METADATA_DIR
)


class TestECGPreprocessing(unittest.TestCase):
    """
    Test suite for the ECG preprocessing module functions.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        logger.info("Setting up test environment...")
        
        # Define paths
        cls.test_dir = Path(__file__).resolve().parent
        cls.project_root = cls.test_dir.parent
        cls.raw_data_dir = cls.project_root / "data/raw"
        
        # Create temporary output directories for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_signals_dir = Path(cls.temp_dir) / "signals"
        cls.temp_metadata_dir = Path(cls.temp_dir) / "metadata"
        cls.temp_signals_dir.mkdir(parents=True, exist_ok=True)
        cls.temp_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Find an existing sample file for testing
        sample_files = list(cls.raw_data_dir.glob("*.hea"))
        if sample_files:
            cls.sample_file = sample_files[0]
            logger.info(f"Using sample file: {cls.sample_file}")
        else:
            # If no real sample file exists, test will use dummy data
            cls.sample_file = None
            logger.warning("No sample files found in data/raw/. Tests will use dummy data.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        logger.info("Cleaning up test environment...")
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up before each test."""
        # Skip certain tests if no sample file is available
        if self.sample_file is None and not hasattr(self, 'skip_without_sample'):
            self.skipTest("No sample file available")
    
    def test_load_header_valid(self):
        """Test loading header from a valid file."""
        metadata = load_header(self.sample_file)
        
        # Verify metadata is returned and has expected format
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata, dict)
        
        # Check essential metadata fields
        essential_fields = ['record_name', 'n_sig', 'fs', 'sig_len']
        for field in essential_fields:
            self.assertIn(field, metadata)
            self.assertIsNotNone(metadata[field])
        
        # Check lead names
        self.assertIn('lead_names', metadata)
        if metadata['lead_names']:
            self.assertIsInstance(metadata['lead_names'], list)
        
        logger.info(f"Metadata successfully loaded with fields: {', '.join(metadata.keys())}")
    
    def test_load_header_nonexistent(self):
        """Test loading header from a non-existent file."""
        self.skip_without_sample = True
        metadata = load_header(self.raw_data_dir / "nonexistent_file.hea")
        self.assertIsNone(metadata)
        logger.info("Correctly handled non-existent header file")
    
    def test_load_signal_valid(self):
        """Test loading signal from a valid file."""
        signal = load_signal(self.sample_file)
        
        # Verify signal is returned and has expected format
        self.assertIsNotNone(signal)
        self.assertIsInstance(signal, np.ndarray)
        
        # Check signal shape (should be 2D: samples x channels)
        self.assertEqual(len(signal.shape), 2)
        
        # Load header to verify the number of channels
        metadata = load_header(self.sample_file)
        self.assertEqual(signal.shape[1], metadata['n_sig'])
        
        logger.info(f"Signal successfully loaded with shape: {signal.shape}")
    
    def test_load_signal_nonexistent(self):
        """Test loading signal from a non-existent file."""
        self.skip_without_sample = True
        signal = load_signal(self.raw_data_dir / "nonexistent_file")
        self.assertIsNone(signal)
        logger.info("Correctly handled non-existent signal file")
    
    def test_normalize_signal_zscore(self):
        """Test z-score normalization of signal."""
        # Create a test signal
        test_signal = np.random.randn(1000, 12) * 10 + 5  # Non-zero mean, non-unit std
        
        # Apply z-score normalization
        normalized = normalize_signal(test_signal, method='z-score')
        
        # Check shape preservation
        self.assertEqual(normalized.shape, test_signal.shape)
        
        # For each channel, mean should be close to 0 and std close to 1
        for i in range(normalized.shape[1]):
            mean = np.mean(normalized[:, i])
            std = np.std(normalized[:, i])
            
            # Allow some numerical tolerance
            self.assertAlmostEqual(mean, 0, places=1)
            self.assertAlmostEqual(std, 1, places=1)
        
        logger.info("Z-score normalization correctly transforms signal")
    
    def test_normalize_signal_minmax(self):
        """Test min-max normalization of signal."""
        # Create a test signal with known range
        test_signal = np.random.randn(1000, 12) * 10 + 5
        
        # Apply min-max normalization
        normalized = normalize_signal(test_signal, method='min-max')
        
        # Check shape preservation
        self.assertEqual(normalized.shape, test_signal.shape)
        
        # For each channel, values should be in [0, 1] range
        for i in range(normalized.shape[1]):
            min_val = np.min(normalized[:, i])
            max_val = np.max(normalized[:, i])
            
            # Allow some numerical tolerance
            self.assertGreaterEqual(min_val, 0 - 1e-10)
            self.assertLessEqual(max_val, 1 + 1e-10)
        
        logger.info("Min-max normalization correctly transforms signal to [0, 1]")
    
    def test_normalize_signal_constant_channel(self):
        """Test normalization with a constant channel (zero std dev)."""
        # Create a test signal with one constant channel
        test_signal = np.random.randn(1000, 3)
        test_signal[:, 1] = 42  # Middle channel is constant
        
        # Apply normalization
        normalized = normalize_signal(test_signal, method='z-score')
        
        # Constant channel should become all zeros
        self.assertTrue(np.allclose(normalized[:, 1], 0))
        
        # Other channels should be normalized as usual
        for i in [0, 2]:
            mean = np.mean(normalized[:, i])
            std = np.std(normalized[:, i])
            self.assertAlmostEqual(mean, 0, places=1)
            self.assertAlmostEqual(std, 1, places=1)
        
        logger.info("Correctly handled constant channel during normalization")
    
    def test_normalize_signal_invalid_method(self):
        """Test normalization with invalid method."""
        test_signal = np.random.randn(1000, 12)
        
        # Apply normalization with invalid method
        normalized = normalize_signal(test_signal, method='invalid_method')
        
        # Should return original signal
        self.assertTrue(np.array_equal(normalized, test_signal))
        logger.info("Correctly handled invalid normalization method")
    
    def test_save_output(self):
        """Test saving processed signal and metadata."""
        # Create test data
        test_signal = np.random.randn(1000, 12)
        test_metadata = {
            'record_name': 'test_record',
            'n_sig': 12,
            'fs': 100,
            'sig_len': 1000,
            'lead_names': ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        }
        
        # Save output - explicitly pass temp dirs
        result = save_output(test_signal, test_metadata, 'test_output', 
                            signal_dir=self.temp_signals_dir, 
                            metadata_dir=self.temp_metadata_dir)
        
        # Check result
        self.assertTrue(result)
        
        # Check files exist
        signal_path = self.temp_signals_dir / 'test_output.npy'
        metadata_path = self.temp_metadata_dir / 'test_output.json'
        
        self.assertTrue(signal_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Load saved files and check content
        loaded_signal = np.load(signal_path)
        self.assertTrue(np.array_equal(loaded_signal, test_signal))
        
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        self.assertEqual(loaded_metadata, test_metadata)
        
        logger.info("Successfully saved and verified output files")
    
    def test_save_output_none_input(self):
        """Test saving with None inputs."""
        result = save_output(None, {'test': 'metadata'}, 'test_none',
                            signal_dir=self.temp_signals_dir,
                            metadata_dir=self.temp_metadata_dir)
        self.assertFalse(result)
        
        result = save_output(np.zeros((10, 2)), None, 'test_none',
                            signal_dir=self.temp_signals_dir,
                            metadata_dir=self.temp_metadata_dir)
        self.assertFalse(result)
        
        logger.info("Correctly handled None inputs for save_output")
    
    def test_process_ecg_file(self):
        """Test the complete processing of an ECG file."""
        # Process the sample file - explicitly pass temp dirs
        result = process_ecg_file(self.sample_file, 
                                signal_dir=self.temp_signals_dir, 
                                metadata_dir=self.temp_metadata_dir)
        
        # Check result
        self.assertTrue(result)
        
        # Check output files exist
        base_filename = self.sample_file.stem
        signal_path = self.temp_signals_dir / f"{base_filename}.npy"
        metadata_path = self.temp_metadata_dir / f"{base_filename}.json"
        
        self.assertTrue(signal_path.exists())
        self.assertTrue(metadata_path.exists())
        
        logger.info("Successfully processed ECG file")
    
    def test_process_ecg_file_nonexistent(self):
        """Test processing a non-existent ECG file."""
        self.skip_without_sample = True
        result = process_ecg_file(self.raw_data_dir / "nonexistent_file.hea",
                                signal_dir=self.temp_signals_dir,
                                metadata_dir=self.temp_metadata_dir)
        self.assertFalse(result)
        logger.info("Correctly handled non-existent file in process_ecg_file")
    
    def test_process_ecg_file_corrupted(self):
        """
        Test processing a corrupted ECG file.
        
        Since we may not have an actual corrupted file, we'll simulate 
        corruption by intercepting the load_header function.
        """
        self.skip_without_sample = True
        
        with patch("preprocess.preprocess_ptbxl_data.load_header", return_value=None):
            # Process should fail gracefully - explicitly pass temp dirs
            result = process_ecg_file(
                self.sample_file,
                signal_dir=self.temp_signals_dir,
                metadata_dir=self.temp_metadata_dir
            )
            self.assertFalse(result)
            logger.info("Correctly handled corrupted file in process_ecg_file")


if __name__ == '__main__':
    unittest.main()