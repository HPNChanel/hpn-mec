#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the data pipeline components.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import csv

# Import modules to test
from ai_module.preprocess.data_loader import CSVLoader, FeatureProcessor, DataNormalizer
from ai_module.utils.config import get_config

class TestCSVLoader(unittest.TestCase):
    """Test the CSVLoader class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temp directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create sample CSV files
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def create_sample_data(self):
        """Create sample CSV files for testing"""
        # Create a data directory structure
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV file
        self.test_csv_path = self.raw_dir / "test_data.csv"
        
        # Create a sample dataframe
        data = {
            "patient_id": [1, 2, 3, 4, 5],
            "age": [45, 32, 56, 67, 38],
            "heart_rate": [75, 68, 90, 82, 72],
            "blood_pressure_systolic": [120, 110, 140, 150, 115],
            "blood_pressure_diastolic": [80, 70, 95, 92, 75],
            "cholesterol": [200, 180, 240, 250, 190],
            "anomaly": [0, 0, 1, 1, 0]
        }
        
        # Write dataframe to CSV
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv_path, index=False)
        
        # Create a JSON metadata file
        metadata = {
            "features": ["age", "heart_rate", "blood_pressure_systolic", "blood_pressure_diastolic", "cholesterol"],
            "target": "anomaly",
            "id_column": "patient_id"
        }
        
        self.metadata_path = self.raw_dir / "metadata.json"
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def test_init(self):
        """Test initialization"""
        loader = CSVLoader(data_path=self.test_csv_path)
        
        self.assertEqual(loader.data_path, self.test_csv_path)
        self.assertIsNone(loader.metadata_path)
        self.assertIsNone(loader.data)
        
        # Test with metadata path
        loader = CSVLoader(data_path=self.test_csv_path, metadata_path=self.metadata_path)
        self.assertEqual(loader.metadata_path, self.metadata_path)
    
    def test_load_data(self):
        """Test loading data from CSV"""
        loader = CSVLoader(data_path=self.test_csv_path)
        
        # Load data
        data = loader.load()
        
        # Check that data was loaded correctly
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (5, 7))
        self.assertEqual(list(data.columns), [
            "patient_id", "age", "heart_rate", "blood_pressure_systolic", 
            "blood_pressure_diastolic", "cholesterol", "anomaly"
        ])
    
    def test_load_data_with_metadata(self):
        """Test loading data with metadata"""
        loader = CSVLoader(data_path=self.test_csv_path, metadata_path=self.metadata_path)
        
        # Load data and apply metadata filtering
        data = loader.load(apply_metadata=True)
        
        # Check that data was loaded and filtered correctly
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (5, 6))  # Should include features + target
        
        # Check columns
        expected_columns = ["patient_id", "age", "heart_rate", "blood_pressure_systolic", 
                            "blood_pressure_diastolic", "cholesterol", "anomaly"]
        self.assertEqual(list(data.columns), expected_columns)
    
    def test_load_with_feature_subset(self):
        """Test loading specific feature subset"""
        loader = CSVLoader(data_path=self.test_csv_path)
        
        # Load data with specific columns
        columns = ["patient_id", "age", "heart_rate", "anomaly"]
        data = loader.load(columns=columns)
        
        # Check that only specified columns were loaded
        self.assertEqual(list(data.columns), columns)
        self.assertEqual(data.shape, (5, 4))
    
    def test_split_features_target(self):
        """Test splitting features and target"""
        loader = CSVLoader(data_path=self.test_csv_path, metadata_path=self.metadata_path)
        
        # Load data
        data = loader.load()
        
        # Split features and target
        X, y = loader.split_features_target(data, target_column="anomaly")
        
        # Check shapes
        self.assertEqual(X.shape, (5, 6))  # All columns except target
        self.assertEqual(y.shape, (5,))
        
        # Check that target column is not in features
        self.assertNotIn("anomaly", X.columns)


class TestFeatureProcessor(unittest.TestCase):
    """Test the FeatureProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create sample data
        self.data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "numeric_feature": [10, 20, 30, 40, 50],
            "categorical_feature": ["A", "B", "A", "C", "B"],
            "text_feature": ["text one", "text two", "text three", "text four", "text five"],
            "missing_feature": [1.0, 2.0, np.nan, 4.0, np.nan],
            "constant_feature": [5, 5, 5, 5, 5],
            "binary_feature": [0, 1, 0, 1, 0],
            "target": [0, 0, 1, 1, 0]
        })
    
    def test_init(self):
        """Test initialization"""
        processor = FeatureProcessor()
        
        self.assertIsNotNone(processor.numeric_features)
        self.assertIsNotNone(processor.categorical_features)
        self.assertIsNotNone(processor.text_features)
        self.assertIsNotNone(processor.drop_features)
    
    def test_detect_feature_types(self):
        """Test automatic feature type detection"""
        processor = FeatureProcessor()
        
        # Detect feature types
        processor.detect_feature_types(self.data)
        
        # Check numeric features
        self.assertIn("numeric_feature", processor.numeric_features)
        self.assertIn("missing_feature", processor.numeric_features)
        self.assertIn("binary_feature", processor.numeric_features)
        
        # Check categorical features
        self.assertIn("categorical_feature", processor.categorical_features)
        
        # Check if constant feature is detected for dropping
        self.assertIn("constant_feature", processor.drop_features)
    
    def test_fill_missing_values(self):
        """Test filling missing values"""
        processor = FeatureProcessor()
        
        # Setup processor
        processor.numeric_features = ["numeric_feature", "missing_feature"]
        processor.categorical_features = ["categorical_feature"]
        
        # Process data
        result = processor.fill_missing_values(self.data)
        
        # Check that missing values are filled
        self.assertEqual(result["missing_feature"].isna().sum(), 0)
        
        # Check inferred fill values (mean for numeric)
        self.assertEqual(result.loc[2, "missing_feature"], np.mean([1.0, 2.0, 4.0]))
    
    def test_encode_categorical(self):
        """Test encoding categorical features"""
        processor = FeatureProcessor()
        
        # Setup processor
        processor.categorical_features = ["categorical_feature"]
        
        # Process data
        result = processor.encode_categorical(self.data)
        
        # Check that original categorical column is removed
        self.assertNotIn("categorical_feature", result.columns)
        
        # Check that new one-hot columns are created
        self.assertIn("categorical_feature_A", result.columns)
        self.assertIn("categorical_feature_B", result.columns)
        self.assertIn("categorical_feature_C", result.columns)
        
        # Check encoding correctness
        self.assertEqual(result.loc[0, "categorical_feature_A"], 1)
        self.assertEqual(result.loc[0, "categorical_feature_B"], 0)
        self.assertEqual(result.loc[0, "categorical_feature_C"], 0)
    
    def test_drop_constant_and_id(self):
        """Test dropping constant and ID features"""
        processor = FeatureProcessor()
        
        # Setup processor
        processor.drop_features = ["constant_feature", "id"]
        
        # Process data
        result = processor.drop_features_from_df(self.data)
        
        # Check that dropped columns are not in result
        self.assertNotIn("constant_feature", result.columns)
        self.assertNotIn("id", result.columns)
    
    def test_process(self):
        """Test complete processing pipeline"""
        processor = FeatureProcessor()
        
        # Setting up feature sets
        processor.numeric_features = ["numeric_feature", "missing_feature", "binary_feature"]
        processor.categorical_features = ["categorical_feature"]
        processor.text_features = ["text_feature"]
        processor.drop_features = ["constant_feature", "id"]
        
        # Process data
        result, feature_names = processor.process(self.data, target_column="target")
        
        # Check result shape (assuming one-hot encoding creates 3 columns)
        expected_cols = 7  # numeric(3) + categorical_onehot(3) + text(skipped) - drop(2)
        self.assertEqual(result.shape, (5, expected_cols))
        
        # Check feature names list
        self.assertEqual(len(feature_names), expected_cols - 1)  # excluding target
        
        # Check that target is included in result
        self.assertIn("target", result.columns)
        
        # Check basic processing steps were applied
        self.assertNotIn("constant_feature", result.columns)  # dropped
        self.assertIn("categorical_feature_A", result.columns)  # encoded
        self.assertEqual(result["missing_feature"].isna().sum(), 0)  # filled


class TestDataNormalizer(unittest.TestCase):
    """Test the DataNormalizer class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create sample data
        self.X = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [100, 200, 300, 400, 500],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        self.X_np = self.X.values
    
    def test_init(self):
        """Test initialization"""
        normalizer = DataNormalizer()
        
        self.assertIsNone(normalizer.mean)
        self.assertIsNone(normalizer.std)
        self.assertIsNone(normalizer.min)
        self.assertIsNone(normalizer.max)
    
    def test_fit_transform_standardization(self):
        """Test standardization (z-score normalization)"""
        normalizer = DataNormalizer(method="standardize")
        
        # Fit and transform
        X_norm = normalizer.fit_transform(self.X_np)
        
        # Check shape
        self.assertEqual(X_norm.shape, self.X_np.shape)
        
        # Check that mean is close to 0 and std is close to 1 for each feature
        feature_means = X_norm.mean(axis=0)
        feature_stds = X_norm.std(axis=0)
        
        for mean, std in zip(feature_means, feature_stds):
            self.assertAlmostEqual(mean, 0, places=10)
            self.assertAlmostEqual(std, 1, places=10)
        
        # Check that mean and std are saved
        self.assertIsNotNone(normalizer.mean)
        self.assertIsNotNone(normalizer.std)
    
    def test_fit_transform_minmax(self):
        """Test min-max normalization"""
        normalizer = DataNormalizer(method="minmax")
        
        # Fit and transform
        X_norm = normalizer.fit_transform(self.X_np)
        
        # Check shape
        self.assertEqual(X_norm.shape, self.X_np.shape)
        
        # Check that values are in [0, 1] range
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))
        
        # Check that min and max are saved
        self.assertIsNotNone(normalizer.min)
        self.assertIsNotNone(normalizer.max)
    
    def test_transform_standardization(self):
        """Test transform with pre-fitted standardizer"""
        normalizer = DataNormalizer(method="standardize")
        
        # Fit on training data
        normalizer.fit(self.X_np)
        
        # Transform new data
        new_data = np.array([[10, 1000, 1.0], [0, 0, 0.0]])
        X_norm = normalizer.transform(new_data)
        
        # Check shape
        self.assertEqual(X_norm.shape, new_data.shape)
        
        # Manual standardization for verification
        expected = (new_data - normalizer.mean) / normalizer.std
        np.testing.assert_array_almost_equal(X_norm, expected)
    
    def test_transform_minmax(self):
        """Test transform with pre-fitted min-max normalizer"""
        normalizer = DataNormalizer(method="minmax")
        
        # Fit on training data
        normalizer.fit(self.X_np)
        
        # Transform new data
        new_data = np.array([[10, 1000, 1.0], [0, 0, 0.0]])
        X_norm = normalizer.transform(new_data)
        
        # Check shape
        self.assertEqual(X_norm.shape, new_data.shape)
        
        # Manual min-max normalization for verification
        expected = (new_data - normalizer.min) / (normalizer.max - normalizer.min)
        np.testing.assert_array_almost_equal(X_norm, expected)
    
    def test_inverse_transform_standardization(self):
        """Test inverse transform with standardization"""
        normalizer = DataNormalizer(method="standardize")
        
        # Fit and transform
        X_norm = normalizer.fit_transform(self.X_np)
        
        # Inverse transform
        X_restored = normalizer.inverse_transform(X_norm)
        
        # Check that original data is restored
        np.testing.assert_array_almost_equal(X_restored, self.X_np)
    
    def test_inverse_transform_minmax(self):
        """Test inverse transform with min-max"""
        normalizer = DataNormalizer(method="minmax")
        
        # Fit and transform
        X_norm = normalizer.fit_transform(self.X_np)
        
        # Inverse transform
        X_restored = normalizer.inverse_transform(X_norm)
        
        # Check that original data is restored
        np.testing.assert_array_almost_equal(X_restored, self.X_np)
    
    def test_save_load(self):
        """Test saving and loading normalizer"""
        normalizer = DataNormalizer(method="standardize")
        
        # Fit normalizer
        normalizer.fit(self.X_np)
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save normalizer
            save_path = Path(temp_dir) / "normalizer.json"
            normalizer.save(save_path)
            
            # Check that file exists
            self.assertTrue(save_path.exists())
            
            # Load normalizer
            loaded_normalizer = DataNormalizer.load(save_path)
            
            # Check that loaded normalizer has the same parameters
            self.assertEqual(loaded_normalizer.method, normalizer.method)
            np.testing.assert_array_equal(loaded_normalizer.mean, normalizer.mean)
            np.testing.assert_array_equal(loaded_normalizer.std, normalizer.std)
            
            # Check that transform results are the same
            test_data = np.array([[10, 1000, 1.0]])
            original_result = normalizer.transform(test_data)
            loaded_result = loaded_normalizer.transform(test_data)
            
            np.testing.assert_array_almost_equal(original_result, loaded_result)


if __name__ == "__main__":
    unittest.main() 