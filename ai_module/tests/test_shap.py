#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the SHAP explainer module.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import unittest
import tempfile
import json
import numpy as np
import torch
import shap
from pathlib import Path
from typing import Dict, Optional, Union, Any, List

# Import modules to test
from ai_module.shap_analysis.shap_explainer import SHAPExplainer
from ai_module.models.base_model import BaseModelWrapper, PyTorchModelWrapper, SKLearnModelWrapper

# Create mock model classes for testing
class MockSKLearnWrapper(SKLearnModelWrapper):
    """Mock sklearn model wrapper for testing"""
    
    def _initialize_model(self) -> None:
        """Initialize with a simple isolation forest"""
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(random_state=42, n_estimators=10)
        self.model.fit(np.random.random((10, 5)))
    
    def predict(self, X, **kwargs):
        """Return anomaly scores"""
        return self.model.decision_function(X)
    
    def train(self, X, y=None, **kwargs):
        """Train the model"""
        self.model.fit(X)
        return {"loss": 0.1}
    
    def evaluate(self, X, y=None, **kwargs):
        """Evaluate the model"""
        return {"score": 0.9}
    
    def save(self, path):
        """Save the model"""
        return str(path)
    
    def load(self, path):
        """Load the model"""
        pass


class MockPyTorchWrapper(PyTorchModelWrapper):
    """Mock PyTorch model wrapper for testing"""
    
    class MockAutoEncoder(torch.nn.Module):
        """Simple autoencoder model for testing"""
        def __init__(self, input_dim=5, latent_dim=2):
            super().__init__()
            self.encoder = torch.nn.Linear(input_dim, latent_dim)
            self.decoder = torch.nn.Linear(latent_dim, input_dim)
        
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
    
    def _initialize_model(self) -> None:
        """Initialize with a simple autoencoder"""
        self.model = self.MockAutoEncoder(input_dim=5, latent_dim=2)
    
    def predict(self, X, **kwargs):
        """Return reconstruction error"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed, _ = self.model(X)
            error = torch.mean((reconstructed - X) ** 2, dim=1)
        
        return error.numpy()
    
    def train(self, X, y=None, **kwargs):
        """Train the model"""
        return {"loss": 0.1}
    
    def evaluate(self, X, y=None, **kwargs):
        """Evaluate the model"""
        return {"reconstruction_error": 0.1}
    
    def save(self, path):
        """Save the model"""
        return str(path)
    
    def load(self, path):
        """Load the model"""
        pass


class TestSHAPExplainer(unittest.TestCase):
    """Test the SHAPExplainer class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temp directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Create test data
        self.X = np.random.random((10, 5))
        self.feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        
        # Initialize models
        self.sklearn_model = MockSKLearnWrapper(model_type="isolation_forest")
        self.pytorch_model = MockPyTorchWrapper(model_type="autoencoder")
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_init_sklearn_model(self):
        """Test initialization with scikit-learn model"""
        # Initialize explainer without background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            output_dir=self.output_dir
        )
        
        # Check that explainer is initialized
        self.assertEqual(explainer.model_type, "isolation_forest")
        self.assertEqual(explainer.feature_names, self.feature_names)
        self.assertEqual(explainer.output_dir, self.output_dir)
        self.assertIsNone(explainer.explainer)  # Should be None until setup_explainer is called
    
    def test_init_pytorch_model(self):
        """Test initialization with PyTorch model"""
        # Initialize explainer without background data
        explainer = SHAPExplainer(
            model=self.pytorch_model,
            feature_names=self.feature_names,
            output_dir=self.output_dir
        )
        
        # Check that explainer is initialized
        self.assertEqual(explainer.model_type, "autoencoder")
        self.assertEqual(explainer.feature_names, self.feature_names)
        self.assertEqual(explainer.output_dir, self.output_dir)
        self.assertIsNone(explainer.explainer)  # Should be None until setup_explainer is called
    
    def test_setup_sklearn_explainer(self):
        """Test setup_explainer with scikit-learn model"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Check that explainer is set up
        self.assertIsNotNone(explainer.explainer)
        self.assertIsInstance(explainer.explainer, shap.Explainer)
    
    def test_setup_pytorch_explainer(self):
        """Test setup_explainer with PyTorch model"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.pytorch_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Check that explainer is set up
        self.assertIsNotNone(explainer.explainer)
        self.assertIsInstance(explainer.explainer, shap.Explainer)
    
    def test_explain_sklearn_model(self):
        """Test explain method with scikit-learn model"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Call explain method
        shap_values, data_subset = explainer.explain(self.X, max_samples=5)
        
        # Check shap_values shape
        self.assertEqual(shap_values.shape, (5, 5))  # 5 samples, 5 features
        
        # Check data_subset shape
        self.assertEqual(data_subset.shape, (5, 5))  # 5 samples, 5 features
    
    def test_explain_pytorch_model(self):
        """Test explain method with PyTorch model"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.pytorch_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Call explain method
        shap_values, data_subset = explainer.explain(self.X, max_samples=5)
        
        # Check shap_values shape
        self.assertEqual(shap_values.shape, (5, 5))  # 5 samples, 5 features
        
        # Check data_subset shape
        self.assertEqual(data_subset.shape, (5, 5))  # 5 samples, 5 features
    
    def test_explain_with_sample_indices(self):
        """Test explain method with specific sample indices"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Call explain method with specific indices
        sample_indices = [1, 3, 5]
        shap_values, data_subset = explainer.explain(self.X, sample_indices=sample_indices)
        
        # Check shap_values shape
        self.assertEqual(shap_values.shape, (3, 5))  # 3 samples, 5 features
        
        # Check data_subset shape
        self.assertEqual(data_subset.shape, (3, 5))  # 3 samples, 5 features
        
        # Check that data_subset contains the requested samples
        for i, sample_idx in enumerate(sample_indices):
            np.testing.assert_array_equal(data_subset[i], self.X[sample_idx])
    
    def test_explain_with_batching(self):
        """Test explain method with batching"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Create larger test data
        large_X = np.random.random((20, 5))
        
        # Call explain method with batching
        shap_values, data_subset = explainer.explain(large_X, max_samples=10, batch_size=3)
        
        # Check shap_values shape
        self.assertEqual(shap_values.shape, (10, 5))  # 10 samples, 5 features
        
        # Check data_subset shape
        self.assertEqual(data_subset.shape, (10, 5))  # 10 samples, 5 features
    
    def test_get_top_features(self):
        """Test get_top_features method"""
        # Create test SHAP values
        shap_values = np.array([
            [0.1, 0.5, 0.2, 0.3, 0.1],
            [0.2, 0.6, 0.3, 0.1, 0.2],
            [0.3, 0.4, 0.1, 0.2, 0.3]
        ])
        
        # Initialize explainer
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            output_dir=self.output_dir
        )
        
        # Get top features
        top_features = explainer.get_top_features(shap_values, top_k=3)
        
        # Check result format
        self.assertEqual(len(top_features), 3)
        self.assertIsInstance(top_features[0], tuple)
        self.assertEqual(len(top_features[0]), 2)
        
        # Feature 2 (index 1) should be the most important
        self.assertEqual(top_features[0][0], "feature_2")
        
        # Check descending order
        self.assertGreaterEqual(top_features[0][1], top_features[1][1])
        self.assertGreaterEqual(top_features[1][1], top_features[2][1])
    
    def test_save_shap_values(self):
        """Test save_shap_values method"""
        # Create test SHAP values
        shap_values = np.random.random((5, 5))
        data = np.random.random((5, 5))
        
        # Initialize explainer
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            output_dir=self.output_dir
        )
        
        # Save SHAP values
        output_file = explainer.save_shap_values(shap_values, data)
        
        # Check that file exists
        self.assertTrue(output_file.exists())
        
        # Load and check saved data
        with np.load(output_file) as loaded_data:
            self.assertTrue("shap_values" in loaded_data)
            self.assertTrue("data" in loaded_data)
            self.assertTrue("feature_names" in loaded_data)
            self.assertTrue("model_type" in loaded_data)
            
            np.testing.assert_array_equal(loaded_data["shap_values"], shap_values)
            np.testing.assert_array_equal(loaded_data["data"], data)
    
    def test_load_shap_values(self):
        """Test load_shap_values method"""
        # Create test SHAP values
        shap_values = np.random.random((5, 5))
        data = np.random.random((5, 5))
        
        # Save as npz
        output_file = self.output_dir / "test_shap_values.npz"
        np.savez(
            output_file,
            shap_values=shap_values,
            data=data,
            model_type="test_model",
            feature_names=self.feature_names
        )
        
        # Load SHAP values
        loaded_data = SHAPExplainer.load_shap_values(output_file)
        
        # Check loaded data
        self.assertIn("shap_values", loaded_data)
        self.assertIn("data", loaded_data)
        self.assertIn("model_type", loaded_data)
        self.assertIn("feature_names", loaded_data)
        
        np.testing.assert_array_equal(loaded_data["shap_values"], shap_values)
        np.testing.assert_array_equal(loaded_data["data"], data)
    
    def test_plot_summary(self):
        """Test plot_summary method"""
        # Create test SHAP values
        shap_values = np.random.random((5, 5))
        data = np.random.random((5, 5))
        
        # Initialize explainer
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            output_dir=self.output_dir
        )
        
        # Create plot without output file
        explainer.plot_summary(shap_values, data)
        
        # Check that at least one plot file was created
        plot_files = list(self.output_dir.glob("*.png"))
        self.assertGreater(len(plot_files), 0)
        
        # Clear output directory
        for file in plot_files:
            file.unlink()
        
        # Create plot with output file
        output_file = self.output_dir / "summary_plot.png"
        explainer.plot_summary(shap_values, data, output_file=output_file)
        
        # Check that specified file was created
        self.assertTrue(output_file.exists())
    
    def test_explain_sample(self):
        """Test explain_sample method"""
        # Initialize explainer with background data
        explainer = SHAPExplainer(
            model=self.sklearn_model,
            feature_names=self.feature_names,
            background_data=self.X,
            output_dir=self.output_dir
        )
        
        # Create test sample
        test_sample = np.random.random((1, 5))
        
        # Call explain_sample method
        result = explainer.explain_sample(test_sample)
        
        # Check result structure
        self.assertIn("model_type", result)
        self.assertIn("shap_values", result)
        self.assertIn("data", result)
        self.assertIn("top_features", result)
        self.assertIn("plots", result)
        
        # Check SHAP values shape
        self.assertEqual(result["shap_values"].shape, (1, 5))
        
        # Check that at least one plot was created
        self.assertGreater(len(result["plots"]), 0)
        
        # Check that top features were identified
        self.assertEqual(len(result["top_features"]), 10)  # Default is top 10
        
        # Check that SHAP values were saved
        if "shap_values_path" in result:
            self.assertTrue(Path(result["shap_values_path"]).exists())


if __name__ == "__main__":
    unittest.main() 