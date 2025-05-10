#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the model registry module.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Union, Any

# Import modules to test
from ai_module.utils.model_registry import get_registry, ModelRegistry
from ai_module.models.base_model import BaseModelWrapper, PyTorchModelWrapper, SKLearnModelWrapper

# Create mock model classes for testing
class MockModelWrapper(BaseModelWrapper):
    """Mock model wrapper for testing"""
    
    def _initialize_model(self) -> None:
        self.model = "mock_model"
    
    def train(self, X, y=None, **kwargs):
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        return np.zeros(len(X))
    
    def evaluate(self, X, y=None, **kwargs):
        return {"accuracy": 0.9}
    
    def save(self, path):
        return str(path)
    
    def load(self, path):
        pass

class MockPyTorchModel(PyTorchModelWrapper):
    """Mock PyTorch model wrapper for testing"""
    
    def _initialize_model(self) -> None:
        import torch.nn as nn
        
        # Simple linear model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.linear(x)
        
        self.model = SimpleModel()
    
    def train(self, X, y=None, **kwargs):
        return {"loss": 0.1}

class MockSKLearnModel(SKLearnModelWrapper):
    """Mock scikit-learn model wrapper for testing"""
    
    def _initialize_model(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10)
    
    def train(self, X, y=None, **kwargs):
        return {"loss": 0.1}


class TestModelRegistry(unittest.TestCase):
    """Test the ModelRegistry class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a fresh registry for each test
        self.registry = ModelRegistry()
        
        # Clear registry between tests
        self.registry._registry = {}
        self.registry._default_versions = {}
        
        # Create temp directory for model paths
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        
        # Create mock model paths
        self.mock_model_path = self.model_dir / "mock_model.pt"
        self.mock_model_path.write_text("mock model data")
        
        # Register models for testing
        self.registry.register(
            name="mock_model",
            model_class=MockModelWrapper,
            version="1.0.0",
            model_path=self.mock_model_path,
            is_default=True
        )
        
        self.registry.register(
            name="mock_model",
            model_class=MockModelWrapper,
            version="2.0.0",
            model_path=self.mock_model_path,
            is_default=False
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_singleton_pattern(self):
        """Test that the registry uses the singleton pattern"""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        
        # Should be the same instance
        self.assertIs(registry1, registry2)
        
        # Should have the same registry dictionary
        self.assertIs(registry1._registry, registry2._registry)
    
    def test_register_model(self):
        """Test registering models"""
        # Test that models were registered correctly
        self.assertIn("mock_model", self.registry._registry)
        self.assertEqual(self.registry._registry["mock_model"]["class"], MockModelWrapper)
        self.assertEqual(len(self.registry._registry["mock_model"]["versions"]), 2)
        self.assertIn("1.0.0", self.registry._registry["mock_model"]["versions"])
        self.assertIn("2.0.0", self.registry._registry["mock_model"]["versions"])
    
    def test_register_decorator(self):
        """Test the registration decorator"""
        # Register a model using the decorator
        @self.registry.register_decorator("decorated_model", version="1.0.0")
        class DecoratedModel(MockModelWrapper):
            pass
        
        # Check that the model was registered correctly
        self.assertIn("decorated_model", self.registry._registry)
        self.assertEqual(self.registry._registry["decorated_model"]["class"], DecoratedModel)
        self.assertIn("1.0.0", self.registry._registry["decorated_model"]["versions"])
    
    def test_get_model_class(self):
        """Test getting model class"""
        model_class = self.registry.get_model_class("mock_model")
        self.assertEqual(model_class, MockModelWrapper)
        
        # Test with nonexistent model
        with self.assertRaises(KeyError):
            self.registry.get_model_class("nonexistent_model")
    
    def test_get_model_path(self):
        """Test getting model path"""
        # Test with specific version
        model_path = self.registry.get_model_path("mock_model", "1.0.0")
        self.assertEqual(model_path, self.mock_model_path)
        
        # Test with default version
        model_path = self.registry.get_model_path("mock_model")
        self.assertEqual(model_path, self.mock_model_path)
        
        # Test with nonexistent model
        with self.assertRaises(KeyError):
            self.registry.get_model_path("nonexistent_model")
        
        # Test with nonexistent version
        with self.assertRaises(KeyError):
            self.registry.get_model_path("mock_model", "3.0.0")
    
    def test_create_model(self):
        """Test creating model instance"""
        # Create model with default version
        model = self.registry.create_model("mock_model")
        self.assertIsInstance(model, MockModelWrapper)
        self.assertEqual(model.model_type, "mock_model")
        
        # Create model with specific version
        model = self.registry.create_model("mock_model", "2.0.0")
        self.assertIsInstance(model, MockModelWrapper)
        
        # Create model with custom arguments
        model = self.registry.create_model("mock_model", model_params={"param1": "value1"})
        self.assertIsInstance(model, MockModelWrapper)
        self.assertEqual(model.model_params.get("param1"), "value1")
        
        # Test with nonexistent model
        with self.assertRaises(KeyError):
            self.registry.create_model("nonexistent_model")
    
    def test_list_models(self):
        """Test listing models"""
        models = self.registry.list_models()
        self.assertIn("mock_model", models)
        
        # Add another model and check list again
        self.registry.register(
            name="another_model",
            model_class=MockModelWrapper,
            version="1.0.0"
        )
        
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)
        self.assertIn("mock_model", models)
        self.assertIn("another_model", models)
    
    def test_list_versions(self):
        """Test listing versions"""
        versions = self.registry.list_versions("mock_model")
        self.assertIn("1.0.0", versions)
        self.assertIn("2.0.0", versions)
        
        # Test with nonexistent model
        with self.assertRaises(KeyError):
            self.registry.list_versions("nonexistent_model")
    
    def test_default_version(self):
        """Test default version handling"""
        # Test getting default version
        default_version = self.registry.get_default_version("mock_model")
        self.assertEqual(default_version, "1.0.0")
        
        # Test setting default version
        self.registry.set_default_version("mock_model", "2.0.0")
        default_version = self.registry.get_default_version("mock_model")
        self.assertEqual(default_version, "2.0.0")
        
        # Test with nonexistent model
        with self.assertRaises(KeyError):
            self.registry.get_default_version("nonexistent_model")
        
        # Test setting default with nonexistent version
        with self.assertRaises(KeyError):
            self.registry.set_default_version("mock_model", "3.0.0")
    
    def test_get_registry_function(self):
        """Test get_registry function"""
        registry = get_registry()
        self.assertIsInstance(registry, ModelRegistry)
        
        # Should return the same instance
        registry2 = get_registry()
        self.assertIs(registry, registry2)
    
    def test_validate_model_class(self):
        """Test validation of model class"""
        # Test with valid model class
        self.registry.register(
            name="valid_model",
            model_class=MockPyTorchModel,
            version="1.0.0"
        )
        
        # Test with invalid model class
        class InvalidModel:
            pass
        
        with self.assertRaises(TypeError):
            self.registry.register(
                name="invalid_model",
                model_class=InvalidModel,
                version="1.0.0"
            )


if __name__ == "__main__":
    unittest.main() 