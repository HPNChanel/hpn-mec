#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the model evaluator module.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Union, Any

# Import modules to test
from ai_module.utils.evaluator import ModelEvaluator, MetricResult, get_evaluator
from ai_module.models.base_model import BaseModelWrapper

# Create mock model class for testing
class MockModelWrapper(BaseModelWrapper):
    """Mock model wrapper for testing evaluator"""
    
    def __init__(
        self,
        model_type: str,
        model_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[Union[str, Path]] = None
    ):
        super().__init__(model_type, model_params, model_path)
        self.predict_return_value = np.array([0.1, 0.8, 0.2, 0.9, 0.05])
        self.metadata = {
            "model_type": model_type,
            "version": "1.0.0"
        }
    
    def _initialize_model(self) -> None:
        self.model = "mock_model"
    
    def train(self, X, y=None, **kwargs):
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        # Return pre-defined probabilities for testing
        if len(X) != len(self.predict_return_value):
            # Create random predictions if length doesn't match
            return np.random.random(len(X))
        return self.predict_return_value
    
    def evaluate(self, X, y=None, **kwargs):
        return {"accuracy": 0.9}
    
    def save(self, path):
        return str(path)
    
    def load(self, path):
        pass
    
    def get_metadata(self):
        return self.metadata


class TestMetricResult(unittest.TestCase):
    """Test the MetricResult class"""
    
    def setUp(self):
        """Set up test environment"""
        self.model_name = "test_model"
        self.model_version = "1.0.0"
        self.metrics = {
            "accuracy": 0.95,
            "f1": 0.92,
            "roc_auc": 0.98
        }
        self.metadata = {
            "threshold": 0.5,
            "n_samples": 100
        }
        
        self.result = MetricResult(
            model_name=self.model_name,
            model_version=self.model_version,
            metrics=self.metrics,
            metadata=self.metadata
        )
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.result.model_name, self.model_name)
        self.assertEqual(self.result.model_version, self.model_version)
        self.assertEqual(self.result.metrics, self.metrics)
        self.assertEqual(self.result.metadata, self.metadata)
        self.assertIsNotNone(self.result.timestamp)
    
    def test_add_metric(self):
        """Test adding a single metric"""
        # Add a new metric
        self.result.add_metric("precision", 0.93)
        self.assertEqual(self.result.metrics["precision"], 0.93)
        
        # Update existing metric
        self.result.add_metric("accuracy", 0.96)
        self.assertEqual(self.result.metrics["accuracy"], 0.96)
    
    def test_add_metrics(self):
        """Test adding multiple metrics"""
        new_metrics = {
            "precision": 0.93,
            "recall": 0.91
        }
        self.result.add_metrics(new_metrics)
        
        # Check that metrics were added
        self.assertEqual(self.result.metrics["precision"], 0.93)
        self.assertEqual(self.result.metrics["recall"], 0.91)
        
        # Check that existing metrics remain
        self.assertEqual(self.result.metrics["accuracy"], 0.95)
    
    def test_get_metric(self):
        """Test getting a metric value"""
        # Get existing metric
        self.assertEqual(self.result.get_metric("accuracy"), 0.95)
        
        # Get nonexistent metric with default
        self.assertEqual(self.result.get_metric("nonexistent", -1), -1)
        
        # Get nonexistent metric without default
        self.assertIsNone(self.result.get_metric("nonexistent"))
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result_dict = self.result.to_dict()
        
        self.assertEqual(result_dict["model_name"], self.model_name)
        self.assertEqual(result_dict["model_version"], self.model_version)
        self.assertEqual(result_dict["metrics"], self.metrics)
        self.assertEqual(result_dict["metadata"], self.metadata)
        self.assertIn("timestamp", result_dict)
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        result_dict = {
            "model_name": "model_from_dict",
            "model_version": "2.0.0",
            "metrics": {"precision": 0.9, "recall": 0.85},
            "metadata": {"threshold": 0.7},
            "timestamp": "2025-06-10T12:00:00"
        }
        
        result = MetricResult.from_dict(result_dict)
        
        self.assertEqual(result.model_name, "model_from_dict")
        self.assertEqual(result.model_version, "2.0.0")
        self.assertEqual(result.metrics, {"precision": 0.9, "recall": 0.85})
        self.assertEqual(result.metadata, {"threshold": 0.7})
        self.assertEqual(result.timestamp, "2025-06-10T12:00:00")
    
    def test_save_load(self):
        """Test saving and loading from file"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)
        
        try:
            # Save to file
            self.result.save(temp_path)
            
            # Load from file
            loaded_result = MetricResult.load(temp_path)
            
            # Check that loaded result matches original
            self.assertEqual(loaded_result.model_name, self.model_name)
            self.assertEqual(loaded_result.model_version, self.model_version)
            self.assertEqual(loaded_result.metrics, self.metrics)
            self.assertEqual(loaded_result.metadata, self.metadata)
            self.assertEqual(loaded_result.timestamp, self.result.timestamp)
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)
    
    def test_str(self):
        """Test string representation"""
        result_str = str(self.result)
        
        # Should contain model name and version
        self.assertIn(self.model_name, result_str)
        self.assertIn(self.model_version, result_str)
        
        # Should contain metrics
        for metric_name, metric_value in self.metrics.items():
            self.assertIn(f"{metric_name}=", result_str)


class TestModelEvaluator(unittest.TestCase):
    """Test the ModelEvaluator class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temp directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Create evaluator
        self.evaluator = ModelEvaluator(output_dir=self.output_dir)
        
        # Create mock model
        self.model = MockModelWrapper(model_type="test_model")
        
        # Create test data
        self.X = np.random.random((5, 10))
        self.y = np.array([0, 1, 0, 1, 0])
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization"""
        # Check that output directory was created
        self.assertTrue(self.output_dir.exists())
        
        # Check initialization with None output_dir
        default_evaluator = ModelEvaluator()
        self.assertIsNotNone(default_evaluator.output_dir)
    
    def test_evaluate(self):
        """Test basic evaluation"""
        # Evaluate model
        result = self.evaluator.evaluate(
            model=self.model,
            X=self.X,
            y=self.y,
            generate_plots=False,
            save_results=False
        )
        
        # Check result type
        self.assertIsInstance(result, MetricResult)
        
        # Check result contents
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.model_version, "1.0.0")
        
        # Check that metrics were calculated
        self.assertIn("roc_auc", result.metrics)
        self.assertIn("pr_auc", result.metrics)
        self.assertIn("accuracy", result.metrics)
        self.assertIn("f1", result.metrics)
        self.assertIn("precision", result.metrics)
        self.assertIn("recall", result.metrics)
        
        # Check that confusion matrix elements were calculated
        self.assertIn("true_positives", result.metrics)
        self.assertIn("false_positives", result.metrics)
        self.assertIn("true_negatives", result.metrics)
        self.assertIn("false_negatives", result.metrics)
        
        # Check metadata
        self.assertIn("threshold", result.metadata)
        self.assertIn("n_samples", result.metadata)
        self.assertIn("positive_class_ratio", result.metadata)
        self.assertIn("evaluation_time", result.metadata)
    
    def test_evaluate_with_custom_metrics(self):
        """Test evaluation with custom metrics"""
        # Define custom metrics
        def custom_metric1(y_true, y_pred):
            return np.mean(y_true == (y_pred > 0.5))
        
        def custom_metric2(y_true, y_pred):
            return np.sum(y_true)
        
        custom_metrics = {
            "custom_metric1": custom_metric1,
            "custom_metric2": custom_metric2
        }
        
        # Evaluate model with custom metrics
        result = self.evaluator.evaluate(
            model=self.model,
            X=self.X,
            y=self.y,
            custom_metrics=custom_metrics,
            generate_plots=False,
            save_results=False
        )
        
        # Check that custom metrics were calculated
        self.assertIn("custom_metric1", result.metrics)
        self.assertIn("custom_metric2", result.metrics)
    
    def test_generate_evaluation_plots(self):
        """Test plot generation"""
        # Evaluate model with plot generation
        result = self.evaluator.evaluate(
            model=self.model,
            X=self.X,
            y=self.y,
            generate_plots=True,
            save_results=False
        )
        
        # Check that plot paths were added to metadata
        self.assertIn("roc_curve_path", result.metadata)
        self.assertIn("pr_curve_path", result.metadata)
        self.assertIn("score_dist_path", result.metadata)
        
        # Check that plot files were created
        for path_key in ["roc_curve_path", "pr_curve_path", "score_dist_path"]:
            plot_path = Path(result.metadata[path_key])
            self.assertTrue(plot_path.exists())
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        # Evaluate model with result saving
        result = self.evaluator.evaluate(
            model=self.model,
            X=self.X,
            y=self.y,
            generate_plots=False,
            save_results=True
        )
        
        # Check that results directory was created
        results_dir = self.output_dir / "results"
        self.assertTrue(results_dir.exists())
        
        # Check that at least one result file was created
        result_files = list(results_dir.glob("*.json"))
        self.assertGreater(len(result_files), 0)
        
        # Check that the result file contains the expected data
        with open(result_files[0], 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["model_name"], "test_model")
        self.assertEqual(saved_data["model_version"], "1.0.0")
        self.assertIn("metrics", saved_data)
        self.assertIn("metadata", saved_data)
    
    def test_compare_models(self):
        """Test model comparison"""
        # Create a few MetricResult instances
        result1 = MetricResult(
            model_name="model1",
            model_version="1.0.0",
            metrics={"accuracy": 0.9, "f1": 0.85, "roc_auc": 0.92}
        )
        
        result2 = MetricResult(
            model_name="model2",
            model_version="1.0.0",
            metrics={"accuracy": 0.88, "f1": 0.87, "roc_auc": 0.91}
        )
        
        # Compare models
        df = self.evaluator.compare_models([result1, result2])
        
        # Check dataframe shape
        self.assertEqual(df.shape, (2, 5))  # 2 rows, 5 columns (model, version, 3 metrics)
        
        # Check dataframe content
        self.assertIn("model1", df["model"].values)
        self.assertIn("model2", df["model"].values)
        self.assertIn("accuracy", df.columns)
        self.assertIn("f1", df.columns)
        self.assertIn("roc_auc", df.columns)
        
        # Test with specified metrics
        df_subset = self.evaluator.compare_models([result1, result2], metrics=["accuracy"])
        self.assertEqual(df_subset.shape, (2, 3))  # 2 rows, 3 columns (model, version, 1 metric)
        self.assertIn("accuracy", df_subset.columns)
        self.assertNotIn("f1", df_subset.columns)
        
        # Test with empty list
        df_empty = self.evaluator.compare_models([])
        self.assertEqual(df_empty.shape, (0, 0))
    
    def test_load_result(self):
        """Test loading result from file"""
        # Create and save a result
        result = MetricResult(
            model_name="test_model",
            model_version="1.0.0",
            metrics={"accuracy": 0.9, "f1": 0.85}
        )
        
        result_path = self.output_dir / "test_result.json"
        result.save(result_path)
        
        # Load the result
        loaded_result = self.evaluator.load_result(result_path)
        
        # Check that loaded result matches original
        self.assertEqual(loaded_result.model_name, result.model_name)
        self.assertEqual(loaded_result.model_version, result.model_version)
        self.assertEqual(loaded_result.metrics, result.metrics)
    
    def test_load_all_results(self):
        """Test loading all results"""
        # Create results directory
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Create and save multiple results
        result1 = MetricResult(
            model_name="model1",
            model_version="1.0.0",
            metrics={"accuracy": 0.9}
        )
        result1.save(results_dir / "model1_1.0.0_20250610.json")
        
        result2 = MetricResult(
            model_name="model1",
            model_version="2.0.0",
            metrics={"accuracy": 0.92}
        )
        result2.save(results_dir / "model1_2.0.0_20250610.json")
        
        result3 = MetricResult(
            model_name="model2",
            model_version="1.0.0",
            metrics={"accuracy": 0.88}
        )
        result3.save(results_dir / "model2_1.0.0_20250610.json")
        
        # Load all results
        all_results = self.evaluator.load_all_results()
        self.assertEqual(len(all_results), 3)
        
        # Load filtered by model name
        model1_results = self.evaluator.load_all_results(model_name="model1")
        self.assertEqual(len(model1_results), 2)
        self.assertTrue(all(r.model_name == "model1" for r in model1_results))
        
        # Load filtered by model name and version
        specific_results = self.evaluator.load_all_results(model_name="model1", model_version="1.0.0")
        self.assertEqual(len(specific_results), 1)
        self.assertEqual(specific_results[0].model_name, "model1")
        self.assertEqual(specific_results[0].model_version, "1.0.0")
    
    def test_get_evaluator(self):
        """Test get_evaluator function"""
        # Get evaluator
        evaluator = get_evaluator()
        self.assertIsInstance(evaluator, ModelEvaluator)
        
        # Get evaluator again
        evaluator2 = get_evaluator()
        
        # Should be the same instance
        self.assertIs(evaluator, evaluator2)


if __name__ == "__main__":
    unittest.main() 