#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Model Evaluator for AI Module

This module implements a centralized evaluation system for all models.
It provides standardized metrics calculation, visualization options, and result handling.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

# Import from project
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger
from ai_module.models.base_model import BaseModelWrapper

# Initialize logger
logger = get_logger("utils.evaluator")

# Get configuration
config = get_config()

class MetricResult:
    """
    Container for evaluation metrics with utilities for serialization and comparison.
    """
    
    def __init__(
        self, 
        model_name: str,
        model_version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metric result.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (optional)
            metrics: Dictionary of metric names and values
            metadata: Additional metadata about the evaluation
        """
        self.model_name = model_name
        self.model_version = model_version
        self.metrics = metrics or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the result set."""
        self.metrics[name] = float(value)
    
    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """Add multiple metrics to the result set."""
        for name, value in metrics.items():
            self.add_metric(name, value)
    
    def get_metric(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get a metric value by name."""
        return self.metrics.get(name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricResult':
        """Create from dictionary."""
        result = cls(
            model_name=data["model_name"],
            model_version=data.get("model_version"),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {})
        )
        if "timestamp" in data:
            result.timestamp = data["timestamp"]
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MetricResult':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation with key metrics."""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in self.metrics.items()])
        version_str = f" v{self.model_version}" if self.model_version else ""
        return f"MetricResult({self.model_name}{version_str}: {metrics_str})"


class ModelEvaluator:
    """
    Unified evaluator for all model types.
    
    This class provides standardized evaluation methods for any model
    implementing the BaseModelWrapper interface, with support for common
    metrics and visualizations.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else Path(config.get("paths.evaluation_dir", "reports/evaluation"))
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with output directory: {self.output_dir}")
    
    def evaluate(
        self, 
        model: BaseModelWrapper,
        X: np.ndarray,
        y: np.ndarray,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        threshold: float = 0.5,
        generate_plots: bool = True,
        save_results: bool = True,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate a model on the provided data.
        
        Args:
            model: Model instance implementing BaseModelWrapper
            X: Feature matrix
            y: Ground truth labels
            custom_metrics: Dictionary of custom metric functions
            threshold: Classification threshold for binary predictions
            generate_plots: Whether to generate evaluation plots
            save_results: Whether to save results to disk
            **kwargs: Additional arguments for model.predict
            
        Returns:
            MetricResult containing evaluation metrics
        """
        # Get model metadata
        metadata = model.get_metadata()
        model_name = metadata.get("model_type", "unknown")
        model_version = metadata.get("version", "latest")
        
        # Get predictions
        logger.info(f"Generating predictions for evaluation of {model_name}")
        y_pred_proba = model.predict(X, **kwargs)
        
        # For multi-output models, take the first output or anomaly score
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            # For 2D output, assume first column is the score of interest
            y_pred_proba = y_pred_proba[:, 0]
        
        # Convert to flattened arrays
        y = np.array(y).flatten()
        y_pred_proba = np.array(y_pred_proba).flatten()
        
        # Binary predictions based on threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Initialize metric result
        result = MetricResult(
            model_name=model_name,
            model_version=model_version,
            metadata={
                "threshold": threshold,
                "n_samples": len(X),
                "positive_class_ratio": float(y.sum() / len(y)),
                "evaluation_time": datetime.now().isoformat()
            }
        )
        
        # Calculate standard classification metrics
        try:
            # ROC AUC
            roc_auc = roc_auc_score(y, y_pred_proba)
            result.add_metric("roc_auc", roc_auc)
            
            # PR AUC
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            pr_auc = auc(recall, precision)
            result.add_metric("pr_auc", pr_auc)
            
            # Accuracy
            accuracy = accuracy_score(y, y_pred)
            result.add_metric("accuracy", accuracy)
            
            # F1 Score
            f1 = f1_score(y, y_pred)
            result.add_metric("f1", f1)
            
            # Precision
            precision = precision_score(y, y_pred)
            result.add_metric("precision", precision)
            
            # Recall
            recall = recall_score(y, y_pred)
            result.add_metric("recall", recall)
            
            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            result.add_metric("true_negatives", tn)
            result.add_metric("false_positives", fp)
            result.add_metric("false_negatives", fn)
            result.add_metric("true_positives", tp)
            
            logger.info(f"Calculated standard metrics: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating standard metrics: {e}")
        
        # Calculate custom metrics if provided
        if custom_metrics:
            for name, metric_fn in custom_metrics.items():
                try:
                    value = metric_fn(y, y_pred_proba)
                    result.add_metric(name, value)
                    logger.info(f"Calculated custom metric {name}={value:.4f}")
                except Exception as e:
                    logger.error(f"Error calculating custom metric {name}: {e}")
        
        # Generate plots if requested
        if generate_plots:
            self._generate_evaluation_plots(y, y_pred_proba, result)
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(result)
        
        return result
    
    def _generate_evaluation_plots(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        result: MetricResult
    ) -> None:
        """
        Generate standard evaluation plots.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            result: Metric result to add plot paths to
        """
        model_name = result.model_name
        model_version = result.model_version or "latest"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        try:
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # ROC Curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = result.get_metric("roc_auc")
            
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name} v{model_version}')
            plt.legend(loc="lower right")
            
            roc_path = plots_dir / f"{model_name}_{model_version}_roc_{timestamp}.png"
            plt.savefig(roc_path)
            plt.close()
            result.metadata["roc_curve_path"] = str(roc_path)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = result.get_metric("pr_auc")
            
            plt.figure(figsize=(10, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name} v{model_version}')
            plt.legend(loc="lower left")
            
            pr_path = plots_dir / f"{model_name}_{model_version}_pr_curve_{timestamp}.png"
            plt.savefig(pr_path)
            plt.close()
            result.metadata["pr_curve_path"] = str(pr_path)
            
            # Score Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(
                [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]], 
                bins=50, 
                alpha=0.5, 
                label=['Normal', 'Anomaly']
            )
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.title(f'Score Distribution - {model_name} v{model_version}')
            plt.legend()
            
            dist_path = plots_dir / f"{model_name}_{model_version}_score_dist_{timestamp}.png"
            plt.savefig(dist_path)
            plt.close()
            result.metadata["score_dist_path"] = str(dist_path)
            
            logger.info(f"Generated evaluation plots for {model_name} v{model_version}")
            
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {e}")
    
    def _save_evaluation_results(self, result: MetricResult) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            result: Metric result to save
        """
        try:
            model_name = result.model_name
            model_version = result.model_version or "latest"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # Save JSON results
            results_dir = self.output_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            json_path = results_dir / f"{model_name}_{model_version}_{timestamp}.json"
            result.save(json_path)
            
            logger.info(f"Saved evaluation results to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def compare_models(
        self, 
        results: List[MetricResult],
        metrics: Optional[List[str]] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            results: List of metric results
            metrics: List of metrics to compare (defaults to all common metrics)
            output_file: Path to save comparison table (optional)
            
        Returns:
            DataFrame with model comparison
        """
        if not results:
            logger.warning("No results provided for comparison")
            return pd.DataFrame()
        
        # Get common metrics if not specified
        if metrics is None:
            # Find metrics common to all results
            common_metrics = set(results[0].metrics.keys())
            for result in results[1:]:
                common_metrics &= set(result.metrics.keys())
            metrics = sorted(list(common_metrics))
        
        # Create comparison dataframe
        data = []
        for result in results:
            row = {
                "model": result.model_name,
                "version": result.model_version or "latest"
            }
            for metric in metrics:
                row[metric] = result.get_metric(metric)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix == '.xlsx':
                df.to_excel(output_path, index=False)
            else:
                # Default to CSV
                df.to_csv(output_path.with_suffix('.csv'), index=False)
            
            logger.info(f"Saved model comparison to {output_path}")
        
        return df
    
    def load_result(self, path: Union[str, Path]) -> MetricResult:
        """
        Load a result from disk.
        
        Args:
            path: Path to result JSON file
            
        Returns:
            Loaded MetricResult
        """
        return MetricResult.load(path)
    
    def load_all_results(
        self, 
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> List[MetricResult]:
        """
        Load all results from the results directory.
        
        Args:
            model_name: Filter by model name (optional)
            model_version: Filter by model version (optional)
            
        Returns:
            List of loaded MetricResults
        """
        results_dir = self.output_dir / "results"
        if not results_dir.exists():
            logger.warning(f"Results directory does not exist: {results_dir}")
            return []
        
        # Construct glob pattern based on filters
        if model_name and model_version:
            pattern = f"{model_name}_{model_version}_*.json"
        elif model_name:
            pattern = f"{model_name}_*.json"
        else:
            pattern = "*.json"
        
        results = []
        for file_path in results_dir.glob(pattern):
            try:
                result = self.load_result(file_path)
                
                # Apply additional filtering if needed
                if (model_name is None or result.model_name == model_name) and \
                   (model_version is None or result.model_version == model_version):
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error loading result from {file_path}: {e}")
        
        return results


# Singleton instance
_evaluator_instance = None

def get_evaluator(output_dir: Optional[Union[str, Path]] = None) -> ModelEvaluator:
    """
    Get the global model evaluator instance.
    
    Args:
        output_dir: Output directory (only used on first call)
        
    Returns:
        ModelEvaluator: Singleton instance of ModelEvaluator
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = ModelEvaluator(output_dir)
    return _evaluator_instance 