#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Isolation Forest Performance

This script evaluates a trained Isolation Forest model by loading the model,
predicting anomaly scores on latent vectors, and computing metrics such as 
AUC, F1 score, and precision-recall AUC if ground truth labels are available.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants and default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LATENTS_PATH = PROJECT_ROOT / "data/processed/latents/latents.npy"
DEFAULT_MODEL_PATH = Path("ai_module/models/isolation_forest.pkl")
DEFAULT_LABELS_PATH = None  # No default labels path since y_train.npy doesn't exist
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/"


class IsolationForestEvaluator:
    """Class to evaluate Isolation Forest model performance."""
    
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        latents_path=DEFAULT_LATENTS_PATH,
        labels_path=DEFAULT_LABELS_PATH,
        output_dir=DEFAULT_OUTPUT_DIR
    ):
        """
        Initialize the Isolation Forest evaluator.
        
        Args:
            model_path: Path to the Isolation Forest model
            latents_path: Path to the latent vectors
            labels_path: Path to ground truth labels (optional)
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.latents_path = Path(latents_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data and model attributes
        self.latents = None
        self.labels = None
        self.model = None
        self.anomaly_scores = None
        self.predictions = None
        self.metrics = {}
    
    def load_data(self):
        """
        Load latent vectors and labels if available.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Load latent vectors
            if self.latents_path.exists():
                self.latents = np.load(self.latents_path)
                logger.info(f"Loaded latent vectors with shape {self.latents.shape}")
            else:
                logger.error(f"Latents file not found: {self.latents_path}")
                return False
            
            # Load labels if available
            if self.labels_path and self.labels_path.exists():
                self.labels = np.load(self.labels_path)
                logger.info(f"Loaded labels with shape {self.labels.shape}")
                
                # Validate shapes
                if len(self.labels) != len(self.latents):
                    logger.warning(
                        f"Shape mismatch: latents={len(self.latents)}, "
                        f"labels={len(self.labels)}"
                    )
            else:
                logger.warning(f"No labels provided or labels file not found.")
                self.labels = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def load_model(self):
        """
        Load the trained Isolation Forest model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_anomaly_scores(self):
        """
        Predict anomaly scores using the loaded model.
        
        Returns:
            bool: True if prediction successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return False
            
            if self.latents is None:
                logger.error("Latent vectors not loaded")
                return False
            
            # Predict anomaly scores
            # For Isolation Forest, decision_function returns the anomaly score
            # Higher scores mean more normal (less anomalous)
            # We negate it so that higher scores mean more anomalous
            self.anomaly_scores = -self.model.decision_function(self.latents)
            
            logger.info(f"Predicted anomaly scores with min={np.min(self.anomaly_scores)}, "
                       f"max={np.max(self.anomaly_scores)}, "
                       f"mean={np.mean(self.anomaly_scores)}")
            
            # Save scores to file
            scores_dir = PROJECT_ROOT / "data/processed/scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            scores_path = scores_dir / "isolation_forest_scores.npy"
            np.save(scores_path, self.anomaly_scores)
            logger.info(f"Saved anomaly scores to {scores_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error predicting anomaly scores: {e}")
            return False
    
    def optimize_threshold(self):
        """
        Optimize threshold for binary classification using F1 score.
        
        Returns:
            float: Optimal threshold
        """
        if self.labels is None:
            logger.warning("No labels available, using default threshold")
            threshold = 0.0  # Default threshold for Isolation Forest
            self.predictions = (self.anomaly_scores > threshold).astype(int)
            return threshold
        
        if self.anomaly_scores is None:
            logger.error("No anomaly scores available")
            return None
        
        try:
            # Find optimal F1 threshold
            f1_scores = []
            thresholds = np.linspace(np.min(self.anomaly_scores), np.max(self.anomaly_scores), num=100)
            
            for threshold in thresholds:
                predictions = (self.anomaly_scores >= threshold).astype(int)
                f1 = f1_score(self.labels, predictions)
                f1_scores.append(f1)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Generate predictions using optimal threshold
            self.predictions = (self.anomaly_scores >= optimal_threshold).astype(int)
            
            logger.info(f"Optimized threshold: {optimal_threshold:.4f} with F1: {f1_scores[optimal_idx]:.4f}")
            
            return optimal_threshold
            
        except Exception as e:
            logger.error(f"Error optimizing threshold: {e}")
            return None
    
    def calculate_metrics(self):
        """
        Calculate evaluation metrics if ground truth labels are available.
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        if self.labels is None:
            logger.warning("No labels available, skipping metrics calculation")
            return {}
        
        if self.anomaly_scores is None:
            logger.warning("No anomaly scores available, skipping metrics calculation")
            return {}
        
        try:
            # Calculate ROC AUC
            roc_auc = roc_auc_score(self.labels, self.anomaly_scores)
            
            # Calculate Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(self.labels, self.anomaly_scores)
            pr_auc = auc(recall, precision)
            
            # Optimize threshold and get predictions
            optimal_threshold = self.optimize_threshold()
            
            # Calculate precision and recall at optimal threshold
            precision = precision_score(self.labels, self.predictions)
            recall = recall_score(self.labels, self.predictions)
            max_f1 = f1_score(self.labels, self.predictions)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(self.labels, self.predictions).ravel()
            
            # Store metrics
            self.metrics = {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "f1": float(max_f1),
                "optimal_threshold": float(optimal_threshold),
                "precision": float(precision),
                "recall": float(recall),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }
            
            logger.info(f"Calculated metrics: ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}, F1={max_f1:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix if labels and predictions are available.
        
        Returns:
            str: Path to saved plot
        """
        if self.labels is None or self.predictions is None:
            logger.warning("Labels or predictions not available, skipping confusion matrix plot")
            return None
        
        try:
            plt.figure(figsize=(8, 6))
            
            cm = confusion_matrix(self.labels, self.predictions)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=["Normal", "Anomaly"],
                       yticklabels=["Normal", "Anomaly"])
            
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Isolation Forest Confusion Matrix")
            
            # Save plot
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / "isolation_forest_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved confusion matrix plot to {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            return None
    
    def plot_anomaly_score_distribution(self):
        """
        Plot distribution of anomaly scores.
        
        Returns:
            str: Path to saved plot
        """
        if self.anomaly_scores is None:
            logger.warning("No anomaly scores available, skipping distribution plot")
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot distributions
            if self.labels is not None:
                # Separate normal and anomaly scores
                normal_scores = self.anomaly_scores[self.labels == 0]
                anomaly_scores = self.anomaly_scores[self.labels == 1]
                
                # Plot distributions
                sns.histplot(normal_scores, color="green", label="Normal", alpha=0.6, 
                           kde=True, bins=30)
                sns.histplot(anomaly_scores, color="red", label="Anomaly", alpha=0.6,
                           kde=True, bins=30)
                
                plt.legend()
            else:
                # Plot overall distribution
                sns.histplot(self.anomaly_scores, kde=True, bins=50)
            
            # Add optimal threshold if available
            if self.metrics and "optimal_threshold" in self.metrics:
                plt.axvline(
                    self.metrics["optimal_threshold"],
                    color="black",
                    linestyle="--",
                    label=f"Threshold: {self.metrics['optimal_threshold']:.4f}"
                )
                plt.legend()
            
            plt.xlabel("Anomaly Score")
            plt.ylabel("Count")
            plt.title("Distribution of Isolation Forest Anomaly Scores")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / "isolation_forest_score_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved anomaly score distribution plot to {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting anomaly score distribution: {e}")
            return None
    
    def generate_classification_report(self):
        """
        Generate classification report if labels and predictions are available.
        
        Returns:
            str: Classification report
        """
        if self.labels is None or self.predictions is None:
            logger.warning("Labels or predictions not available, skipping classification report")
            return None
        
        try:
            # Generate classification report
            report = classification_report(
                self.labels,
                self.predictions,
                target_names=["Normal", "Anomaly"],
                output_dict=True
            )
            
            # Save report to file
            report_path = self.output_dir / "isolation_forest_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Saved classification report to {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return None
    
    def save_metrics(self):
        """
        Save calculated metrics to JSON file.
        
        Returns:
            str: Path to saved metrics file
        """
        if not self.metrics:
            logger.warning("No metrics available to save")
            return None
        
        try:
            # Save metrics to JSON
            metrics_path = self.output_dir / "isolation_forest_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logger.info(f"Saved metrics to {metrics_path}")
            
            return str(metrics_path)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return None
    
    def run_evaluation(self):
        """
        Run full evaluation pipeline.
        
        Returns:
            dict: Results including metrics and plot paths
        """
        results = {
            "success": False,
            "metrics": {},
            "plots": []
        }
        
        # Load data and model
        if not self.load_data():
            return results
        
        if not self.load_model():
            return results
        
        # Predict anomaly scores
        if not self.predict_anomaly_scores():
            return results
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        results["metrics"] = metrics
        
        # Generate plots
        confusion_matrix_path = self.plot_confusion_matrix()
        if confusion_matrix_path:
            results["plots"].append(confusion_matrix_path)
        
        distribution_plot_path = self.plot_anomaly_score_distribution()
        if distribution_plot_path:
            results["plots"].append(distribution_plot_path)
        
        # Generate classification report
        classification_report = self.generate_classification_report()
        if classification_report:
            results["classification_report"] = classification_report
        
        # Save metrics
        metrics_path = self.save_metrics()
        if metrics_path:
            results["metrics_path"] = metrics_path
        
        # Mark as successful
        results["success"] = True
        
        return results


def main():
    """Main entry point for Isolation Forest evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the Isolation Forest model"
    )
    parser.add_argument(
        "--latents-path",
        type=str,
        default=str(DEFAULT_LATENTS_PATH),
        help="Path to the latent vectors"
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        help="Path to ground truth labels (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run evaluator
    evaluator = IsolationForestEvaluator(
        model_path=args.model_path,
        latents_path=args.latents_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_evaluation()
    
    # Print summary
    if results["success"]:
        print("\n=== Isolation Forest Evaluation Summary ===")
        
        if results["metrics"]:
            print("\nMetrics:")
            for name, value in results["metrics"].items():
                print(f"  {name}: {value}")
        else:
            print("\nNo metrics available (no labels provided)")
        
        if "plots" in results and results["plots"]:
            print("\nGenerated plots:")
            for plot_path in results["plots"]:
                print(f"  {plot_path}")
            
        if "metrics_path" in results:
            print(f"\nDetailed metrics saved to: {results['metrics_path']}")
        
        print("\nEvaluation completed successfully.")
    else:
        print("\nEvaluation failed. Check logs for details.")


if __name__ == "__main__":
    main() 