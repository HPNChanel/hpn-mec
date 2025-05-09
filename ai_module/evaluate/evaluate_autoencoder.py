#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AutoEncoder Performance

This script evaluates a trained autoencoder model by loading latent vectors,
calculating reconstruction error, and computing metrics such as AUC, F1 score,
and precision-recall AUC if ground truth labels are available.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, confusion_matrix
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
DEFAULT_MODEL_PATH = Path("ai_module/models/autoencoder.pt")
DEFAULT_LABELS_PATH = None  # No default labels path since y_train.npy doesn't exist
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/"


class AutoencoderEvaluator:
    """Class to evaluate autoencoder model performance."""
    
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        latents_path=DEFAULT_LATENTS_PATH,
        labels_path=DEFAULT_LABELS_PATH,
        output_dir=DEFAULT_OUTPUT_DIR
    ):
        """
        Initialize the autoencoder evaluator.
        
        Args:
            model_path: Path to the autoencoder model
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
        
        # Set device for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize data and model attributes
        self.latents = None
        self.original_data = None
        self.labels = None
        self.model = None
        self.reconstruction_errors = None
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
        Load the trained autoencoder model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Import model class here to avoid circular imports
            from ai_module.models.autoencoder import AutoEncoder
            
            # Load model metadata if available
            metadata_path = self.model_path.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    input_dim = metadata.get('input_dim', 22)  # Default fallback
                    latent_dim = metadata.get('latent_dim', 16)  # Default fallback
                    logger.info(f"Loaded model dimensions: input_dim={input_dim}, latent_dim={latent_dim}")
            else:
                logger.warning("Model metadata not found, using default dimensions")
                input_dim = 22  # Default fallback
                latent_dim = 16  # Default fallback
            
            # Initialize model with the determined dimensions
            self.model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
            
            # Load weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model = self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def calculate_reconstruction_error(self):
        """
        Calculate reconstruction error for the latent vectors.
        
        Returns:
            bool: True if calculation successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return False
            
            # Convert latents to torch tensor
            latents_tensor = torch.tensor(self.latents, dtype=torch.float32).to(self.device)
            
            # Get reconstructions from latent space
            with torch.no_grad():
                reconstructions = self.model.decoder(latents_tensor)
                
            # Load original data if available
            original_data_path = self.latents_path.parent.parent / "signals" / "X_train.npy"
            if original_data_path.exists():
                self.original_data = np.load(original_data_path)
                logger.info(f"Loaded original data with shape {self.original_data.shape}")
                
                # Convert to tensor
                original_tensor = torch.tensor(self.original_data, dtype=torch.float32).to(self.device)
                
                # Calculate MSE between original and reconstruction
                mse = torch.mean(torch.square(original_tensor - reconstructions), dim=1)
                self.reconstruction_errors = mse.cpu().numpy()
                
                logger.info(f"Calculated reconstruction errors with min={np.min(self.reconstruction_errors)}, "
                           f"max={np.max(self.reconstruction_errors)}, "
                           f"mean={np.mean(self.reconstruction_errors)}")
                
                return True
            else:
                logger.warning(f"Original data not found: {original_data_path}")
                # Use a placeholder for errors since we can't compute them
                self.reconstruction_errors = np.zeros(len(self.latents))
                logger.warning("Using placeholder reconstruction errors (zeros)")
                return False
            
        except Exception as e:
            logger.error(f"Error calculating reconstruction error: {e}")
            return False
    
    def calculate_metrics(self):
        """
        Calculate evaluation metrics if ground truth labels are available.
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        if self.labels is None:
            logger.warning("No labels available, skipping metrics calculation")
            return {}
        
        if self.reconstruction_errors is None:
            logger.warning("No reconstruction errors available, skipping metrics calculation")
            return {}
        
        try:
            # Use reconstruction error as anomaly score (higher = more anomalous)
            scores = self.reconstruction_errors
            
            # Calculate ROC AUC
            roc_auc = roc_auc_score(self.labels, scores)
            
            # Calculate Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(self.labels, scores)
            pr_auc = auc(recall, precision)
            
            # Find optimal F1 threshold
            f1_scores = []
            thresholds = np.linspace(np.min(scores), np.max(scores), num=100)
            
            for threshold in thresholds:
                predictions = (scores >= threshold).astype(int)
                f1 = f1_score(self.labels, predictions)
                f1_scores.append(f1)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            max_f1 = f1_scores[optimal_idx]
            
            # Get predictions using optimal threshold
            predictions = (scores >= optimal_threshold).astype(int)
            
            # Calculate precision and recall at optimal threshold
            precision = precision_score(self.labels, predictions)
            recall = recall_score(self.labels, predictions)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(self.labels, predictions).ravel()
            
            # Store metrics
            self.metrics = {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "f1_max": float(max_f1),
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
    
    def plot_reconstruction_error_histogram(self):
        """
        Plot histogram of reconstruction errors.
        
        Returns:
            str: Path to saved plot
        """
        if self.reconstruction_errors is None:
            logger.warning("No reconstruction errors available, skipping plot")
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot error histogram
            plt.hist(self.reconstruction_errors, bins=50, alpha=0.7)
            
            # Add vertical line for mean
            plt.axvline(
                np.mean(self.reconstruction_errors),
                color='r',
                linestyle='--',
                label=f"Mean: {np.mean(self.reconstruction_errors):.4f}"
            )
            
            # Add optimal threshold if metrics calculated
            if self.metrics and "optimal_threshold" in self.metrics:
                plt.axvline(
                    self.metrics["optimal_threshold"],
                    color='g',
                    linestyle='-',
                    label=f"Optimal Threshold: {self.metrics['optimal_threshold']:.4f}"
                )
            
            plt.xlabel("Reconstruction Error")
            plt.ylabel("Count")
            plt.title("Distribution of Reconstruction Errors")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / "reconstruction_error_histogram.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved reconstruction error histogram to {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting reconstruction error histogram: {e}")
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
            metrics_path = self.output_dir / "autoencoder_metrics.json"
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
        
        # Calculate reconstruction error
        if not self.calculate_reconstruction_error():
            # We can continue even if reconstruction error calculation fails
            logger.warning("Continuing evaluation with limited data")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        results["metrics"] = metrics
        
        # Generate plots
        plot_path = self.plot_reconstruction_error_histogram()
        if plot_path:
            results["plots"].append(plot_path)
        
        # Save metrics
        metrics_path = self.save_metrics()
        if metrics_path:
            results["metrics_path"] = metrics_path
        
        # Mark as successful
        results["success"] = True
        
        return results


def main():
    """Main entry point for autoencoder evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate autoencoder model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the autoencoder model"
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
    evaluator = AutoencoderEvaluator(
        model_path=args.model_path,
        latents_path=args.latents_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_evaluation()
    
    # Print summary
    if results["success"]:
        print("\n=== Autoencoder Evaluation Summary ===")
        
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