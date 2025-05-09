# HPN Medicare AI Module

This module contains the AI and machine learning components for anomaly detection in health data for the HPN Medicare system.

## Structure

The module is organized into several submodules:

- `data_loader.py`: Central entry point for data loading and preprocessing
- `preprocess/`: Contains data preprocessing utilities
- `train/`: Scripts for training various models
- `evaluate/`: Tools for evaluating model performance
- `inference/`: API and handlers for model inference
- `models/`: Model definitions and architecture
- `visualize/`: Visualization tools for results
- `shap_analysis/`: Explainable AI utilities using SHAP
- `utils/`: Common utility functions

## Data Directory Structure

- `data/raw/`: Raw input data (CSV files)
- `data/processed/`: Processed data outputs
  - `features/`: Preprocessed feature matrices
  - `labels/`: Target labels if available
  - `combined/`: Combined datasets
  - `latents/`: Latent vectors from autoencoder
  - `scores/`: Anomaly scores

## Workflow

### 1. Data Preprocessing

```bash
# Load and preprocess data
python -m ai_module.preprocess.data_loader
```

### 2. Training Models

```bash
# Train the autoencoder model
python -m ai_module.train.train_autoencoder --epochs 50 --batch-size 64

# Train the Isolation Forest on latent vectors
python -m ai_module.train.train_isolation_forest --n-estimators 100
```

### 3. Evaluating Models

```bash
# Evaluate autoencoder performance
python -m ai_module.evaluate.evaluate_autoencoder

# Evaluate isolation forest performance
python -m ai_module.evaluate.evaluate_isolation_forest
```

### 4. Visualizing Results

```bash
# Visualize anomaly scores from both models
python -m ai_module.visualize.visualize_anomaly_scores --model-type combined
```

### 5. SHAP Analysis

```bash
# Run SHAP analysis on autoencoder
python -m ai_module.shap_analysis.shap_handler --model-type autoencoder
```

### 6. Running the API

```bash
# Start the inference API
uvicorn ai_module.inference.api_inference_handler:app --reload
```

## Using the API

Once the API is running, you can access:

- `/docs`: Swagger documentation
- `/healthcheck`: API health status
- `/predict`: Make predictions on new data

## Development

### Requirements

See `requirements.txt` for the required Python packages.

### Running Tests

```bash
# Run all tests
python -m unittest discover ai_module/tests

# Run specific test
python -m unittest ai_module.tests.test_data_loader
```

## Python Module Imports

All imports should follow the pattern:

```python
from ai_module.submodule import component
```

### Paths and File Handling

Always use `pathlib.Path` for file path handling, not `os.path`.

```python
from pathlib import Path

# Good
data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# Bad
data_dir = os.path.join("data", "processed")
os.makedirs(data_dir, exist_ok=True)
```

## Logging

Use the built-in logging instead of print statements:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log messages
logger.info("Processing data...")
logger.warning("Missing values detected")
logger.error("Failed to load model")
```

## Usage

### Data Loading

All data loading should use the central DataLoader class:

```python
from ai_module.data_loader import DataLoader

# Initialize with default paths
data_loader = DataLoader()

# Load data for supervised learning (features and labels)
X, y = data_loader.load_data(supervised=True, normalize=True)

# Load data for unsupervised learning (features only)
X = data_loader.load_data(supervised=False, normalize=True)
```

### Training Models

Each model has its own training script in the `train/` directory:

```bash
# Train the autoencoder
python -m ai_module.train.train_autoencoder --epochs 50 --batch-size 64

# Train the Isolation Forest
python -m ai_module.train.train_isolation_forest --n-estimators 100
```

### Inference

To use trained models for inference:

```python
from ai_module.inference.api_inference_handler import generate_predictions

# Make predictions for new data
predictions = generate_predictions(features, enable_shap=True)
```

## Models Implemented

- **AutoEncoder**: Neural network for dimensionality reduction and anomaly detection
- **Isolation Forest**: Unsupervised anomaly detection model
- **XGBoost**: For risk prediction (placeholder)
- **LSTM**: For time series analysis (placeholder)

## Directory Conventions

- Model checkpoints: `ai_module/models/`
- Processed data: `data/processed/`
- Raw data: `data/raw/`
- Latent vectors: `data/processed/latents/`
- Anomaly scores: `data/processed/scores/`
