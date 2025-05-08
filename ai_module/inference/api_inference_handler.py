# TODO: Implement this module
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Inference Handler for Health Anomaly Detection

This module implements a FastAPI-based API for real-time anomaly detection
in healthcare data, using a combination of AutoEncoder for latent representation
and Isolation Forest for anomaly detection.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure the parent directory is in path so relative imports work
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import model handlers
try:
    from .handlers import autoencoder_handler
    from .handlers import isolation_forest_handler
    from .handlers import shap_handler
    logger.info("Successfully imported handlers using relative imports")
except ImportError as e:
    # Fallback to absolute imports if needed
    try:
        from ai_module.inference.handlers import autoencoder_handler
        from ai_module.inference.handlers import isolation_forest_handler
        from ai_module.inference.handlers import shap_handler
        logger.info("Successfully imported handlers using absolute imports")
    except ImportError as e:
        logger.error(f"Error importing handlers: {e}")
        logger.error("Make sure the handler modules are in the correct location")
        sys.exit(1)

# Define Pydantic models for API
class HealthInput(BaseModel):
    """
    Input model for health data.
    
    Includes all potential health metrics that can be used for anomaly detection.
    All fields are optional, but the system expects most fields to be provided.
    """
    # Demographics
    age: Optional[float] = Field(None, description="Age in years")
    gender: Optional[int] = Field(None, description="Gender (0=female, 1=male)")
    
    # Vital signs
    systolic_bp: Optional[float] = Field(None, description="Systolic blood pressure")
    diastolic_bp: Optional[float] = Field(None, description="Diastolic blood pressure")
    heart_rate: Optional[float] = Field(None, description="Heart rate (bpm)")
    
    # Physical measurements
    height: Optional[float] = Field(None, description="Height in cm")
    weight: Optional[float] = Field(None, description="Weight in kg")
    bmi: Optional[float] = Field(None, description="Body Mass Index")
    
    # Lab values
    glucose: Optional[float] = Field(None, description="Blood glucose level")
    cholesterol_level: Optional[float] = Field(None, description="Cholesterol level")
    total_cholesterol: Optional[float] = Field(None, description="Total cholesterol")
    high_cholesterol: Optional[int] = Field(None, description="High cholesterol flag (0/1)")
    
    # Risk factors
    smoker: Optional[int] = Field(None, description="Smoking status (0/1)")
    cigarettes_per_day: Optional[float] = Field(None, description="Cigarettes per day")
    alcohol: Optional[int] = Field(None, description="Alcohol consumption flag (0/1)")
    physically_active: Optional[int] = Field(None, description="Physical activity flag (0/1)")
    
    # Medical history
    stroke_history: Optional[int] = Field(None, description="History of stroke (0/1)")
    hypertension_history: Optional[int] = Field(None, description="History of hypertension (0/1)")
    
    # Other
    education_level: Optional[int] = Field(None, description="Education level (coded)")
    difficulty_walking: Optional[int] = Field(None, description="Difficulty walking flag (0/1)")
    general_health: Optional[int] = Field(None, description="General health rating (coded)")
    
    # Additional fields can be added here as needed
    
    @validator('*')
    def check_numeric_fields(cls, v, values, **kwargs):
        """Validate that all provided fields are numeric"""
        if v is not None and not isinstance(v, (int, float)):
            try:
                return float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Value must be numeric, got {type(v)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "bmi": 27.5,
                "systolic_bp": 130,
                "diastolic_bp": 85,
                "glucose": 105,
                "cholesterol_level": 2,
                "physically_active": 1,
                "smoker": 0
            }
        }

class AnomalyPrediction(BaseModel):
    """Output model for anomaly prediction results."""
    anomaly: int = Field(..., description="Anomaly flag (0=normal, 1=anomaly)")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    reconstruction_error: float = Field(..., description="Reconstruction error from AutoEncoder")
    shap_top_features: Optional[List[str]] = Field(None, description="Top features contributing to anomaly")
    latent_vector: Optional[List[float]] = Field(None, description="Latent vector from AutoEncoder (debug only)")

class PredictionResponse(BaseModel):
    """Response model for the prediction API."""
    status: str = "success"
    predictions: List[AnomalyPrediction]
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information (debug only)")

# Create FastAPI app
app = FastAPI(
    title="Health Anomaly Detection API",
    description="API for detecting anomalies in healthcare data using AutoEncoder and Isolation Forest",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
autoencoder = None
isolation_forest = None
shap_explainer = None
feature_names = None

def load_models():
    """
    Load all required models at startup.
    
    Returns:
        tuple: (autoencoder, isolation_forest, shap_explainer, feature_names)
    """
    global autoencoder, isolation_forest, shap_explainer, feature_names
    
    try:
        # Load AutoEncoder
        autoencoder = autoencoder_handler.load_model()
        logger.info("AutoEncoder loaded successfully")
        
        # Load Isolation Forest
        isolation_forest = isolation_forest_handler.load_model()
        logger.info("Isolation Forest loaded successfully")
        
        # Load SHAP explainer
        shap_explainer = shap_handler.load_explainer()
        logger.info("SHAP explainer loaded successfully")
        
        # Load feature names
        feature_names = load_feature_names()
        logger.info(f"Loaded {len(feature_names)} feature names")
        
        return autoencoder, isolation_forest, shap_explainer, feature_names
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")

def load_feature_names():
    """
    Load feature names for SHAP explanations.
    
    Returns:
        list: Feature names
    """
    try:
        # Try to load from file using Path
        features_path = Path(__file__).parent.parent / "data" / "processed" / "features" / "feature_names.txt"
        
        # Try alternative paths if main path doesn't exist
        if not features_path.exists():
            logger.warning(f"Feature names not found at {features_path}")
            
            # Try in the parent directory
            alt_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features" / "feature_names.txt"
            if alt_path.exists():
                features_path = alt_path
                logger.info(f"Using alternative feature names path: {features_path}")
            else:
                # Fallback to default feature names
                logger.warning("Using default feature names")
                return [
                    "age", "gender", "bmi", "systolic_bp", "diastolic_bp", 
                    "glucose", "cholesterol_level", "smoker", "physically_active"
                ]
        
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        logger.info(f"Loaded {len(feature_names)} feature names from {features_path}")
        return feature_names
    
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        # Return default feature names on error
        return [f"feature_{i}" for i in range(10)]

def preprocess_input(input_data: Union[HealthInput, List[HealthInput]]) -> np.ndarray:
    """
    Preprocess input data to match model expectations.
    
    Args:
        input_data: Input health data (single patient or list)
        
    Returns:
        np.ndarray: Preprocessed features ready for the model
    """
    # Convert to list if single input
    if isinstance(input_data, HealthInput):
        inputs = [input_data]
    else:
        inputs = input_data
    
    # Convert inputs to feature matrix
    features = []
    for patient in inputs:
        # Convert patient dict to list of values in expected order
        patient_dict = patient.dict()
        
        # Use the feature names to ensure correct order
        # For missing values, use 0 (models should handle this appropriately)
        patient_features = [patient_dict.get(name, 0) for name in feature_names]
        features.append(patient_features)
    
    return np.array(features, dtype=np.float32)

def generate_predictions(
    features: np.ndarray,
    enable_shap: bool = False,
    debug: bool = False
) -> List[AnomalyPrediction]:
    """
    Generate anomaly predictions for input features.
    
    Args:
        features: Preprocessed feature matrix
        enable_shap: Whether to include SHAP explanations
        debug: Whether to include debug information
        
    Returns:
        list: List of AnomalyPrediction objects
    """
    # Ensure models are loaded
    if None in (autoencoder, isolation_forest):
        raise RuntimeError("Models not loaded. Please restart the server.")
    
    # Get latent vectors and reconstruction errors
    latent_vectors, reconstructions, recon_errors = autoencoder_handler.encode_and_reconstruct(
        autoencoder, features
    )
    
    # Get anomaly predictions and scores
    anomaly_labels, anomaly_scores = isolation_forest_handler.predict_anomalies(
        isolation_forest, latent_vectors
    )
    
    # Generate SHAP explanations if requested
    shap_explanations = None
    if enable_shap and shap_explainer is not None:
        shap_explanations = shap_handler.explain_predictions(
            shap_explainer, latent_vectors, feature_names
        )
    
    # Create prediction objects
    predictions = []
    for i in range(len(features)):
        # Get top SHAP features if available
        top_features = None
        if shap_explanations is not None:
            top_features = shap_explanations[i]
        
        # Create prediction object
        prediction = AnomalyPrediction(
            anomaly=int(anomaly_labels[i]),
            anomaly_score=float(anomaly_scores[i]),
            reconstruction_error=float(recon_errors[i]),
            shap_top_features=top_features,
            # Include latent vector only in debug mode
            latent_vector=latent_vectors[i].tolist() if debug else None
        )
        
        predictions.append(prediction)
    
    return predictions

@app.on_event("startup")
async def startup_event():
    """Load models and resources on startup."""
    try:
        load_models()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # We don't exit here as FastAPI will still run and show errors on requests

@app.get("/healthcheck")
async def healthcheck():
    """
    Health check endpoint to verify API is running correctly.
    
    Returns:
        dict: Status and version information
    """
    # Check if models are loaded
    models_loaded = all(model is not None for model in [autoencoder, isolation_forest])
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    input_data: Union[HealthInput, List[HealthInput]],
    enable_shap: bool = Query(False, description="Enable SHAP explanations"),
    debug: bool = Query(False, description="Include debug information")
):
    """
    Generate anomaly predictions for health input data.
    
    Args:
        input_data: Health data input (single patient or list)
        enable_shap: Whether to include SHAP explanations
        debug: Whether to include debug information
        
    Returns:
        PredictionResponse: Prediction results
    """
    start_time = time.time()
    
    try:
        # Preprocess input data
        features = preprocess_input(input_data)
        
        # Generate predictions
        predictions = generate_predictions(features, enable_shap, debug)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            status="success",
            predictions=predictions,
            processing_time=processing_time,
            # Include model info only in debug mode
            model_info={
                "autoencoder_type": type(autoencoder).__name__,
                "isolation_forest_params": isolation_forest.get_params() if debug else None,
                "feature_names": feature_names if debug else None
            } if debug else None
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e),
                "processing_time": processing_time
            }
        )

@app.post("/batch-predict")
async def batch_predict(
    input_data: List[HealthInput],
    background_tasks: BackgroundTasks,
    enable_shap: bool = Query(False, description="Enable SHAP explanations"),
    debug: bool = Query(False, description="Include debug information")
):
    """
    Process batch predictions asynchronously for large datasets.
    
    For large batches, this endpoint returns immediately and processes
    the predictions in the background. The results can then be retrieved
    separately.
    
    Args:
        input_data: List of health data inputs
        background_tasks: FastAPI background tasks
        enable_shap: Whether to include SHAP explanations
        debug: Whether to include debug information
        
    Returns:
        dict: Job information
    """
    # For large batches, process asynchronously
    if len(input_data) > 100:
        job_id = f"batch_{int(time.time())}"
        
        # Add to background tasks
        background_tasks.add_task(
            process_batch, 
            input_data, 
            job_id, 
            enable_shap, 
            debug
        )
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": f"Processing {len(input_data)} records in the background"
        }
    
    # For smaller batches, process immediately
    return await predict(input_data, enable_shap, debug)

async def process_batch(
    input_data: List[HealthInput],
    job_id: str,
    enable_shap: bool,
    debug: bool
):
    """
    Process a batch of predictions in the background.
    
    Args:
        input_data: List of health data inputs
        job_id: Unique job identifier
        enable_shap: Whether to include SHAP explanations
        debug: Whether to include debug information
    """
    try:
        # Preprocess input data
        features = preprocess_input(input_data)
        
        # Generate predictions
        predictions = generate_predictions(features, enable_shap, debug)
        
        # Save results for later retrieval
        # This would typically be stored in a database or cache
        # For simplicity, we'll just log the completion
        logger.info(f"Batch job {job_id} completed with {len(predictions)} predictions")
        
    except Exception as e:
        logger.error(f"Error in batch prediction job {job_id}: {e}")

# Main entry point for running the API with uvicorn
if __name__ == "__main__":
    # Ensure the parent directory is in path
    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Health Anomaly Detection API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    
    args = parser.parse_args()
    
    # Initialize module path if needed
    module_path = f"{Path(__file__).parent.name}.{Path(__file__).stem}"
    
    # Run with uvicorn
    uvicorn.run(
        f"{module_path}:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )