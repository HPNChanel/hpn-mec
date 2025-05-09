#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Module for HPN Medicare AI System

This module provides utilities for evaluating model performance.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import key components to expose at module level
try:
    from ai_module.evaluate.evaluate_autoencoder import evaluate_autoencoder, calculate_reconstruction_error
    from ai_module.evaluate.evaluate_isolation_forest import evaluate_isolation_forest, calculate_anomaly_scores
    
    __all__ = [
        'evaluate_autoencoder', 
        'calculate_reconstruction_error',
        'evaluate_isolation_forest',
        'calculate_anomaly_scores'
    ]
except ImportError as e:
    logger.warning(f"Evaluation functions could not be imported: {e}")
    __all__ = [] 