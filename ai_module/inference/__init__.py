#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Module for HPN Medicare AI System

This module provides inference utilities for making predictions with trained models.

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
    from ai_module.inference.inference_api import generate_predictions, run_api, API_PORT
    
    __all__ = ['generate_predictions', 'run_api', 'API_PORT']
except ImportError as e:
    logger.warning(f"Inference components could not be imported: {e}")
    __all__ = [] 