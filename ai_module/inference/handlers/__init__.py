#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Handlers for Inference

This module provides handlers for different model types, allowing for
consistent inference interfaces across different model implementations.

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

# Import model handlers
try:
    from ai_module.inference.handlers.autoencoder_handler import AutoencoderHandler
    from ai_module.inference.handlers.isolation_forest_handler import IsolationForestHandler
    
    __all__ = ['AutoencoderHandler', 'IsolationForestHandler']
except ImportError as e:
    logger.warning(f"Model handlers could not be imported: {e}")
    __all__ = [] 