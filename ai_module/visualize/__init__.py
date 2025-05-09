#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for HPN Medicare AI System

This module provides visualization utilities for model results and analyses.

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
    from ai_module.visualize.visualize_anomaly_scores import plot_anomaly_histogram, plot_reconstruction_error, compare_anomaly_scores
    __all__ = ['plot_anomaly_histogram', 'plot_reconstruction_error', 'compare_anomaly_scores']
except ImportError:
    logger.warning("Visualization functions could not be imported")
    __all__ = [] 