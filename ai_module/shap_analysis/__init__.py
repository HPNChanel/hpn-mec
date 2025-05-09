#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis Module for HPN Medicare AI System

This module provides explainable AI utilities using SHAP
(SHapley Additive exPlanations) to interpret model decisions.

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
    from ai_module.shap_analysis.shap_handler import ShapHandler
    __all__ = ['ShapHandler']
except ImportError:
    logger.warning("SHAP Handler could not be imported")
    __all__ = [] 