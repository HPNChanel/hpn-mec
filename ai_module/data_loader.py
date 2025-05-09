#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module for HPN Medicare AI Module

This module provides a unified interface to access data loading functionality.
It redirects to the implementation in preprocess/data_loader.py.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the actual implementation
try:
    from ai_module.preprocess.data_loader import DataLoader, COLUMN_MAPPING, IMPORTANT_FEATURES, TARGET_COLUMNS
    logger.info("Successfully imported DataLoader from preprocess module")
except ImportError as e:
    logger.error(f"Error importing DataLoader: {e}")
    
    # Define a fallback DataLoader class if import fails
    class DataLoader:
        """
        Fallback DataLoader class that logs an error when used.
        """
        def __init__(self, *args, **kwargs):
            logger.error("Using fallback DataLoader. The actual implementation could not be imported.")
            
        def load_data(self, *args, **kwargs):
            logger.error("DataLoader implementation is missing. Cannot load data.")
            return None
        
    # Define fallback constants
    COLUMN_MAPPING = {}
    IMPORTANT_FEATURES = []
    TARGET_COLUMNS = {}

# Export the same interface as the original module
__all__ = ['DataLoader', 'COLUMN_MAPPING', 'IMPORTANT_FEATURES', 'TARGET_COLUMNS'] 