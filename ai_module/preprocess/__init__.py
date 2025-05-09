#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module for HPN Medicare AI System

This module contains utilities for loading and preprocessing health data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import key components to expose them at module level
from ai_module.preprocess.data_loader import DataLoader, COLUMN_MAPPING, IMPORTANT_FEATURES, TARGET_COLUMNS

__all__ = ['DataLoader', 'COLUMN_MAPPING', 'IMPORTANT_FEATURES', 'TARGET_COLUMNS'] 