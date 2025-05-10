#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPN Medicare AI Module

This module serves as the main entry point for the AI components of the HPN Medicare system.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import key components to make them available directly from the module


__all__ = ['DataLoader', 'AutoEncoder']
