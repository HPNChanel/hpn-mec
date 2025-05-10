#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Configuration System for AI Module

This module implements a singleton configuration system that provides
a unified way to access configuration values from YAML files and environment
variables across the entire AI module.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass

class Config:
    """
    Singleton configuration class for accessing and managing configurations.
    
    This class reads configurations from YAML files and environment variables,
    providing a unified interface to access them across the application.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration system if not already initialized."""
        if self._initialized:
            return
            
        # Initialize configuration dict
        self._config = {}
        
        # Default paths
        self._config_dir = Path(__file__).parent.parent / "config"
        self._config_file = self._config_dir / "config.yaml"
        
        # Load configuration immediately
        self.reload()
        
        self._initialized = True
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        # Clear existing configuration
        self._config = {}
        
        # Load from default config file
        self._load_from_yaml(self._config_file)
        
        # Check for environment-specific config file
        env = os.environ.get("AI_ENV", "development")
        env_config_file = self._config_dir / f"config.{env}.yaml"
        
        if env_config_file.exists():
            self._load_from_yaml(env_config_file, override=True)
            logger.info(f"Loaded environment-specific config from {env_config_file}")
        
        # Override with environment variables
        self._load_from_env()
        
        logger.info(f"Configuration loaded with {len(self._config)} top-level keys")
    
    def _load_from_yaml(self, file_path: Path, override: bool = False) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            override: Whether to override existing values
        """
        try:
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return
                
            with open(file_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                
            if not yaml_config:
                logger.warning(f"Empty or invalid config file: {file_path}")
                return
                
            # Update configuration
            if override:
                self._update_recursive(self._config, yaml_config)
            else:
                self._config.update(yaml_config)
                
            logger.info(f"Loaded configuration from {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format:
        AI_MODULE_SECTION_KEY=value
        """
        prefix = "AI_MODULE_"
        
        # Find all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split into sections
                config_path = key[len(prefix):].lower().split('_')
                
                # Update nested configuration
                self._set_nested_config(self._config, config_path, value)
                
        logger.info("Loaded configuration from environment variables")
    
    def _set_nested_config(self, config_dict: Dict, path: List[str], value: Any) -> None:
        """
        Set a value in a nested dictionary using a path of keys.
        
        Args:
            config_dict: The dictionary to update
            path: List of keys forming the path to the value
            value: The value to set
        """
        if len(path) == 1:
            # Try to convert value to appropriate type
            config_dict[path[0]] = self._convert_value(value)
        else:
            # Create nested dictionaries if they don't exist
            if path[0] not in config_dict:
                config_dict[path[0]] = {}
            
            # Recursively set value in nested dict
            self._set_nested_config(config_dict[path[0]], path[1:], value)
    
    def _update_recursive(self, base_dict: Dict, new_dict: Dict) -> None:
        """
        Recursively update a dictionary with values from another dictionary.
        
        Args:
            base_dict: The dictionary to update
            new_dict: The dictionary with new values
        """
        for key, value in new_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_recursive(base_dict[key], value)
            else:
                # Set or override value
                base_dict[key] = value
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Boolean conversion
        if value.lower() in ('true', 'yes', 'y', '1'):
            return True
        if value.lower() in ('false', 'no', 'n', '0'):
            return False
            
        # Numeric conversion
        try:
            # Try integer first
            return int(value)
        except ValueError:
            try:
                # Then try float
                return float(value)
            except ValueError:
                # If not numeric, return as string
                return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key (can be nested with dots, e.g., 'model.autoencoder.latent_dim')
            default: Default value to return if key not found
            
        Returns:
            The configuration value or default
        """
        # Split the key into parts
        parts = key.split('.')
        
        # Start with the whole config
        current = self._config
        
        # Traverse the nested dictionary
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key (can be nested with dots)
            value: The value to set
        """
        # Split the key into parts
        parts = key.split('.')
        
        # Set value in nested dictionary
        self._set_nested_config(self._config, parts, value)
    
    @property
    def config(self) -> Dict:
        """Get a copy of the complete configuration dictionary."""
        return self._config.copy()


# Helper function to get Config instance
@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the singleton Config instance."""
    return Config() 