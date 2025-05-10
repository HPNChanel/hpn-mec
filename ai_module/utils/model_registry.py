#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Registry System for AI Module

This module implements a central model registry to simplify model loading,
version tracking, and standardized access. The registry provides a
unified way to register and retrieve models across the system.

Author: Huỳnh Phúc Nguyên
Created: June 2025
"""

import logging
from pathlib import Path
from typing import Dict, Type, Callable, Optional, Any, Union, TypeVar, List, Tuple

# Import from project
from ai_module.utils.config import get_config
from ai_module.utils.logging_config import get_logger
from ai_module.models.base_model import BaseModelWrapper

# Initialize logger
logger = get_logger("utils.model_registry")

# Get configuration
config = get_config()

# Type variable for models
ModelType = TypeVar('ModelType', bound=BaseModelWrapper)

class ModelRegistry:
    """
    Central registry for all models in the system.
    
    This class implements a singleton pattern to provide a global registry
    for model registration, lookup, and instantiation.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model registry if not already initialized."""
        if self._initialized:
            return
            
        # Dictionary of registered models with structure:
        # {model_name: {'class': ModelClass, 'versions': {version: model_path}}}
        self._registry = {}
        
        # Default model versions
        self._default_versions = {}
        
        self._initialized = True
        logger.info("Model registry initialized")
    
    def register(
        self, 
        name: str, 
        model_class: Type[ModelType],
        version: str = "latest",
        model_path: Optional[Union[str, Path]] = None,
        is_default: bool = False
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            name: Unique name for the model
            model_class: The model class (must subclass BaseModelWrapper)
            version: Version identifier (e.g., "1.0.0", "latest")
            model_path: Path to saved model weights/parameters
            is_default: Whether this should be the default version
        
        Raises:
            TypeError: If model_class is not a subclass of BaseModelWrapper
            ValueError: If model path is provided but doesn't exist
        """
        # Validate model class
        if not issubclass(model_class, BaseModelWrapper):
            raise TypeError(f"Model class must inherit from BaseModelWrapper, got {model_class.__name__}")
        
        # Convert model_path to Path if provided
        if model_path is not None:
            model_path = Path(model_path)
            # Validate model path exists
            if not model_path.exists() and not str(model_path).startswith("s3://"):
                logger.warning(f"Model path does not exist: {model_path}")
        
        # Initialize model entry if it doesn't exist
        if name not in self._registry:
            self._registry[name] = {
                'class': model_class,
                'versions': {}
            }
        
        # Register version
        self._registry[name]['versions'][version] = model_path
        
        # Set as default version if specified or if it's the first version
        if is_default or name not in self._default_versions:
            self._default_versions[name] = version
        
        logger.info(f"Registered model '{name}' version '{version}'")
    
    def register_decorator(
        self, 
        name: str, 
        version: str = "latest",
        model_path: Optional[Union[str, Path]] = None,
        is_default: bool = False
    ) -> Callable[[Type[ModelType]], Type[ModelType]]:
        """
        Decorator for registering models.
        
        Args:
            name: Unique name for the model
            version: Version identifier (e.g., "1.0.0", "latest")
            model_path: Path to saved model weights/parameters
            is_default: Whether this should be the default version
            
        Returns:
            Decorator function that registers the model
        
        Example:
            @registry.register_decorator("isolation_forest", version="1.0.0")
            class IsolationForestModel(SKLearnModelWrapper):
                ...
        """
        def decorator(model_class: Type[ModelType]) -> Type[ModelType]:
            self.register(name, model_class, version, model_path, is_default)
            return model_class
        return decorator
    
    def get_model_class(
        self, 
        name: str
    ) -> Type[BaseModelWrapper]:
        """
        Get the model class by name.
        
        Args:
            name: Name of the registered model
            
        Returns:
            The model class
            
        Raises:
            KeyError: If model name is not registered
        """
        if name not in self._registry:
            available_models = list(self._registry.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available models: {available_models}")
        
        return self._registry[name]['class']
    
    def get_model_path(
        self, 
        name: str, 
        version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get the path to a model's weights/parameters.
        
        Args:
            name: Name of the registered model
            version: Version to retrieve (default: use default version)
            
        Returns:
            Path to the model or None if no path registered
            
        Raises:
            KeyError: If model name or version is not registered
        """
        if name not in self._registry:
            available_models = list(self._registry.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available models: {available_models}")
        
        # Use default version if none specified
        if version is None:
            if name not in self._default_versions:
                raise KeyError(f"No default version set for model '{name}'")
            version = self._default_versions[name]
        
        # Check if version exists
        if version not in self._registry[name]['versions']:
            available_versions = list(self._registry[name]['versions'].keys())
            raise KeyError(f"Version '{version}' not found for model '{name}'. Available versions: {available_versions}")
        
        return self._registry[name]['versions'][version]
    
    def create_model(
        self, 
        name: str, 
        version: Optional[str] = None,
        load_weights: bool = True,
        **kwargs
    ) -> BaseModelWrapper:
        """
        Create and optionally load a model instance.
        
        Args:
            name: Name of the registered model
            version: Version to create (default: use default version)
            load_weights: Whether to load weights from the registered path
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Instantiated model
            
        Raises:
            KeyError: If model name or version is not registered
        """
        # Get model class
        model_class = self.get_model_class(name)
        
        # Get model path if loading weights
        model_path = None
        if load_weights:
            model_path = self.get_model_path(name, version)
        
        # Create model instance
        model = model_class(
            model_type=name,
            model_path=model_path,
            **kwargs
        )
        
        return model
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(self._registry.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """
        List all versions for a model.
        
        Args:
            name: Name of the registered model
            
        Returns:
            List of version strings
            
        Raises:
            KeyError: If model name is not registered
        """
        if name not in self._registry:
            available_models = list(self._registry.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available models: {available_models}")
        
        return list(self._registry[name]['versions'].keys())
    
    def get_default_version(self, name: str) -> str:
        """
        Get the default version for a model.
        
        Args:
            name: Name of the registered model
            
        Returns:
            Default version string
            
        Raises:
            KeyError: If model name is not registered or no default set
        """
        if name not in self._default_versions:
            raise KeyError(f"No default version set for model '{name}'")
        
        return self._default_versions[name]
    
    def set_default_version(self, name: str, version: str) -> None:
        """
        Set the default version for a model.
        
        Args:
            name: Name of the registered model
            version: Version to set as default
            
        Raises:
            KeyError: If model name or version is not registered
        """
        if name not in self._registry:
            available_models = list(self._registry.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available models: {available_models}")
        
        if version not in self._registry[name]['versions']:
            available_versions = list(self._registry[name]['versions'].keys())
            raise KeyError(f"Version '{version}' not found for model '{name}'. Available versions: {available_versions}")
        
        self._default_versions[name] = version
        logger.info(f"Set default version for '{name}' to '{version}'")

# Singleton instance
_registry_instance = None

def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry: Singleton instance of ModelRegistry
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance 