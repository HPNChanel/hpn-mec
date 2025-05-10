#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Logging Configuration for AI Module

This module provides a unified logging configuration system that can be
used across the entire AI module to ensure consistent log formats and levels.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional, Union, List
import json
from datetime import datetime

# Import configuration system
from ai_module.utils.config import get_config

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log levels by environment
DEFAULT_LOG_LEVELS = {
    "development": logging.DEBUG,
    "testing": logging.INFO,
    "production": logging.WARNING
}

class LogManager:
    """
    Central logging manager to configure and retrieve loggers.
    
    This class provides a unified way to set up logging across the application,
    ensuring consistent formatting and handling.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logging manager if not already initialized."""
        if self._initialized:
            return
            
        # Get configuration
        config = get_config()
        
        # Determine environment
        self.environment = os.environ.get("AI_ENV", "development")
        
        # Set up log directory
        self.log_dir = Path(config.get("logging.directory", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self.configure_root_logger()
        
        # Track configured loggers
        self.loggers = {}
        
        self._initialized = True
    
    def configure_root_logger(self) -> None:
        """Configure the root logger with basic settings."""
        # Get configuration
        config = get_config()
        
        # Determine log level based on environment
        log_level_name = config.get(f"logging.level.{self.environment}", None)
        if log_level_name:
            log_level = getattr(logging, log_level_name.upper(), None)
        else:
            log_level = DEFAULT_LOG_LEVELS.get(self.environment, logging.INFO)
        
        # Get log format
        log_format = config.get("logging.format", DEFAULT_LOG_FORMAT)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
        
        # Add file handler for root logger if enabled
        if config.get("logging.root_to_file", False):
            file_handler = self._create_file_handler("root", log_format, log_level)
            root_logger.addHandler(file_handler)
    
    def _create_file_handler(
        self, 
        name: str, 
        log_format: str, 
        log_level: int,
        max_bytes: int = 10485760,  # 10 MB
        backup_count: int = 5
    ) -> logging.Handler:
        """
        Create a rotating file handler.
        
        Args:
            name: Logger name (used in filename)
            log_format: Format string for logs
            log_level: Logging level
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured file handler
        """
        # Create log filename based on name and date
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        # Set formatter and level
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(log_level)
        
        return file_handler
    
    def get_logger(
        self, 
        name: str, 
        level: Optional[Union[int, str]] = None,
        write_to_file: bool = False,
        propagate: bool = True
    ) -> logging.Logger:
        """
        Get a configured logger by name.
        
        Args:
            name: Logger name
            level: Logging level (if None, uses environment default)
            write_to_file: Whether to write logs to a file
            propagate: Whether to propagate to parent loggers
            
        Returns:
            Configured logger
        """
        # If logger already configured, return it
        if name in self.loggers:
            return self.loggers[name]
        
        # Get configuration
        config = get_config()
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Set level
        if level is None:
            # Try to get level from config, fall back to environment default
            level_name = config.get(f"logging.loggers.{name}.level", None)
            if level_name:
                level = getattr(logging, level_name.upper(), None)
            else:
                level = DEFAULT_LOG_LEVELS.get(self.environment, logging.INFO)
        elif isinstance(level, str):
            # Convert string level to int
            level = getattr(logging, level.upper(), logging.INFO)
            
        logger.setLevel(level)
        
        # Set propagation
        logger.propagate = propagate
        
        # Add file handler if requested
        if write_to_file or config.get(f"logging.loggers.{name}.file", False):
            # Get log format
            log_format = config.get("logging.format", DEFAULT_LOG_FORMAT)
            
            # Create and add file handler
            file_handler = self._create_file_handler(name, log_format, level)
            logger.addHandler(file_handler)
        
        # Store logger
        self.loggers[name] = logger
        
        return logger
    
    def set_log_level(self, name: str, level: Union[int, str]) -> None:
        """
        Change the logging level for a specific logger.
        
        Args:
            name: Logger name ('root' for root logger)
            level: New logging level (int or string like 'DEBUG')
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        if name == 'root':
            # Set root logger level
            logging.getLogger().setLevel(level)
        elif name in self.loggers:
            # Set specific logger level
            self.loggers[name].setLevel(level)
    
    def add_json_handler(self, name: str, json_file: Union[str, Path]) -> None:
        """
        Add a JSON formatter handler to a logger.
        
        Args:
            name: Logger name
            json_file: Path to JSON log file
        """
        # Get or create logger
        logger = self.get_logger(name)
        
        # Create JSON formatter
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_record["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": self.formatException(record.exc_info)
                    }
                
                return json.dumps(log_record)
        
        # Create handler
        handler = logging.FileHandler(json_file)
        handler.setFormatter(JsonFormatter())
        
        # Add handler to logger
        logger.addHandler(handler)


# Initialize singleton instance
_log_manager = LogManager()

# Helper functions
def get_logger(
    name: str, 
    level: Optional[Union[int, str]] = None,
    write_to_file: bool = False,
    propagate: bool = True
) -> logging.Logger:
    """
    Get a configured logger by name.
    
    This is the main function that should be used to get loggers throughout the application.
    
    Args:
        name: Logger name
        level: Logging level (if None, uses environment default)
        write_to_file: Whether to write logs to a file
        propagate: Whether to propagate to parent loggers
        
    Returns:
        Configured logger
    """
    return _log_manager.get_logger(name, level, write_to_file, propagate)

def set_log_level(name: str, level: Union[int, str]) -> None:
    """
    Change the logging level for a specific logger.
    
    Args:
        name: Logger name ('root' for root logger)
        level: New logging level (int or string like 'DEBUG')
    """
    _log_manager.set_log_level(name, level)

def add_json_handler(name: str, json_file: Union[str, Path]) -> None:
    """
    Add a JSON formatter handler to a logger.
    
    Args:
        name: Logger name
        json_file: Path to JSON log file
    """
    _log_manager.add_json_handler(name, json_file) 