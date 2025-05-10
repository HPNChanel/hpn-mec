#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module for HPN Medicare AI Module

This module provides modular components for loading, preprocessing, and unifying
health data from multiple CSV sources into a standardized format for ML models.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Union, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - Column name standardization mapping
COLUMN_MAPPING = {
    # Blood pressure columns
    "ap_hi": "systolic_bp",
    "ap_lo": "diastolic_bp",
    "highbp": "high_bp",
    "sysbp": "systolic_bp",
    "diabp": "diastolic_bp",
    
    # Glucose related columns
    "gluc": "glucose",
    
    # Cholesterol related columns
    "cholesterol": "cholesterol_level",
    "highchol": "high_cholesterol",
    "totchol": "total_cholesterol",
    
    # Disease risk targets
    "diabetes_012": "diabetes_risk",
    "cardio": "cardiovascular_risk",
    "tenyearchd": "chd_risk",
    
    # Other health indicators
    "bmi": "bmi",
    "weight": "weight",
    "height": "height",
    "smoke": "smoker",
    "smoking": "smoker",
    "currentsmoker": "smoker",
    "cigsperday": "cigarettes_per_day",
    "alco": "alcohol",
    "alcohol": "alcohol",
    "active": "physically_active",
    "physactivity": "physically_active",
    "heartrate": "heart_rate",
    "prevalentstroke": "stroke_history",
    "prevalenthyp": "hypertension_history",
    "age": "age",
    "gender": "gender",
    "sex": "gender",
    "male": "gender",
    "education": "education_level",
    "diffwalk": "difficulty_walking",
    "genhlth": "general_health",
}

# Important health features to keep (if present)
IMPORTANT_FEATURES = [
    "age", "gender", "bmi", "systolic_bp", "diastolic_bp", "high_bp",
    "glucose", "cholesterol_level", "total_cholesterol", "high_cholesterol",
    "smoker", "cigarettes_per_day", "alcohol", "physically_active", "heart_rate",
    "stroke_history", "hypertension_history", "education_level",
    "weight", "height", "difficulty_walking", "general_health"
]

# Target label mapping for supervised learning
TARGET_COLUMNS = {
    "diabetes_risk": "label",
    "cardiovascular_risk": "label",
    "chd_risk": "label"
}


class CSVLoader:
    """
    Class to handle loading CSV files from various sources.
    
    Handles file discovery, delimiter detection, and initial loading.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        """
        Initialize the CSV Loader with configuration parameters.
        
        Args:
            raw_data_dir: Directory containing raw CSV files (default: "data/raw")
        """
        # Convert string paths to Path objects
        self.raw_data_dir = Path(raw_data_dir)
        logger.info(f"Initialized CSVLoader with data directory: {self.raw_data_dir}")
    
    def find_csv_files(self) -> List[Path]:
        """
        Find all CSV files in the raw data directory recursively.
        
        Returns:
            List of Path objects for CSV files
        """
        csv_files = []
        
        # Use recursive glob to find all CSV files
        for csv_path in self.raw_data_dir.glob("**/*.csv"):
            csv_files.append(csv_path)
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_data_dir}")
        else:
            logger.info(f"Found {len(csv_files)} CSV files in {self.raw_data_dir}")
        
        return csv_files
    
    def detect_delimiter(self, file_path: Path) -> str:
        """
        Detect the delimiter used in a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Detected delimiter (';' or ',')
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                # Check for semicolon
                if ';' in first_line:
                    return ';'
                
                # Default to comma
                return ','
        except Exception as e:
            logger.error(f"Error detecting delimiter for {file_path}: {e}")
            # Default to comma if there's an error
            return ','
    
    def load_single_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file with automatic delimiter detection.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the CSV data or None if loading fails
        """
        try:
            # Detect delimiter
            delimiter = self.detect_delimiter(file_path)
            logger.info(f"Loading {file_path.name} with delimiter: '{delimiter}'")
            
            # Load CSV
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Add source column based on filename
            source_name = file_path.stem
            df['source'] = source_name
            
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_csvs(self) -> pd.DataFrame:
        """
        Load all CSV files from the raw data directory into a single DataFrame.
        
        Returns:
            Combined DataFrame with all CSV data
        """
        csv_files = self.find_csv_files()
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_dir}")
            
        # Load each CSV and collect in a list
        dataframes = []
        for file_path in csv_files:
            df = self.load_single_csv(file_path)
            if df is not None and not df.empty:
                dataframes.append(df)
        
        if not dataframes:
            raise ValueError("No valid data loaded from CSV files")
        
        # Combine all dataframes into one
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        logger.info(f"Combined {len(dataframes)} CSV files into single DataFrame with {len(combined_df)} rows")
        
        return combined_df


class FeatureProcessor:
    """
    Class to handle feature processing, standardization, and cleaning.
    
    Standardizes column names, cleans features, and filters important features.
    """
    
    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        important_features: Optional[List[str]] = None
    ):
        """
        Initialize the FeatureProcessor.
        
        Args:
            column_mapping: Custom column name mapping (default: None, uses built-in mapping)
            important_features: List of important features to keep (default: None, uses built-in list)
        """
        # Use custom column mapping if provided, otherwise use default
        self.column_mapping = column_mapping if column_mapping else COLUMN_MAPPING
        
        # Use custom important features if provided, otherwise use default
        self.important_features = important_features if important_features else IMPORTANT_FEATURES
        
        logger.info(f"Initialized FeatureProcessor with {len(self.column_mapping)} column mappings")
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names using the column mapping.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Convert all column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Create rename mapping based on column_mapping
        rename_dict = {}
        for old_name in df.columns:
            if old_name in self.column_mapping:
                rename_dict[old_name] = self.column_mapping[old_name]
        
        # Rename columns
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"Renamed {len(rename_dict)} columns using standardized names")
        
        return df
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess features in the DataFrame.
        
        This includes:
        - Converting age from days to years (if needed)
        - Handling categorical variables
        - Initial data cleaning
        
        Args:
            df: DataFrame with features to clean
            
        Returns:
            DataFrame with cleaned features
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Convert age from days to years if large values detected
            if 'age' in df.columns and df['age'].max() > 365:
                logger.info("Converting age from days to years")
                df.loc[:, 'age'] = df['age'] / 365.25
            
            # Handle gender/sex encoding
            if 'gender' in df.columns:
                # Convert text values to binary
                if df['gender'].dtype == 'object':
                    gender_map = {
                        'f': 0, 'female': 0, 'woman': 0, 'm': 1, 'male': 1, 'man': 1
                    }
                    df.loc[:, 'gender'] = df['gender'].str.lower().map(gender_map).fillna(df['gender'])
                
                # Ensure integer encoding
                df.loc[:, 'gender'] = df['gender'].astype(float).astype('Int64')
            
            # Convert categorical variables to numeric where needed
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['source']:
                    try:
                        # Try to convert to numeric
                        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            logger.info(f"Cleaned features in DataFrame with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning features: {e}")
            return df
    
    def filter_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to keep only important features.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with only important features
        """
        # Get list of available important features in this dataset
        available_features = [f for f in self.important_features if f in df.columns]
        
        if not available_features:
            logger.warning("No important features found in dataset. Keeping all features.")
            return df
        
        # Always keep the source column
        if 'source' in df.columns and 'source' not in available_features:
            available_features.append('source')
        
        # Filter DataFrame
        filtered_df = df[available_features].copy()
        
        logger.info(f"Filtered features from {len(df.columns)} to {len(filtered_df.columns)} columns")
        return filtered_df
    
    def unify_labels(self, df: pd.DataFrame, supervised: bool = True) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Extract and unify target labels from DataFrame.
        
        Args:
            df: DataFrame with features and potential target columns
            supervised: Whether to extract labels for supervised learning
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        if not supervised:
            return df, None
        
        # Look for target columns
        target_columns = [col for col in df.columns if col in TARGET_COLUMNS.keys()]
        
        if not target_columns:
            logger.warning("No target columns found for supervised learning")
            return df, None
        
        # Use the first found target column
        target_col = target_columns[0]
        logger.info(f"Using '{target_col}' as target for supervised learning")
        
        # Extract labels
        labels = df[target_col].values
        
        # Remove all target columns from features
        for col in target_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        return df, labels


class DataNormalizer:
    """
    Class to handle data normalization and preprocessing for machine learning.
    
    Normalizes numerical features and handles missing values.
    """
    
    def __init__(self):
        """Initialize the DataNormalizer."""
        logger.info("Initialized DataNormalizer")
    
    def normalize_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
        """
        Normalize numerical features using Min-Max scaling.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (normalized_df, scaler_dict)
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Identify numerical columns (excluding 'source')
        numerical_cols = [
            col for col in df.columns 
            if col != 'source' and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if not numerical_cols:
            logger.warning("No numerical columns found for normalization")
            return df, {}
        
        # Initialize scalers dictionary
        scalers = {}
        
        # Normalize each numerical column separately
        for col in numerical_cols:
            # Handle missing values
            df[col] = df[col].fillna(df[col].median())
            
            # Create and fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Reshape for scaler
            values = df[col].values.reshape(-1, 1)
            # Fit and transform
            normalized_values = scaler.fit_transform(values).flatten()
            # Update column values
            df.loc[:, col] = normalized_values
            # Store scaler
            scalers[col] = scaler
        
        logger.info(f"Normalized {len(numerical_cols)} numerical features")
        return df, scalers
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Fill missing values for each column
        for col in df.columns:
            if col == 'source':
                continue
                
            # Get non-null values
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                # If all values are missing, drop the column
                df = df.drop(columns=[col])
                logger.warning(f"Dropped column '{col}' as all values were missing")
            elif pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, fill with median
                df.loc[:, col] = df[col].fillna(df[col].median())
            else:
                # For non-numeric columns, fill with mode
                df.loc[:, col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
        
        logger.info(f"Handled missing values in DataFrame with {len(df)} rows")
        return df


class DataLoader:
    """
    Coordinator class that integrates CSV loading, feature processing, and normalization.
    
    This class orchestrates the entire data loading and preprocessing pipeline.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        column_mapping: Optional[Dict[str, str]] = None,
        important_features: Optional[List[str]] = None
    ):
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            raw_data_dir: Directory containing raw CSV files (default: "data/raw")
            processed_dir: Directory to save processed data (default: "data/processed")
            column_mapping: Custom column name mapping (default: None, uses built-in mapping)
            important_features: List of important features to keep (default: None, uses built-in list)
        """
        # Convert string paths to Path objects
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create output directories if they don't exist
        self.features_dir = self.processed_dir / "features"
        self.labels_dir = self.processed_dir / "labels"
        self.combined_dir = self.processed_dir / "combined"
        self.latents_dir = self.processed_dir / "latents"
        self.scores_dir = self.processed_dir / "scores"
        
        for directory in [self.features_dir, self.labels_dir, self.combined_dir, 
                          self.latents_dir, self.scores_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize component classes
        self.csv_loader = CSVLoader(raw_data_dir)
        self.feature_processor = FeatureProcessor(column_mapping, important_features)
        self.normalizer = DataNormalizer()
        
        logger.info(f"Initialized DataLoader with raw data directory: {self.raw_data_dir}")
    
    def save_data(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        timestamp: bool = False
    ) -> Dict[str, Path]:
        """
        Save processed features and labels to designated directories.
        
        Args:
            X: Feature array
            y: Labels array (optional)
            feature_names: List of feature names (optional)
            timestamp: Whether to add timestamp to filenames
            
        Returns:
            Dictionary of saved file paths
        """
        # Generate timestamp if requested
        ts = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
        
        # Prepare paths
        features_path = self.features_dir / f"X{ts}.npy"
        feature_names_path = self.features_dir / f"feature_names{ts}.txt"
        
        # Save features
        np.save(features_path, X)
        logger.info(f"Saved features to {features_path}")
        
        # Save feature names if provided
        if feature_names:
            with open(feature_names_path, 'w') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            logger.info(f"Saved feature names to {feature_names_path}")
        
        # Prepare result dict
        result = {
            "features": features_path,
            "feature_names": feature_names_path if feature_names else None
        }
        
        # Save labels if provided
        if y is not None:
            labels_path = self.labels_dir / f"y{ts}.npy"
            np.save(labels_path, y)
            logger.info(f"Saved labels to {labels_path}")
            result["labels"] = labels_path
        
        return result
    
    def save_combined_dataset(self, df: pd.DataFrame, timestamp: bool = False) -> Path:
        """
        Save the complete processed DataFrame for future reference.
        
        Args:
            df: Processed DataFrame
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to saved file
        """
        # Generate timestamp if requested
        ts = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
        
        # Prepare path
        combined_path = self.combined_dir / f"combined_data{ts}.csv"
        
        # Save as CSV
        df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined dataset to {combined_path}")
        
        return combined_path
    
    def load_data(
        self, 
        supervised: bool = True, 
        normalize: bool = True,
        filter_features: bool = True,
        save_files: bool = True,
        timestamp: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Execute the full data loading and preprocessing pipeline.
        
        Args:
            supervised: Whether to extract labels for supervised learning
            normalize: Whether to normalize numerical features
            filter_features: Whether to filter to keep only important features
            save_files: Whether to save processed files
            timestamp: Whether to add timestamp to filenames
            
        Returns:
            For supervised=True: Tuple of (features, labels)
            For supervised=False: Features array only
        """
        try:
            # Load all CSVs
            df = self.csv_loader.load_all_csvs()
            
            # Standardize column names
            df = self.feature_processor.standardize_column_names(df)
            
            # Clean features
            df = self.feature_processor.clean_features(df)
            
            # Handle missing values
            df = self.normalizer.handle_missing_values(df)
            
            # Filter important features if requested
            if filter_features:
                df = self.feature_processor.filter_important_features(df)
            
            # Extract labels if supervised
            df, labels = self.feature_processor.unify_labels(df, supervised)
            
            # Normalize numerical features if requested
            if normalize:
                df, _ = self.normalizer.normalize_numerical_features(df)
            
            # Remove 'source' column if present for final features
            if 'source' in df.columns:
                feature_names = [col for col in df.columns if col != 'source']
                df = df.drop(columns=['source'])
            else:
                feature_names = list(df.columns)
            
            # Convert to numpy array
            features = df.values
            
            # Save files if requested
            if save_files:
                self.save_data(features, labels, feature_names, timestamp)
                
                # Create a copy to save as combined dataset with labels
                combined_df = df.copy()
                if labels is not None:
                    combined_df['label'] = labels
                
                self.save_combined_dataset(combined_df, timestamp)
            
            # Return results based on supervised flag
            if supervised and labels is not None:
                return features, labels
            else:
                return features
            
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            raise


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and preprocess health data for ML models")
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        default="data/raw",
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--unsupervised", 
        action="store_true",
        help="Use unsupervised mode (don't extract labels)"
    )
    parser.add_argument(
        "--no-normalize", 
        action="store_true",
        help="Skip normalization of numerical features"
    )
    parser.add_argument(
        "--no-filter", 
        action="store_true",
        help="Don't filter to keep only important features"
    )
    parser.add_argument(
        "--timestamp", 
        action="store_true",
        help="Include timestamp in saved filenames"
    )
    
    args = parser.parse_args()
    
    # Initialize DataLoader with CLI arguments
    dl = DataLoader(
        raw_data_dir=args.raw_dir,
        processed_dir=args.processed_dir
    )
    
    # Load data with CLI options
    result = dl.load_data(
        supervised=not args.unsupervised,
        normalize=not args.no_normalize,
        filter_features=not args.no_filter,
        timestamp=args.timestamp
    )
    
    # Print shape of result
    if isinstance(result, tuple):
        X, y = result
        print(f"Loaded data: X.shape={X.shape}, y.shape={y.shape}")
    else:
        X = result
        print(f"Loaded data: X.shape={X.shape}")