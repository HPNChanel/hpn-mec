#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module for HPN Medicare AI Module

This module provides a DataLoader class to load, preprocess, and unify
health data from multiple CSV sources into a standardized format for ML models.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Union, Optional

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


class DataLoader:
    """
    A class to load and preprocess health data from multiple CSV sources.
    
    This class handles loading CSVs from multiple directories, standardizing
    column names, cleaning features, unifying target labels, and saving
    processed data to designated output directories.
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
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create output directories if they don't exist
        self.features_dir = self.processed_dir / "features"
        self.labels_dir = self.processed_dir / "labels"
        self.combined_dir = self.processed_dir / "combined"
        
        for directory in [self.features_dir, self.labels_dir, self.combined_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Use custom column mapping if provided, otherwise use default
        self.column_mapping = column_mapping if column_mapping else COLUMN_MAPPING
        
        # Use custom important features if provided, otherwise use default
        self.important_features = important_features if important_features else IMPORTANT_FEATURES
        
        logger.info(f"Initialized DataLoader with raw data directory: {self.raw_data_dir}")
    
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
        
        # Create rename mapping based on COLUMN_MAPPING
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
        - Normalizing numerical columns
        - Filling missing values
        - Handling categorical variables
        
        Args:
            df: DataFrame with loaded data
            
        Returns:
            DataFrame with cleaned features
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Step 1: Convert age from days to years if needed
        if 'age' in df_clean.columns:
            # Check if age is likely in days (assuming age > 1000 means it's in days)
            if df_clean['age'].mean() > 1000:
                logger.info("Converting age from days to years")
                df_clean['age'] = df_clean['age'] / 365.25
        
        # Step 2: Convert gender to binary if needed
        if 'gender' in df_clean.columns:
            # Check if 'gender' contains strings or non-binary values
            if df_clean['gender'].dtype == 'object' or df_clean['gender'].nunique() > 2:
                logger.info("Converting gender to binary (0=female, 1=male)")
                # Map strings to binary (1 for male, 0 for female)
                gender_map = {'m': 1, 'male': 1, 'f': 0, 'female': 0}
                # Convert case-insensitive
                if df_clean['gender'].dtype == 'object':
                    df_clean['gender'] = df_clean['gender'].str.lower().map(gender_map)
                    # Fill unmapped values with most common
                    if df_clean['gender'].isna().any():
                        most_common = df_clean['gender'].mode()[0]
                        df_clean['gender'] = df_clean['gender'].fillna(most_common)
        
        # Step 3: Handle missing values (use median for numerical, mode for categorical)
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                missing_count = df_clean[col].isna().sum()
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.info(f"Filled {missing_count} missing values in '{col}' with median: {median_val}")
                else:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    logger.info(f"Filled {missing_count} missing values in '{col}' with mode: {mode_val}")
        
        # Step 4: Check for and convert categorical columns to appropriate format
        for col in df_clean.columns:
            if col in IMPORTANT_FEATURES and not pd.api.types.is_numeric_dtype(df_clean[col]):
                logger.info(f"Converting non-numeric column '{col}' to numeric")
                try:
                    # Try to convert to numeric, setting errors to 'coerce' will turn
                    # non-convertable values to NaN
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    # Fill new NaN values with median
                    if df_clean[col].isna().any():
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                except Exception as e:
                    logger.warning(f"Could not convert '{col}' to numeric: {e}")
        
        return df_clean
    
    def normalize_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
        """
        Normalize numerical features using MinMaxScaler.
        
        Args:
            df: DataFrame with cleaned features
            
        Returns:
            Tuple of (DataFrame with normalized features, Dict of scalers)
        """
        df_norm = df.copy()
        scalers = {}
        
        # Identify numerical columns
        numerical_cols = [col for col in df_norm.columns 
                          if pd.api.types.is_numeric_dtype(df_norm[col]) 
                          and col not in TARGET_COLUMNS.keys()
                          and col != 'source']
        
        logger.info(f"Normalizing {len(numerical_cols)} numerical features")
        
        # Normalize each numerical column separately
        for col in numerical_cols:
            if df_norm[col].nunique() > 1:  # Only normalize if more than one unique value
                scaler = MinMaxScaler()
                # Reshape to 2D array for scaler
                values = df_norm[col].values.reshape(-1, 1)
                df_norm[col] = scaler.fit_transform(values).flatten()
                scalers[col] = scaler
        
        return df_norm, scalers
    
    def unify_labels(self, df: pd.DataFrame, supervised: bool = True) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Unify target labels from different sources into a single 'label' column.
        
        Args:
            df: DataFrame with processed features
            supervised: Whether to extract labels (True) or ignore them (False)
            
        Returns:
            Tuple of (DataFrame with features only, Optional array of labels)
        """
        if not supervised:
            logger.info("Supervised learning disabled, not extracting labels")
            return df, None
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Check for any target columns in the DataFrame
        target_columns = [col for col in TARGET_COLUMNS.keys() if col in df_features.columns]
        
        if not target_columns:
            logger.warning("No known target columns found for supervised learning")
            return df_features, None
        
        logger.info(f"Found target columns: {target_columns}")
        
        # Handle diabetes risk (convert values > 0 to 1)
        if 'diabetes_risk' in target_columns:
            logger.info("Converting diabetes_risk to binary (1 if > 0 else 0)")
            df_features['diabetes_risk'] = df_features['diabetes_risk'].apply(lambda x: 1 if x > 0 else 0)
        
        # Create unified label column (use first available target column)
        primary_target = target_columns[0]
        df_features['label'] = df_features[primary_target]
        
        # Extract labels and remove target columns from features
        labels = df_features['label'].values
        
        # Remove all target columns and the unified label from features
        columns_to_drop = target_columns + ['label']
        df_features = df_features.drop(columns=columns_to_drop)
        
        logger.info(f"Extracted labels from '{primary_target}', removed {len(columns_to_drop)} columns from features")
        
        return df_features, labels
    
    def filter_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to keep only important features (if present).
        
        Args:
            df: DataFrame with processed features
            
        Returns:
            DataFrame with only important features and metadata
        """
        # Always keep 'source' column
        columns_to_keep = ['source']
        
        # Add important features that are present in the DataFrame
        for feature in self.important_features:
            if feature in df.columns:
                columns_to_keep.append(feature)
        
        # Filter columns
        df_filtered = df[columns_to_keep]
        
        features_kept = len(columns_to_keep) - 1  # Subtract 1 for 'source'
        logger.info(f"Kept {features_kept}/{len(self.important_features)} important features")
        
        return df_filtered
    
    def save_data(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        timestamp: bool = False
    ) -> Dict[str, Path]:
        """
        Save processed data to output directories.
        
        Args:
            X: Feature matrix
            y: Labels (if available)
            feature_names: List of feature names (if available)
            timestamp: Whether to include timestamp in filenames
            
        Returns:
            Dictionary with paths to saved files
        """
        # Generate timestamp suffix if requested
        suffix = ""
        if timestamp:
            from datetime import datetime
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save feature matrix
        X_path = self.features_dir / f"X_train{suffix}.npy"
        np.save(X_path, X)
        logger.info(f"Saved feature matrix with shape {X.shape} to {X_path}")
        
        saved_paths = {'X': X_path}
        
        # Save feature names if available
        if feature_names:
            feature_names_path = self.features_dir / f"feature_names{suffix}.txt"
            with open(feature_names_path, 'w') as f:
                f.write('\n'.join(feature_names))
            logger.info(f"Saved feature names to {feature_names_path}")
            saved_paths['feature_names'] = feature_names_path
        
        # Save labels if available
        if y is not None:
            y_path = self.labels_dir / f"y_train{suffix}.npy"
            np.save(y_path, y)
            logger.info(f"Saved labels with shape {y.shape} to {y_path}")
            saved_paths['y'] = y_path
        
        return saved_paths
    
    def save_combined_dataset(self, df: pd.DataFrame, timestamp: bool = False) -> Path:
        """
        Save combined dataset to CSV.
        
        Args:
            df: DataFrame with processed data
            timestamp: Whether to include timestamp in filename
            
        Returns:
            Path to saved CSV file
        """
        # Generate timestamp suffix if requested
        suffix = ""
        if timestamp:
            from datetime import datetime
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to CSV
        output_path = self.combined_dir / f"final_dataset{suffix}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved combined dataset with {len(df)} rows to {output_path}")
        
        return output_path
    
    def load_data(
        self, 
        supervised: bool = True, 
        normalize: bool = True,
        filter_features: bool = True,
        save_files: bool = True,
        timestamp: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Main method to load, process, and return the data.
        
        Args:
            supervised: Whether to extract labels (True) or not (False)
            normalize: Whether to normalize numerical features
            filter_features: Whether to filter to keep only important features
            save_files: Whether to save processed data to disk
            timestamp: Whether to include timestamp in saved filenames
            
        Returns:
            If supervised=True: Tuple of (feature matrix, labels)
            If supervised=False: Feature matrix only
        """
        # Step 1: Find all CSV files
        csv_files = self.find_csv_files()
        if not csv_files:
            logger.error("No CSV files found. Cannot proceed.")
            return (np.array([]), np.array([])) if supervised else np.array([])
        
        # Step 2: Load all CSV files
        dataframes = []
        for file_path in csv_files:
            df = self.load_single_csv(file_path)
            if df is not None:
                dataframes.append(df)
        
        if not dataframes:
            logger.error("No CSV files could be loaded successfully.")
            return (np.array([]), np.array([])) if supervised else np.array([])
        
        logger.info(f"Successfully loaded {len(dataframes)} CSV files")
        
        # Step 3: Process each DataFrame
        processed_dfs = []
        for i, df in enumerate(dataframes):
            logger.info(f"Processing DataFrame {i+1}/{len(dataframes)} (source: {df['source'].iloc[0]})")
            
            # Standardize column names
            df = self.standardize_column_names(df)
            
            # Clean features
            df = self.clean_features(df)
            
            processed_dfs.append(df)
        
        # Step 4: Combine all processed DataFrames
        df_combined = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Combined {len(processed_dfs)} DataFrames into one with {len(df_combined)} rows")
        
        # Step 5: Filter to keep only important features if requested
        if filter_features:
            df_combined = self.filter_important_features(df_combined)
        
        # Step 6: Normalize numerical features if requested
        if normalize:
            df_combined, scalers = self.normalize_numerical_features(df_combined)
        
        # Save combined dataset before extracting labels
        if save_files:
            self.save_combined_dataset(df_combined, timestamp)
        
        # Step 7: Unify and extract labels if supervised
        if supervised:
            df_features, labels = self.unify_labels(df_combined, supervised=True)
            if labels is None:
                logger.warning("No labels found for supervised learning. Returning unsupervised result.")
                # Return unsupervised result if no labels found
                # Drop 'source' column from features
                if 'source' in df_combined.columns:
                    df_combined = df_combined.drop(columns=['source'])
                
                # Convert to numpy array
                X = df_combined.values
                
                # Save data if requested
                if save_files:
                    self.save_data(X, feature_names=df_combined.columns.tolist(), timestamp=timestamp)
                
                return X
        else:
            df_features = df_combined
            labels = None
        
        # Step 8: Convert to numpy arrays
        # Keep track of feature names before dropping 'source'
        feature_names = [col for col in df_features.columns if col != 'source']
        
        # Drop 'source' column from features
        if 'source' in df_features.columns:
            df_features = df_features.drop(columns=['source'])
        
        # Convert to numpy array
        X = df_features.values
        
        # Step 9: Save processed data if requested
        if save_files:
            self.save_data(X, labels, feature_names=feature_names, timestamp=timestamp)
        
        # Step 10: Return the processed data
        if supervised and labels is not None:
            return X, labels
        else:
            return X


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