#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB-XL ECG Data Preprocessing Summary Report

This script analyzes all ECG header (.hea) files in the dataset,
extracts key metadata fields, and generates a summary CSV report
without reprocessing the signal data.

Author: Huỳnh Phúc Nguyên
Created: May 2025
"""

import os
import logging
import pandas as pd
from pathlib import Path

from ai_module.preprocess.preprocess_ptbxl_data import load_header

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def summarize_all_headers():
    """
    Iterate through all .hea files in the raw data directory,
    extract metadata fields, and create a summary DataFrame.
    
    Returns:
        pandas.DataFrame: Summary of metadata from all header files
    """
    # Get all .hea files in the raw data directory
    hea_files = list(RAW_DATA_DIR.glob('*.hea'))
    
    total_files = len(hea_files)
    logger.info(f"Found {total_files} header files to summarize")
    
    # Initialize list to store metadata dictionaries
    all_metadata = []
    successful_count = 0
    
    # Process each file
    for file_path in hea_files:
        try:
            logger.info(f"Processing header file: {file_path.name}")
            
            # Load header metadata
            metadata = load_header(file_path)
            
            if metadata is None:
                logger.warning(f"Could not process header file: {file_path.name}")
                continue
            
            # Extract relevant fields
            summary_record = {
                'record_name': metadata.get('record_name', None),
                'fs': metadata.get('fs', None),
                'sig_len': metadata.get('sig_len', None),
                'n_sig': metadata.get('n_sig', None),
                'lead_names': "|".join(metadata.get('lead_names', [])) if metadata.get('lead_names') else None,
                'adc_gain': str(metadata.get('adc_gain', None)),
                'baseline': str(metadata.get('baseline', None)),
                'patient_id': metadata.get('patient_id', None)
            }
            
            all_metadata.append(summary_record)
            successful_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    logger.info(f"Successfully processed {successful_count} out of {total_files} files")
    
    # Create DataFrame from collected metadata
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        return df
    else:
        logger.warning("No valid metadata found, returning empty DataFrame")
        return pd.DataFrame()


def main():
    """
    Main function to summarize ECG header metadata and save to CSV.
    """
    logger.info("Starting ECG header metadata summarization")
    
    # Generate metadata summary
    summary_df = summarize_all_headers()
    
    if summary_df.empty:
        logger.error("No valid metadata to save")
        return
    
    # Save to CSV
    output_path = REPORTS_DIR / "preprocessing_summary.csv"
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved preprocessing summary to {output_path}")
    logger.info(f"Summary contains {len(summary_df)} records with {summary_df.shape[1]} fields")
    
    # Print basic statistics
    logger.info("----- Dataset Summary -----")
    logger.info(f"Number of records: {len(summary_df)}")
    
    # Sampling rate statistics
    if 'fs' in summary_df.columns:
        fs_values = summary_df['fs'].dropna().unique()
        logger.info(f"Sampling rates: {fs_values}")
    
    # Lead count statistics
    if 'n_sig' in summary_df.columns:
        n_sig_values = summary_df['n_sig'].dropna().unique()
        logger.info(f"Number of leads: {n_sig_values}")
    
    logger.info("ECG header metadata summarization completed")


if __name__ == "__main__":
    main()