#!/usr/bin/env python3
"""
batch_process_traces.py

Batch process selected_traces.pkl files into processed_traces.pkl files
by walking through all subdirectories under a given root directory.
Uses existing processing functionality from pickle_file_expansion module.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from modules import pickle_file_expansion
except ImportError as e:
    print(f"ERROR: Could not import processing module: {e}")
    print("Make sure you're running this script from the ROI_processing root directory")
    sys.exit(1)


def load_config():
    """Load configuration from config.json file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        else:
            return {}
    except Exception as e:
        print(f"Warning: Error loading config file: {e}")
        return {}


def get_initial_directory():
    """Get initial directory from config or default."""
    config = load_config()
    if config and 'data_folder' in config:
        config_dir = config['data_folder']
        if os.path.exists(config_dir):
            return config_dir
    
    # Fallback to current working directory
    return os.getcwd()


def find_selected_traces_files(root_dir):
    """
    Walk through directories and find selected_traces.pkl files.
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        list: List of paths to selected_traces.pkl files
    """
    selected_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Look for selected_traces.pkl files
        if 'selected_traces.pkl' in files:
            selected_files.append(os.path.join(root, 'selected_traces.pkl'))
    
    return selected_files


def validate_required_files(selected_traces_path):
    """
    Check if all required files exist for processing.
    
    Args:
        selected_traces_path (str): Path to selected_traces.pkl file
        
    Returns:
        tuple: (is_valid, rastermap_path, processed_path)
    """
    # Get the directory containing selected_traces.pkl
    data_dir = os.path.dirname(selected_traces_path)
    
    # Check for rastermap_model.npy in suite2p/plane0/
    rastermap_path = os.path.join(data_dir, 'suite2p', 'plane0', 'rastermap_model.npy')
    
    # Path where processed_traces.pkl would be created
    processed_path = os.path.join(data_dir, 'processed_traces.pkl')
    
    # Validate that rastermap file exists
    if not os.path.exists(rastermap_path):
        return False, rastermap_path, processed_path
    
    return True, rastermap_path, processed_path


def should_process_file(processed_path, overwrite):
    """
    Determine if a file should be processed based on existence and overwrite flag.
    
    Args:
        processed_path (str): Path to potential processed_traces.pkl file
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        tuple: (should_process, reason)
    """
    if os.path.exists(processed_path):
        if overwrite:
            return True, "overwriting existing file"
        else:
            return False, "file already exists (use --overwrite to force)"
    else:
        return True, "creating new file"


def process_directory(selected_traces_path, overwrite):
    """
    Process a single selected_traces.pkl file.
    
    Args:
        selected_traces_path (str): Path to selected_traces.pkl file
        overwrite (bool): Whether to overwrite existing processed files
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Validate required files
        is_valid, rastermap_path, processed_path = validate_required_files(selected_traces_path)
        
        if not is_valid:
            return False, f"Required rastermap file not found: {rastermap_path}"
        
        # Check if we should process this file
        should_process, reason = should_process_file(processed_path, overwrite)
        
        if not should_process:
            return False, f"Skipped: {reason}"
        
        # Process the file using existing functionality
        pickle_file_expansion.process_pickle_file(selected_traces_path)
        
        # Verify that the output file was created
        if os.path.exists(processed_path):
            return True, f"Successfully processed ({reason})"
        else:
            return False, "Processing completed but output file was not created"
        
    except Exception as e:
        return False, f"Processing failed: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Batch process selected_traces.pkl files into processed_traces.pkl files'
    )
    parser.add_argument(
        'directory', 
        nargs='?',  # Make directory optional
        help='Root directory to search for selected_traces.pkl files (default: from config.json)'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true',
        help='Overwrite existing processed_traces.pkl files'
    )
    
    args = parser.parse_args()
    
    # Determine root directory
    if args.directory:
        root_dir = args.directory
    else:
        root_dir = get_initial_directory()
        print(f"Using directory from config: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"ERROR: Directory {root_dir} does not exist")
        sys.exit(1)
    
    print(f"Searching for selected_traces.pkl files in: {root_dir}")
    if args.overwrite:
        print("Overwrite mode enabled - existing processed_traces.pkl files will be replaced")
    
    # Find all selected_traces.pkl files
    selected_files = find_selected_traces_files(root_dir)
    
    if not selected_files:
        print("No selected_traces.pkl files found")
        sys.exit(1)
    
    print(f"Found {len(selected_files)} selected_traces.pkl files")
    
    # Process each file
    total_files = len(selected_files)
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(selected_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {file_path}")
        
        success, message = process_directory(file_path, args.overwrite)
        
        if success:
            processed_count += 1
            print(f"  ✓ {message}")
        elif "Skipped:" in message:
            skipped_count += 1
            print(f"  - {message}")
        else:
            failed_count += 1
            print(f"  ✗ {message}")
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total files found:     {total_files}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped:               {skipped_count}")
    print(f"Failed:                {failed_count}")
    
    if failed_count > 0:
        print(f"\nNote: {failed_count} files failed to process. Check error messages above.")
    
    if skipped_count > 0 and not args.overwrite:
        print(f"\nNote: {skipped_count} files were skipped (already processed). Use --overwrite to force reprocessing.")


if __name__ == "__main__":
    main() 