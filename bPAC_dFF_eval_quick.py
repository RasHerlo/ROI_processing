#!/usr/bin/env python3
"""
bPAC_dFF_eval_quick.py

Script to evaluate bPAC_dFF values across multiple directories.
Walks through subdirectories, finds processed_traces.pkl files,
filters for bPAC_dFF > 1, and creates visualization figures.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import binary_dilation


def find_processed_traces_files(root_dir):
    """
    Walk through directories and find processed_traces.pkl files.
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        list: List of paths to processed_traces.pkl files
    """
    processed_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Check if there's exactly one processed_traces.pkl file in this directory
        pkl_files = [f for f in files if f == 'processed_traces.pkl']
        
        if len(pkl_files) == 1:
            processed_files.append(os.path.join(root, 'processed_traces.pkl'))
        elif len(pkl_files) > 1:
            print(f"ERROR: Multiple processed_traces.pkl files found in {root}. Skipping this folder.")
    
    return processed_files


def load_data(file_path):
    """
    Load processed_traces.pkl and associated suite2p data.
    
    Args:
        file_path (str): Path to processed_traces.pkl
        
    Returns:
        tuple: (processed_data, ops, stat) or (None, None, None) if error
    """
    try:
        # Load processed data
        processed_data = pd.read_pickle(file_path)
        
        # Construct paths to suite2p data
        data_dir = os.path.dirname(file_path)  # This is the "derippled" folder
        suite2p_dir = os.path.join(data_dir, 'suite2p', 'plane0')  # suite2p is inside derippled
        
        ops_path = os.path.join(suite2p_dir, 'ops.npy')
        stat_path = os.path.join(suite2p_dir, 'stat.npy')
        
        # Load suite2p data
        if not os.path.exists(ops_path):
            print(f"ERROR: ops.npy not found at {ops_path}")
            return None, None, None
            
        if not os.path.exists(stat_path):
            print(f"ERROR: stat.npy not found at {stat_path}")
            return None, None, None
        
        ops = np.load(ops_path, allow_pickle=True).item()
        stat = np.load(stat_path, allow_pickle=True)
        
        return processed_data, ops, stat
        
    except Exception as e:
        print(f"ERROR: Failed to load data from {file_path}: {str(e)}")
        return None, None, None


def filter_and_sort_data(processed_data):
    """
    Filter data for bPAC_dFF > 1 and sort by bPAC_dFF descending.
    
    Args:
        processed_data (pd.DataFrame): Processed data
        
    Returns:
        pd.DataFrame: Filtered and sorted data
    """
    # Filter for bPAC_dFF > 1
    filtered_data = processed_data[processed_data['bPAC_dFF'] > 1].copy()
    
    if len(filtered_data) == 0:
        return None
    
    # Sort by bPAC_dFF descending
    filtered_data = filtered_data.sort_values('bPAC_dFF', ascending=False).reset_index(drop=True)
    
    return filtered_data


def create_color_mapping(bpac_values):
    """
    Create color mapping for bPAC_dFF values (same logic as trace_inspection GUI).
    
    Args:
        bpac_values (np.ndarray): bPAC_dFF values
        
    Returns:
        tuple: (norm, cmap, vmin, vmax)
    """
    # Calculate maximum deviation from 1.0
    max_deviation = max(abs(np.max(bpac_values) - 1.0), abs(np.min(bpac_values) - 1.0))
    
    # Set symmetric limits around 1.0
    vmin = 1.0 - max_deviation
    vmax = 1.0 + max_deviation
    
    # Create normalizer centered around 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.bwr
    
    return norm, cmap, vmin, vmax


def create_visualization(processed_data, ops, stat, output_path):
    """
    Create the 2x2 visualization figure.
    
    Args:
        processed_data (pd.DataFrame): Filtered and sorted data
        ops (dict): Suite2p ops data
        stat (np.ndarray): Suite2p stat data
        output_path (str): Path to save the figure
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Get bPAC_dFF values for color mapping (used for other plots, not rasterplot)
    bpac_values = processed_data['bPAC_dFF'].to_numpy()
    norm, cmap, vmin, vmax = create_color_mapping(bpac_values)
    
    # Subplot 1 (upper left): bPAC_dFF distribution histogram
    ax1 = axes[0, 0]
    
    # Get all bPAC_dFF values from the original data (not just filtered)
    # We need to load the original data to get all values
    original_data_path = os.path.join(os.path.dirname(output_path), 'processed_traces.pkl')
    original_data = pd.read_pickle(original_data_path)
    all_bpac_values = original_data['bPAC_dFF'].to_numpy()
    
    # Remove NaN values
    valid_bpac_values = all_bpac_values[~np.isnan(all_bpac_values)]
    
    # Create histogram with custom colors
    bins = np.linspace(valid_bpac_values.min(), valid_bpac_values.max(), 30)
    hist, bin_edges = np.histogram(valid_bpac_values, bins=bins, density=True)
    
    # Color bars based on whether they're above or below 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = ['red' if center > 1 else 'blue' for center in bin_centers]
    
    # Plot histogram bars
    for i in range(len(hist)):
        ax1.bar(bin_centers[i], hist[i], width=bin_edges[1]-bin_edges[0], 
                color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu = np.mean(valid_bpac_values)
    sigma = np.std(valid_bpac_values)
    
    # Generate points for normal distribution curve
    x = np.linspace(valid_bpac_values.min(), valid_bpac_values.max(), 100)
    y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Plot normal distribution curve
    ax1.plot(x, y, 'k-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={sigma:.3f}')
    
    # Add vertical line at 1.0
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='bPAC_dFF = 1.0')
    
    ax1.set_title('bPAC_dFF Distribution')
    ax1.set_xlabel('bPAC_dFF')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2 (upper right): Rasterplot (original subplot 1)
    ax2 = axes[0, 1]
    
    # Convert traces_norm to 2D array
    traces_norm = processed_data['traces_norm'].tolist()
    traces_norm_array = np.array(traces_norm)
    
    # Create rasterplot with simple bwr colormap (no normalization) - matching trace inspection GUI
    im = ax2.imshow(traces_norm_array, aspect='auto', cmap='bwr')
    ax2.set_title('Rasterplot (sorted by bPAC_dFF)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ROI Index')
    
    # Add colorbar for trace intensity (not bPAC_dFF)
    cbar = plt.colorbar(im, ax=ax2, label='Normalized Trace Intensity')
    
    # Subplot 3 (lower left): Overlaid traces (original subplot 2)
    ax3 = axes[1, 0]
    
    # Plot individual traces with single colors
    for i, (_, row) in enumerate(processed_data.iterrows()):
        trace = row['traces_norm']
        bpac_val = row['bPAC_dFF']
        color = cmap(norm(bpac_val))
        ax3.plot(trace, color=color, alpha=0.7, linewidth=0.5)
    
    # Plot mean trace
    mean_trace = np.mean(traces_norm_array, axis=0)
    ax3.plot(mean_trace, color='black', linewidth=2, label='Mean')
    ax3.set_title('Overlaid Traces')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Normalized Intensity')
    ax3.legend()
    
    # Subplot 4 (lower right): ROI visualization (original subplot 3)
    ax4 = axes[1, 1]
    
    # Get background image (meanImgE)
    bg_img = ops['meanImgE']
    ax4.imshow(bg_img, cmap='gray')
    
    # Plot ROIs colored by bPAC_dFF values
    for i, (_, row) in enumerate(processed_data.iterrows()):
        roi_idx = row['roi_indices']
        bpac_val = row['bPAC_dFF']
        
        # Get ROI coordinates
        roi_stat = stat[roi_idx]
        xpix, ypix = roi_stat['xpix'], roi_stat['ypix']
        
        # Color based on bPAC_dFF value
        color = cmap(norm(bpac_val))
        ax4.scatter(xpix, ypix, c=[color], s=1, alpha=0.8)
    
    ax4.set_title('ROI Visualization')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add colorbar for ROI plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax4, label='bPAC_dFF')
    cbar2.ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def process_file(file_path):
    """
    Process a single processed_traces.pkl file.
    
    Args:
        file_path (str): Path to processed_traces.pkl file
    """
    print(f"\nProcessing: {file_path}")
    
    # Load data
    processed_data, ops, stat = load_data(file_path)
    if processed_data is None:
        return
    
    # Filter and sort data
    filtered_data = filter_and_sort_data(processed_data)
    if filtered_data is None:
        print(f"No ROIs with bPAC_dFF > 1 found in {file_path}")
        return
    
    print(f"Found {len(filtered_data)} ROIs with bPAC_dFF > 1")
    print(f"bPAC_dFF range: {filtered_data['bPAC_dFF'].min():.4f} - {filtered_data['bPAC_dFF'].max():.4f}")
    
    # Create output path with channel-specific naming
    output_dir = os.path.dirname(file_path)
    
    # Check if we're in a SUPPORT_ChanA or SUPPORT_ChanB folder
    current_dir = output_dir
    channel_suffix = ""
    
    # Walk up the directory tree to find SUPPORT_ChanA or SUPPORT_ChanB
    while current_dir != os.path.dirname(current_dir):  # Stop at root
        dir_name = os.path.basename(current_dir)
        if dir_name == "SUPPORT_ChanA":
            channel_suffix = "_ChanA"
            break
        elif dir_name == "SUPPORT_ChanB":
            channel_suffix = "_ChanB"
            break
        current_dir = os.path.dirname(current_dir)
    
    # Create output filename with channel suffix
    output_filename = f'bPAC_dFF_evaluation{channel_suffix}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    # Create visualization
    create_visualization(filtered_data, ops, stat, output_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate bPAC_dFF values across directories')
    parser.add_argument('directory', help='Root directory to search for processed_traces.pkl files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"ERROR: Directory {args.directory} does not exist")
        sys.exit(1)
    
    print(f"Searching for processed_traces.pkl files in: {args.directory}")
    
    # Find all processed_traces.pkl files
    processed_files = find_processed_traces_files(args.directory)
    
    if not processed_files:
        print("No processed_traces.pkl files found")
        sys.exit(1)
    
    print(f"Found {len(processed_files)} processed_traces.pkl files")
    
    # Process each file
    for file_path in processed_files:
        process_file(file_path)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main() 