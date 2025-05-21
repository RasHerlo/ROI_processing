import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def create_bpac_dff_histogram(bpac_dff, output_dir):
    """Create and save a histogram of bPAC_dFF values."""
    # Determine color and channel based on directory
    if 'SUPPORT_ChanA' in output_dir:
        color = 'red'
        channel = 'ChanA'
    else:
        color = 'green'
        channel = 'ChanB'
    
    # Calculate mean of valid (non-NaN) values
    valid_values = bpac_dff[~np.isnan(bpac_dff)]
    mean_value = np.mean(valid_values) if len(valid_values) > 0 else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.hist(valid_values, bins=50, color=color, alpha=0.7)
    plt.title(f'bPAC_dFF Distribution - {channel}')
    plt.xlabel('bPAC_dFF Value')
    plt.ylabel('Count')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    
    # Add text box with mean value
    textstr = f'Mean: {mean_value:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', 
             bbox=props)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the histogram
    output_path = os.path.join(output_dir, 'bPAC_dFF_values.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_traces(traces, trace_length):
    """Process traces to calculate bPAC_dFF values."""
    # Define stimulation and bPAC ranges based on trace length
    if trace_length == 1520:
        stim_range = (0, 300)
        bpac_range = (300, 600)
    elif trace_length == 2890:
        stim_range = (0, 600)
        bpac_range = (600, 1200)
    else:
        raise ValueError(f"Non-standard trace-length: {trace_length}")
    
    # Initialize arrays for results
    n_traces = len(traces)
    bpac_stim = np.zeros(n_traces)
    bpac_dff = np.zeros(n_traces)
    
    # Process each trace
    for i, trace in enumerate(traces):
        # Normalize trace
        trace_norm = trace / np.mean(trace)
        
        # Z-score the trace
        trace_z = (trace_norm - np.mean(trace_norm)) / np.std(trace_norm)
        
        # Calculate bPAC_stim (mean of stimulation period)
        bpac_stim[i] = np.mean(trace_z[stim_range[0]:stim_range[1]])
        
        # Calculate bPAC_dFF using only values outside bPAC range
        # Create mask for values outside bPAC range
        outside_bpac_mask = np.ones(len(trace_z), dtype=bool)
        outside_bpac_mask[bpac_range[0]:bpac_range[1]] = False
        
        # Calculate baseline mean using only values outside bPAC range
        baseline_mean = np.mean(trace_z[outside_bpac_mask])
        
        # Calculate bPAC_dFF as mean of bPAC range relative to baseline
        if baseline_mean != 0:
            bpac_dff[i] = np.mean(trace_z[bpac_range[0]:bpac_range[1]]) / baseline_mean
        else:
            bpac_dff[i] = np.nan
    
    return bpac_stim, bpac_dff

def process_pickle_file(file_path):
    """Process a single pickle file and save results."""
    try:
        # Read the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract traces and calculate trace length
        traces = data['traces']
        trace_length = len(traces[0])
        
        # Process traces
        bpac_stim, bpac_dff = process_traces(traces, trace_length)
        
        # Create output directory (same as input file directory)
        output_dir = os.path.dirname(file_path)
        
        # Create histogram
        create_bpac_dff_histogram(bpac_dff, output_dir)
        
        # Save processed data
        output_data = {
            'traces': traces,
            'bpac_stim': bpac_stim,
            'bpac_dff': bpac_dff
        }
        
        output_path = os.path.join(output_dir, 'processed_traces.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"Successfully processed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pickle_file_expansion.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    print(f"Starting processing in directory: {directory}")
    
    success_count = 0
    error_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'selected_traces.pkl':
                file_path = os.path.join(root, file)
                if process_pickle_file(file_path):
                    success_count += 1
                else:
                    error_count += 1
    
    print(f"Processing completed. Successfully processed: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main() 