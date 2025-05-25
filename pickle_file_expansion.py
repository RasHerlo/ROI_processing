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
        # Normalize trace to [0,1]
        trace_min = np.min(trace)
        trace_max = np.max(trace)
        if trace_max == trace_min:  # Handle constant traces
            trace_norm = np.zeros_like(trace)
        else:
            trace_norm = (trace - trace_min) / (trace_max - trace_min)
        
        # Z-score the normalized trace
        if np.std(trace_norm) == 0:  # Handle constant traces
            trace_z = np.zeros_like(trace_norm)
        else:
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
        
        # Check for required keys (excluding rastermap_indices as it's now in a different file)
        required_keys = ['roi_indices', 'traces']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing required keys in input file: {missing_keys}")
        
        # Construct path to rastermap_model.npy
        data_dir = os.path.dirname(file_path)
        rastermap_path = os.path.join(data_dir, 'suite2p', 'plane0', 'rastermap_model.npy')
        
        if not os.path.exists(rastermap_path):
            raise FileNotFoundError(f"Rastermap file not found at: {rastermap_path}")
        
        # Load rastermap indices
        rastermap_data = np.load(rastermap_path, allow_pickle=True).item()
        if 'isort' not in rastermap_data:
            raise KeyError("'isort' key not found in rastermap_model.npy")
        
        rastermap_indices = rastermap_data['isort']
        
        # Extract traces and calculate trace length
        traces = data['traces']
        trace_length = len(traces[0])
        
        # Calculate normalized traces [0,1] and z-scored traces
        traces_norm = []
        traces_Zscr = []
        for trace in traces:
            # Normalize to [0,1]
            trace_min = np.min(trace)
            trace_max = np.max(trace)
            if trace_max == trace_min:  # Handle constant traces
                trace_norm = np.zeros_like(trace)
            else:
                trace_norm = (trace - trace_min) / (trace_max - trace_min)
            traces_norm.append(trace_norm)
            
            # Z-score the normalized trace
            if np.std(trace_norm) == 0:  # Handle constant traces
                trace_z = np.zeros_like(trace_norm)
            else:
                trace_z = (trace_norm - np.mean(trace_norm)) / np.std(trace_norm)
            traces_Zscr.append(trace_z)
        
        # Process traces for bPAC calculations
        bpac_stim, bpac_dff = process_traces(traces, trace_length)
        
        # Create output directory (same as input file directory)
        output_dir = os.path.dirname(file_path)
        
        # Create histogram
        create_bpac_dff_histogram(bpac_dff, output_dir)
        
        # Debug: Print shapes of all arrays before creating DataFrame
        print("\nDebug: Array shapes before DataFrame creation:")
        print(f"roi_indices shape: {data['roi_indices'].shape}")
        print(f"Number of traces: {len(traces)}")
        print(f"Length of each trace: {len(traces[0])}")
        print(f"rastermap_indices shape: {rastermap_indices.shape}")
        print(f"Number of normalized traces: {len(traces_norm)}")
        print(f"Number of Z-scored traces: {len(traces_Zscr)}")
        print(f"bpac_stim shape: {bpac_stim.shape}")
        print(f"bpac_dff shape: {bpac_dff.shape}")
        
        # Create DataFrame with all required columns
        try:
            output_df = pd.DataFrame({
                'roi_indices': data['roi_indices'],
                'traces': list(traces),  # Convert to list of arrays
                'rastermap_indices': rastermap_indices,
                'traces_norm': traces_norm,  # Already a list of arrays
                'traces_Zscr': traces_Zscr,  # Already a list of arrays
                'bPAC_stim': bpac_stim,
                'bPAC_dFF': bpac_dff
            })
        except ValueError as e:
            print("\nError creating DataFrame:")
            print(f"Error message: {str(e)}")
            print("\nChecking each column individually:")
            for col_name, col_data in [
                ('roi_indices', data['roi_indices']),
                ('traces', list(traces)),
                ('rastermap_indices', rastermap_indices),
                ('traces_norm', traces_norm),
                ('traces_Zscr', traces_Zscr),
                ('bPAC_stim', bpac_stim),
                ('bPAC_dFF', bpac_dff)
            ]:
                print(f"\n{col_name}:")
                print(f"Type: {type(col_data)}")
                if isinstance(col_data, list):
                    print(f"Length: {len(col_data)}")
                    if len(col_data) > 0:
                        print(f"First element type: {type(col_data[0])}")
                        if hasattr(col_data[0], 'shape'):
                            print(f"First element shape: {col_data[0].shape}")
                else:
                    print(f"Shape: {getattr(col_data, 'shape', 'No shape attribute')}")
                    print(f"Dimensions: {getattr(col_data, 'ndim', 'No ndim attribute')}")
            raise
        
        # Save processed data as DataFrame
        output_path = os.path.join(output_dir, 'processed_traces.pkl')
        output_df.to_pickle(output_path)
        
        print(f"Successfully processed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pickle_file_expansion.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    print(f"Processing file: {file_path}")
    if process_pickle_file(file_path):
        print("Processing completed successfully")
    else:
        print("Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 