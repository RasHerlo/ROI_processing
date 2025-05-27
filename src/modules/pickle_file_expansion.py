import os
import pickle
import numpy as np
import pandas as pd

def process_traces(traces, trace_length):
    """
    Process traces to calculate various metrics.
    
    Args:
        traces (np.ndarray): Array of traces with NaNs in stimulation range
        trace_length (int): Length of each trace
        
    Returns:
        tuple: Arrays containing processed metrics
    """
    # Define stimulation and post-stimulation ranges based on trace length
    if trace_length == 1520:
        stim_range = (726, 733)  # 7 points
        poststim_range = (726, 929)  # 203 points
    elif trace_length == 2890:
        stim_range = (1381, 1388)  # 7 points
        poststim_range = (1381, 1585)  # 204 points
    else:
        raise ValueError(f"Unsupported trace length: {trace_length}")
    
    # Initialize arrays for results
    num_traces = len(traces)
    bpac_stim = np.zeros(num_traces, dtype=object)  # Store range tuple
    poststim_80s = np.zeros(num_traces, dtype=object)  # Store range tuple
    bpac_dff = np.zeros(num_traces)
    
    # Process each trace
    for i, trace in enumerate(traces):
        # Store stimulation range
        bpac_stim[i] = stim_range
        
        # Store post-stimulation range
        poststim_80s[i] = poststim_range
        
        # Calculate baseline mean (excluding post-stim period)
        baseline_mask = ~np.isin(np.arange(len(trace)), np.arange(poststim_range[0], poststim_range[1]))
        baseline_mean = np.nanmean(trace[baseline_mask])
        
        # Calculate bPAC_dFF as mean of post-stim range relative to baseline
        poststim_mean = np.nanmean(trace[poststim_range[0]:poststim_range[1]])
        bpac_dff[i] = poststim_mean / baseline_mean
    
    return bpac_stim, poststim_80s, bpac_dff

def process_pickle_file(input_file):
    """
    Process a pickle file containing traces and save the results.
    
    Args:
        input_file (str): Path to input pickle file
    """
    # Construct output path in the same directory as input file
    output_file = os.path.join(os.path.dirname(input_file), 'processed_traces.pkl')
    
    # Read the input file
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check if required keys exist (excluding rastermap_indices as it's now in a different file)
    required_keys = ['traces', 'roi_indices']
    if not all(key in data for key in required_keys):
        raise ValueError(f"Input file must contain the following keys: {required_keys}")
    
    # Construct path to rastermap_model.npy
    data_dir = os.path.dirname(input_file)
    rastermap_path = os.path.join(data_dir, 'suite2p', 'plane0', 'rastermap_model.npy')
    
    if not os.path.exists(rastermap_path):
        raise FileNotFoundError(f"Rastermap file not found at: {rastermap_path}")
    
    # Load rastermap indices
    rastermap_data = np.load(rastermap_path, allow_pickle=True).item()
    if 'isort' not in rastermap_data:
        raise KeyError("'isort' key not found in rastermap_model.npy")
    
    rastermap_indices = rastermap_data['isort']
    
    # Compute rasmap_nums (position in sorted rasterplot)
    rasmap_nums = np.zeros_like(rastermap_indices)
    for i, idx in enumerate(rastermap_indices):
        rasmap_nums[idx] = i
    
    # Get the traces and their length
    traces = data['traces']
    trace_length = len(traces[0])
    
    # Create traces with NaNs in stimulation range
    traces_with_nans = traces.copy()
    if trace_length == 1520:
        traces_with_nans[:, 726:733] = np.nan
    elif trace_length == 2890:
        traces_with_nans[:, 1381:1388] = np.nan
    
    # Calculate normalized traces using traces_with_nans
    normalized_traces = np.zeros_like(traces_with_nans)
    for i, trace in enumerate(traces_with_nans):
        # Calculate min and max excluding NaNs
        trace_min = np.nanmin(trace)
        trace_max = np.nanmax(trace)
        # Normalize to 0-1 range, preserving NaNs
        normalized_traces[i] = (trace - trace_min) / (trace_max - trace_min)
    
    # Calculate z-scored traces using traces_with_nans
    z_scored_traces = np.zeros_like(traces_with_nans)
    for i, trace in enumerate(traces_with_nans):
        # Calculate mean and std excluding NaNs
        trace_mean = np.nanmean(trace)
        trace_std = np.nanstd(trace)
        # Z-score the trace, preserving NaNs
        z_scored_traces[i] = (trace - trace_mean) / trace_std
    
    # Process the traces to get additional metrics
    bpac_stim, poststim_80s, bpac_dff = process_traces(traces_with_nans, trace_length)
    
    # Create DataFrame with all the data
    df = pd.DataFrame({
        'roi_indices': data['roi_indices'],
        'rastermap_indices': rastermap_indices,
        'rasmap_nums': rasmap_nums,
        'traces': list(traces_with_nans),  # Store traces_with_nans
        'traces_norm': list(normalized_traces),
        'z_scored_traces': list(z_scored_traces),
        'bpac_stim': bpac_stim,
        'poststim_80s': poststim_80s,
        'bPAC_dFF': bpac_dff
    })
    
    # Save the DataFrame to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Successfully processed: {input_file}") 