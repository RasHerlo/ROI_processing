#!/usr/bin/env python3
"""
Quick script to analyze bPAC_metrics_collect.pkl file
"""

import pandas as pd
import sys

def analyze_pickle_file(file_path):
    """Analyze the pickle file and print statistics."""
    try:
        # Load the dataframe
        df = pd.read_pickle(file_path)
        
        print(f"=== bPAC Metrics Collection Analysis ===")
        print(f"File: {file_path}")
        print(f"Total experiments/rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for directory information
        if 'dir' in df.columns:
            unique_dirs = df['dir'].nunique()
            print(f"Unique experiment directories: {unique_dirs}")
            print(f"Directory breakdown:")
            dir_counts = df['dir'].value_counts()
            for dir_name, count in dir_counts.items():
                print(f"  {dir_name}: {count} experiments")
        
        # Check for cell type information
        if 'Celltype' in df.columns:
            print(f"\nCell type breakdown:")
            celltype_counts = df['Celltype'].value_counts()
            for celltype, count in celltype_counts.items():
                print(f"  {celltype}: {count} experiments")
        
        # Check for total counts
        if 'Tot_Counts' in df.columns:
            total_cells = df['Tot_Counts'].sum()
            print(f"\nTotal cells across all experiments: {total_cells}")
            print(f"Average cells per experiment: {total_cells / len(df):.1f}")
        
        # Show sample data
        print(f"\nFirst 3 experiments:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    file_path = r"E:\Rasmus-Guillermo\ECF1\bPAC_metrics_collect.pkl"
    analyze_pickle_file(file_path)

