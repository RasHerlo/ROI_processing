import pandas as pd
import os
import argparse
import sys

def main():
    """Main function to read and inspect a pickle file."""
    parser = argparse.ArgumentParser(description='Read and inspect a pickle file')
    parser.add_argument('filepath', help='Path to the pickle file to read')
    
    args = parser.parse_args()
    file_path = args.filepath
    
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
        else:
            data = pd.read_pickle(file_path)
            print(f"\nReading pickle file: {file_path}")
            print("\nData type:", type(data))
            
            if isinstance(data, dict):
                print("\nDictionary keys:")
                print(list(data.keys()))
                print("\nDictionary contents:")
                for key, value in data.items():
                    print(f"\n{key}:")
                    if isinstance(value, pd.DataFrame):
                        print("  Type: DataFrame")
                        print("  Shape:", value.shape)
                        print("  Columns:", value.columns.tolist())
                        if len(value) > 0:
                            print("  First few rows:")
                            print(value.head())
                    elif isinstance(value, (list, tuple)):
                        print("  Type:", type(value).__name__)
                        print("  Length:", len(value))
                        if len(value) > 0:
                            print("  First few items:", value[:3])
                    elif hasattr(value, 'shape'):
                        print("  Type:", type(value).__name__)
                        print("  Shape:", value.shape)
                    else:
                        print("  Type:", type(value).__name__)
                        print("  Value:", value)
            elif isinstance(data, pd.DataFrame):
                print("\nDataFrame info:")
                print("  Shape:", data.shape)
                print("  Columns:", data.columns.tolist())
                print("\nFirst few rows:")
                print(data.head())
            else:
                print("\nData info:")
                print("  Type:", type(data).__name__)
                if hasattr(data, 'shape'):
                    print("  Shape:", data.shape)
                elif hasattr(data, '__len__'):
                    print("  Length:", len(data))
                print("  Content preview:", str(data)[:200] + "..." if len(str(data)) > 200 else str(data))
                
    except Exception as e:
        print(f"Error reading pickle file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 