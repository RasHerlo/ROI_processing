import pandas as pd
import os

file_path = r'C:\Users\rasmu\Desktop\TEMP LOCAL FILES\LED 2s\selected_traces.pkl'

try:
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        data = pd.read_pickle(file_path)
        print("\nData type:", type(data))
        print("\nDictionary keys:")
        print(data.keys())
        print("\nDictionary contents:")
        for key, value in data.items():
            print(f"\n{key}:")
            if isinstance(value, pd.DataFrame):
                print("DataFrame columns:", value.columns.tolist())
            else:
                print("Type:", type(value))
                print("Shape/Size:", getattr(value, 'shape', len(value)))
except Exception as e:
    print(f"Error reading pickle file: {str(e)}") 