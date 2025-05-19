# ==================== data_loader.py ====================
# Loading dataset from CSV files

# data_loader.py
import os
import pandas as pd

def load_toniot_dataset(data_dir):
    """
    Load all CSVs from TON_IoT directory, concatenate into a single DataFrame.
    Assumes files named '*.csv' under data_dir.
    """
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            path = os.path.join(data_dir, fname)
            dfs.append(pd.read_csv(path))
    data = pd.concat(dfs, ignore_index=True)
    # Assume last column is label
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
