import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split

from . import utils

logger = utils.get_logger(level='DEBUG')

def load_file(path):
    with np.load(path, allow_pickle=True) as data:
        X = data['x']
        y = data['y']
        fs = data['fs']
        label = data['label']

    return X, y, fs, label

def combine_data(dir):
    X, y = [], []

    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        X_np, y_np, _, _ = load_file(path)
        
        X.append(X_np)
        y.append(y_np[:, 3])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    logger.info(f"Combined data shapes: X={X.shape}, y={y.shape}")
    
    return X, y

def split_data(X, y, val_size=0.1, test_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_size + test_size, random_state=42)
    
    new_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=new_test_size, random_state=42)

    datasets = {'Training': (X_train, y_train), 'Validation': (X_val, y_val), 'Testing': (X_test, y_test)}

    for ds_name, (X_ds, y_ds) in datasets.items():
        logger.info(f"{ds_name} data shape: X={X_ds.shape}, y={y_ds.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_dataset(datasets):
    save_data_dir = utils.get_dir('data', 'proccessed')

    for ds_name, (X, y) in datasets.items():
        path = os.path.join(save_data_dir, f'{ds_name}.npz')
        np.savez(path, x=X, y=y)

def main():
    raw_data_dir = utils.get_dir('data', 'raw')

    X, y = combine_data(dir=raw_data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    save_dataset(datasets={'train': (X_train, y_train),
                           'val': (X_val, y_val),
                           'test': (X_test, y_test)})

if __name__ == "__main__":
    main()