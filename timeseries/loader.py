import pandas as pd
import numpy as np
import os

from . import utils

logger = utils.get_logger(level='DEBUG')

def split_data(dir, train_size=57, val_size=1, test_size=1):
    """
    Split the npz files into training, validation, and test sets.

    :param dir: directory containing the npz files
    :param train_size: number of files for training
    :param val_size: number of files for validation
    :param test_size: number of files for testing
    :return: tuple of lists containing npz file paths for train, val, and test sets
    """
    paths = [utils.get_path(dir, filename=file) for file in os.listdir(dir)]

    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:train_size + val_size + test_size]

    return (train_paths, val_paths, test_paths)

def load_file(path):
    """
    Load data from a .npz file.

    :param path: path to the .npz file
    :return: tuple
    """
    with np.load(path, allow_pickle=True) as data:
        X = data['x']
        y = data['y']
        fs = data['fs']
        label = data['label']

    assert X.shape[1] == fs * 30, f"Expected {fs * 30} samples per epoch, but got {X.shape[1]}"

    return X, y, fs, label

def combine_data(paths):
    """
    Combine data from multiple npz files into a dataframe.

    :param paths: list of file paths to npz files
    :return: dataframe
    """
    dataframes = []

    for path in paths:
        X, y, _, labels = load_file(path)
        
        num_epochs = X.shape[0]
        samples_per_epoch = X.shape[1]
        label_names = [item[0] for sublist in labels for item in sublist]

        for epoch in range(num_epochs):
            df = pd.DataFrame(X[epoch], columns=label_names)

            df['Consensus'] = y[epoch, 3]
            df['Time'] = np.arange(1, samples_per_epoch + 1)

            dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    return df

def create_dataframes(train_paths, val_paths, test_paths, exist=False):
    """
    Create or load dataframes for training, validation, and testing.

    :param train_paths: list of training file paths
    :param val_paths: list of validation file paths
    :param test_paths: list of test file paths
    :param exist: whether dataframe csvs already exist
    :return: tuple of dataframes
    """
    dataframes = []
    names = ['train', 'val', 'test']
    all_paths = [train_paths, val_paths, test_paths]

    for paths, name in zip(all_paths, names):
        csv_path = utils.get_path('data', 'csv', filename=f"{name}.csv")

        if exist:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded existing dataframe from {csv_path}")
        else:
            df = combine_data(paths)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved new dataframe to {csv_path}")

        dataframes.append(df)

    logger.info("Dataframes for training, validation, and testing are ready!")

    return tuple(dataframes)

