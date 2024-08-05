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

        for epoch in range(num_epochs):
            data = X[epoch]
            df = pd.DataFrame(data, columns=labels)
            df['Consensus'] = y[epoch, 3]

            dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined data frame shape: {df.shape}")

    return df

def create_dataframes(train_paths, val_paths, test_paths):
    """
    Create dataframes for training, validation, and testing.

    :param train_paths: list of training file paths
    :param val_paths: list of validation file paths
    :param test_paths: list of test file paths
    :return: tuple of dataframes
    """
    dataframes = []
    names = ['train', 'val', 'test']
    all_paths = [train_paths, val_paths, test_paths]

    for paths, name in zip(all_paths, names):
        df = combine_data(paths)
        df.to_csv(f"{name}.csv", index=False)
        dataframes.append(df)

    logger.info("Saved training, validation, and testing data to CSV files!")
    return tuple(dataframes)

def main():
    npz_dir = utils.get_dir('data', 'npz')

    data = split_data(dir=npz_dir, train_size=57, val_size=1, test_size=1)

    train_df, val_df, test_df = create_dataframes(*data)

if __name__ == "__main__":
    main()