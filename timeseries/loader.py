import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
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
    logger.debug(f"Found {len(paths)} files in directory: {dir} ready for splitting.")

    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:train_size + val_size + test_size]

    logger.debug(f"Splitting complete!")

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

    logger.debug(f"Combining data from {len(paths)} files.")

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

    logger.debug("Creating dataframes for training, validation, and testing.")

    for paths, name in zip(all_paths, names):
        csv_path = utils.get_path('data', 'csv', filename=f"{name}.csv")

        if exist:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded existing dataframe from {csv_path}.")
        else:
            df = combine_data(paths)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved new dataframe to {csv_path}.")

        dataframes.append(df)

    logger.info("Dataframes for training, validation, and testing are ready!")

    return tuple(dataframes)

def create_datasets(train_df, val_df, test_df, seq_len=7680):
    """
    Create datasets for training, validation, and testing.

    :param train_df: training dataframe
    :param val_df: validation dataframe
    :param test_df: testing dataframe
    :param seq_len: length of the input sequence
    :return: tuple of datasets
    """

    datasets = []
    X, t, y = ["HB_1", "HB_2"], ["Time"], ["Consensus"]
    dataframes = [train_df, val_df, test_df]

    logger.debug("Creating datasets from dataframes.")

    class TSDataset(Dataset):
        def __init__(self, df, seq_len, X, t, y, per_epoch=True):
            """
            Initializes a time series dataset.

            :param df: dataframe
            :param seq_len: length of the input sequence
            :param X: list of feature columns
            :param t: list of time columns
            :param y: list of target columns
            :param per_epoch: whether to create sequences with overlapping epochs or not
            """
            self.seq_len = seq_len
            self.X = pd.concat([df[X], df[t]], axis=1)
            self.y = df[y]
            self.per_epoch = per_epoch
            self.num_samples = self.X.shape[0]

            logger.debug(f"Initializing dataset with seq_len={seq_len}, samples={self.num_samples}, sequences={self.num_seqs}")

        def __len__(self):
            """
            :return: length of the dataset
            """
            return self.num_seqs

        def __getitem__(self, idx):
            """
            Retrieves a sample from the dataset at the specified index.

            :param idx: index of the sample
            :return: tuple of features and target tensors
            """
            if self.per_epoch:
                start_idx = idx * self.seq_len
            else:
                start_idx = idx

            end_idx = start_idx + self.seq_len

            X = self.X.iloc[start_idx:end_idx].values
            y = self.y.iloc[start_idx:end_idx].values

            return torch.FloatTensor(X), torch.FloatTensor(y)
        
        @property
        def max_seq_id(self):
            """
            :return: maximum index for a sequence
            """
            return self.num_samples - self.seq_len
        
        @property
        def num_seqs(self):
            """
            :return: number of sequences that can be created from the dataset
            """
            if self.per_epoch:
                return self.max_seq_id + 1
            else:
                return self.num_samples // self.seq_len 

    for df in dataframes:
        dataset = TSDataset(df, seq_len, X, t, y)
        datasets.append(dataset)

        logger.info(f"Dataset created with {len(dataset)} sequences!")

    return tuple(datasets)