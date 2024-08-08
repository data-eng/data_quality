import torch
import multiprocessing
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

from .. import utils

logger = utils.get_logger(level='DEBUG')

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

        logger.info(f"Initializing dataset with seq_len={seq_len}, samples={self.num_samples}, sequences={self.num_seqs}")

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

        X, y = torch.FloatTensor(X), torch.LongTensor(y)

        return X, y
    
    @property
    def num_samples(self):
        return self.X.shape[0]

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
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1

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
    logger.info(f"Found {len(paths)} files in directory: {dir} ready for splitting.")

    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:train_size + val_size + test_size]

    logger.info(f"Splitting complete!")

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

def combine_data(paths, rate):
    """
    Combine data from multiple npz files into a dataframe.

    :param paths: list of file paths to npz files
    :return: dataframe
    """
    dataframes = []

    logger.info(f"Combining data from {len(paths)} files.")

    for path in paths:
        X, y, _, labels = load_file(path)

        num_epochs = X.shape[0]
        samples_per_epoch = X.shape[1]
        label_names = [item[0] for sublist in labels for item in sublist]

        for epoch in range(num_epochs):
            df = pd.DataFrame(X[epoch], columns=label_names)

            df['Consensus'] = y[epoch, 3]
            df['Time'] = np.arange(1, samples_per_epoch + 1)

            df['ID'] = (df['Time'] - 1) // rate + 1

            dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    df = utils.normalize(df, exclude=['Consensus', 'Time', 'ID'])

    return df

def get_dataframes(paths, rate=240, exist=False):
    """
    Create or load dataframes for training, validation, and testing.

    :param paths: list of training, validation and test file paths
    :param exist: whether dataframe csvs already exist
    :return: tuple of dataframes
    """
    dataframes = []
    names = ['train', 'val', 'test']

    logger.info("Creating dataframes for training, validation, and testing.")

    for paths, name in zip(paths, names):
        csv_path = utils.get_path('data', 'csv', filename=f"{name}.csv")

        if exist:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded existing dataframe from {csv_path}.")
        else:
            df = combine_data(paths, rate)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved new dataframe to {csv_path}.")

        dataframes.append(df)

    logger.info("Dataframes for training, validation, and testing are ready!")

    return tuple(dataframes)

def extract_weights(df, label_col):
    """
    Calculate class weights from the training dataframe to handle class imbalance.

    :param df: dataframe containing the training data
    :param label_col: the name of the column containing class labels
    :return: dictionary
    """
    logger.info("Calculating class weights from the training dataframe.")

    occs = df[label_col].value_counts().to_dict()
    inverse_occs = {int(key): 1 / value for key, value in occs.items()}

    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    path = utils.get_path('data', filename='weights.json')
    utils.save_json(data=weights, filename=path)

    return weights

def create_datasets(dataframes, seq_len=7680):
    """
    Create datasets for the specified dataframes, e.g. training, validation and testing, or a subset of those.

    :param dataframes: tuple of dataframes
    :param seq_len: length of the input sequence
    :return: tuple of datasets
    """

    datasets = []
    X, t, y = ["HB_1", "HB_2"], ["Time"], ["Consensus"]

    logger.info("Creating datasets from dataframes.") 

    for df in dataframes:
        dataset = TSDataset(df, seq_len, X, t, y)
        datasets.append(dataset)

    logger.info(f"Datasets created successfully!")

    return tuple(datasets)

def create_dataloaders(datasets, batch_size=8, num_workers=None, shuffle=[True, False, False]):
    """
    Create dataloaders for the specified datasets, e.g. training, validation and testing, or a subset of those.

    :param datasets: tuple of datasets
    :param batch_size: batch size for the dataloaders
    :param num_workers: number of subprocesses to use for data loading
    :param shuffle: whether to shuffle the data
    :return: tuple of dataloaders
    """
    dataloaders = []
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    logger.info(f"System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.")
    
    for dataset, shuffle in zip(datasets, shuffle):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        dataloaders.append(dataloader)

    logger.info("DataLoaders created successfully.")

    return tuple(dataloaders)