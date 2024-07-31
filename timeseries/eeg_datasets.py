import sys
import numpy as np
import pandas as pd
import os
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yasa
import csv
import librosa
import librosa.display

import annotator_agreement

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_files(directory):
    contents = os.listdir(directory)
    files = [os.path.join(directory, name) for name in contents if
             os.path.isfile(os.path.join(directory, name))]
    return files


def create_symbolic_dataset():

    sequence_length = 30

    class_mapping = {0: 'wake', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'rem'}

    symb_data_path = 'timeseries/data/symbolic_data/'
    create_directory(symb_data_path)

    test_perc = 0.2

    class_counter = {
        'train': {'0': 0, '1': 0},
        'test': {'0': 0, '1': 0}
    }

    variable_path = sys.argv[1]

    # Check if the variable is a directory
    if os.path.isdir(variable_path):
        files = get_files(variable_path)
    else:  # It is a single file
        files = [variable_path]

    sequence_number = 1
    for file in files:

        npz = np.load(file, allow_pickle=True)
        quality = annotator_agreement.quality(npz)
        class_labels = np.where(quality == 1.0, 1, 0)

        epoch_size = npz['x'].shape[0]
        # logger.info(epoch_size, epoch_size//sequence_length)

        for epoch in range(0, epoch_size, sequence_length):

            start_idx = epoch
            end_idx = epoch + sequence_length
            # sleep_sequence = npz['x'][start_idx:end_idx]
            sleep_labels = npz['y'][start_idx:end_idx][:, 3]

            class_labels_seq = class_labels[start_idx:end_idx]
            class_seq = 0 if sum(class_labels_seq) == len(class_labels_seq) else 1

            symb_sequence = []
            for i in range(len(sleep_labels)):
                symb_sequence.append(f'seq({sequence_number}, '
                                     f'sleep_cycle({class_mapping[sleep_labels[i]]}), '
                                     f'{i+1})')

            symb_sequence.append(f'class({sequence_number}, {class_seq})')

            sequence_txt = ". ".join(symb_sequence)
            sequence_txt += "."

            if random.random() < test_perc:
                set_path = os.path.join(symb_data_path, "test.csv")
                class_counter['test'][str(class_seq)] += 1
            else:
                set_path = os.path.join(symb_data_path, "train.csv")
                class_counter['train'][str(class_seq)] += 1

            row_data = sequence_txt.split(',')

            with open(set_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            '''agent_type(1, agent1(ped)). agent_type(1, agent2(cyc)). class(1, 0). 
            seq(1, a1(action(movtow)), 1). seq(1, a1(action(movtow)), 2). seq(1, a1(action(movtow)), 3). class(1, 0).'''

        sequence_number += 1


def plot_spectrograms():

    os.makedirs('timeseries/data/spectrograms_yasa', exist_ok=True)
    variable_path = 'data/raw_data'

    # Check if the variable is a directory
    assert os.path.isdir(variable_path)

    files = get_files(variable_path)

    counter = 0

    for file in files:

        npz = np.load(file, allow_pickle=True)
        quality = annotator_agreement.quality(npz)
        # class_labels = quality
        class_labels = np.where(quality == 1.0, 1, 0)

        for epoch in range(len(class_labels)):
            # For both channels
            for channel in [0, 1]:
                # Get data and sampling frequency (rate)
                data = npz['x'][epoch, :, channel]
                sf = npz['fs'].squeeze().item()
                max_win_sec = len(data) / (2 * sf)

                fig = yasa.plot_spectrogram(data, sf, win_sec=max_win_sec // 15)

                # Customize the plot to show time in seconds
                plt.xlabel('Time (seconds)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Spectrogram')

                # Customize the x-axis to display time in seconds
                ax = fig.gca()
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 3600:.1f}'))

                plt.savefig(f'./timeseries/data/spectrograms_yasa/spectrogram_{counter}_channel_{channel+1}.png',
                            dpi=300, bbox_inches='tight')
                plt.clf()

            counter += 1


def create_spectrograms():

    logger.info('Start creating the dataset..')

    plot_librosa = False
    test_set_freq = 0.2

    def clean_data(eeg_data):
        # Replace NaN with 0
        return np.nan_to_num(eeg_data)

    dataset = {
        'train': {'name': [], 'epoch': [], 'channel': [], 'raw_data': [], 'spectrogram': [],
                  'spectrogram_db': [], 'label': []},
        'test': {'name': [], 'epoch': [], 'channel': [], 'raw_data': [], 'spectrogram': [],
                 'spectrogram_db': [], 'label': []}
    }

    variable_path = 'timeseries/data/raw_data'

    # Check if the variable is a directory
    assert os.path.isdir(variable_path)

    files = get_files(variable_path)

    for file in files:
        # Use some samples for training and other for testing.
        # Make sure different full samples are in training and different in testing set.
        set_type = 'test' if random.random() <= test_set_freq else 'train'

        npz = np.load(file, allow_pickle=True)
        quality = annotator_agreement.quality(npz)
        # class_labels = quality
        class_labels = np.where(quality == 1.0, 1, 0)

        for epoch in range(len(class_labels)):
            # For both channels
            for channel in [0, 1]:
                # Get data and sampling frequency (rate)
                data = clean_data(eeg_data=npz['x'][epoch, :, channel])
                sf = npz['fs'].squeeze().item()

                # Compute the Short-Time Fourier Transform (STFT)
                stft_result = librosa.stft(data)
                # Convert the STFT to a spectrogram (magnitude)
                spectrogram = np.abs(stft_result)
                # Convert the spectrogram to a decibel (dB) scale for better visualization
                spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

                sample_name = f'{file[len(variable_path)+2:-4]}'

                set_dictionary = dataset['train'] if set_type == 'train' else dataset['test']

                set_dictionary['name'].append(sample_name)
                set_dictionary['epoch'].append(epoch)
                set_dictionary['channel'].append(channel)
                set_dictionary['raw_data'].append(data)
                set_dictionary['spectrogram'].append(spectrogram)
                set_dictionary['spectrogram_db'].append(spectrogram_db)
                set_dictionary['label'].append(class_labels[epoch])

                if plot_librosa:
                    os.makedirs('timeseries/data/spectrograms_librosa', exist_ok=True)
                    # plt.figure(figsize=(10, 6))
                    librosa.display.specshow(spectrogram_db, sr=sf, x_axis='time', y_axis='log', cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Spectrogram (dB)')
                    plt.xlabel('Time')
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(f'timeseries/data/spectrograms_librosa/{sample_name}.png', dpi=300, bbox_inches='tight')
                    plt.clf()

    # Create and save the DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    logger.info(train_df.shape, test_df.shape)

    processed_data_pth = 'timeseries/data/processed_data'
    os.makedirs('data/processed_data', exist_ok=True)
    train_df.to_csv(f'{processed_data_pth}/train_data.csv', index=False)
    test_df.to_csv(f'{processed_data_pth}/test_data.csv', index=False)

    logger.info('Dataset created..')
