import numpy as np
import pandas as pd
import os
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from . import utils

logger = utils.get_logger(level='DEBUG')

def transform(x):
    """
    Transform EEG data shape from (a, b, c) to (a, c, b) where:
        - a: number of EEG signal recordings (epochs)
        - b: length of each signal in terms of timesteps (samples)
        - c: number of channels
    """
    return np.transpose(x, (0, 2, 1))

def handle_nans(data):
    """
    Replace NaN values in the data with a small number.

    :param data: numpy array containing EEG data
    :return: numpy array with NaNs replaced
    """
    data[np.isnan(data)] = 1e-10
    return data

def get_spectral_features(mneData, plot=False):
    """
    Compute spectral features from EEG data.

    :param mneData: EEG data object containing the EEG signals
    :return: list of spectral features for the last frequency band
    """
    X = []

    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta":  [15.5, 30.0],
        "bad":   [30.0, 60.0],
    }

    spectrum = mneData.compute_psd(picks="eeg", fmin=0.5, fmax=50.0, verbose="ERROR")

    psds, freqs = spectrum.get_data(return_freqs=True)
    sum_psds = np.sum(psds, axis=-1, keepdims=True) + 1e-10
    psds /= sum_psds

    if plot:
        spectrum.plot(average=True, picks="eeg", exclude="bads", amplitude=False)

    for fmin, fmax in FREQ_BANDS.values():

        mask = (freqs >= fmin) & (freqs < fmax)
        selected_psds = psds[:, mask]

        psds_band = selected_psds.sum(axis=-1)
        
        X.append(psds_band.reshape(len(psds), -1))

    X = np.array(X).reshape(6,2)

    return X[-1].tolist() # a list like [a,b] of length 2

def count_muscle_artifacts(mneData, plot=False):
    """
    Count the number of muscle artifacts in EEG data using z-score annotations.

    :param mneData: EEG data object containing the EEG signals
    :return: int number of muscle artifacts detected
    """
    threshold_muscle = 5
    annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
                            mneData, ch_type="eeg",
                            threshold=threshold_muscle, min_length_good=0.2,
                            filter_freq=[110, 140], verbose="ERROR" )
    
    if plot:
        _, ax = plt.subplots()

        ax.plot(mneData.times, scores_muscle)
        ax.axhline(y=threshold_muscle, color="r")
        ax.set(xlabel="Time (s)", ylabel="Z-Score", title="Muscle Activity")

        path = utils.get_path('static', filename='muscle_activity_plot.png')
        plt.savefig(path)
        plt.close()

    bad_n = annot_muscle.count().get("BAD_muscle", 0)
    return bad_n

def extract_features(npz_obj):
    """
    Extract features from EEG data.

    :param npz_obj: numpy array containing EEG data
    :return: tuple containing feature vectors and feature names
    """
    feature_vectors = []

    fs = npz_obj["fs"][0,0]
    X = transform(npz_obj["x"])

    num_epochs = X.shape[0]
    info = mne.create_info(["AF7","AF8"], fs, ch_types="eeg", verbose="ERROR")

    for e in range(num_epochs):
        f = []

        X_without_nans = handle_nans(X[e])

        data = mne.io.RawArray(X_without_nans, info, verbose="ERROR")
        data.resample(300, npad="auto", verbose="ERROR")
        data.notch_filter([60, 120], picks="eeg", verbose="ERROR")

        spectral_features = get_spectral_features(data)
        num_muscle_artifacts = count_muscle_artifacts(data)

        f.extend(spectral_features)
        f.append(num_muscle_artifacts)

        feature_vectors.append(f)

    return np.array(feature_vectors), ["AF7HiFreq","AF8HiFreq","Artifacts"]

def create_df():
    """
    Extracts feature vectors and labels from all .npz files in the specified directory 
    and combines them into a dataframe.
    
    :param dir: directory containing .npz files
    :return: tuple with features and labels
    """
    features = []
    labels = []

    npz_dir = utils.get_dir('data', 'npz')
    
    for file in os.listdir(npz_dir):
        if file.endswith(".npz"):
            npz_path = utils.get_path(npz_dir, filename=file)
            npz_obj = np.load(npz_path)

            vectors, names = extract_features(npz_obj)

            features.append(vectors)
            labels.append(npz_obj['y'][:, -1])

    df = pd.DataFrame(np.vstack(features), columns=names)
    df['Consensus'] = np.hstack(labels)

    csv_path = utils.get_path('data', 'csv', filename='data.csv')
    df.to_csv(csv_path, index=False)
    logger.info("Feature extraction completed!")

    return df

def main():
    df = create_df()

if __name__ == "__main__":
    main()