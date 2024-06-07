import numpy
import mne
import mne.preprocessing
import matplotlib.pyplot as plt

def data_transform(x):
    """
    Transform EEG data shape from (a,b,c) to (a,c,b) where:
        - a: number of EEG siognal recordings
        - b: length of each signal in terms of timesteps
        - c: number of channels
    
    :param x: numpy array (a,b,c)
    :return: numpy array (a,c,b)
    """
    return numpy.transpose(x, (0, 2, 1))

def count_muscle_artifacts(mneData, plot=False):
    """
    Count the number of muscle artifacts in EEG data using z-score annotations.

    :param mneData: EEG data object containing the EEG signals
    :return: int number of muscle artifacts detected
    """
    threshold_muscle = 5  # z-score
    annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
                            mneData, ch_type="eeg",
                            threshold=threshold_muscle, min_length_good=0.2,
                            filter_freq=[110, 140], verbose="ERROR" )
    
    if plot:
        _, ax = plt.subplots()
        ax.plot(mneData.times, scores_muscle)
        ax.axhline(y=threshold_muscle, color="r")
        ax.set(xlabel="Time (s)", ylabel="Z-Score", title="Muscle Activity")
        plt.savefig("muscle_activity_plot.png")
        plt.close()

    bad_n = annot_muscle.count().get("BAD_muscle", 0)
    return bad_n

def spectral_features(mneData):
    """
    Compute spectral features from EEG data.

    :param mneData: EEG data object containing the EEG signals
    :return: list of spectral features for the last frequency band
    """
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta":  [15.5, 30.0],
        "bad":   [30.0, 60.0],
    }

    spectrum = mneData.compute_psd( picks="eeg", fmin=0.5, fmax=50.0, verbose="ERROR" )
    psds, freqs = spectrum.get_data( return_freqs=True )
    psds /= numpy.sum( psds, axis=-1, keepdims=True )

    spectrum.plot(average=True, picks="eeg", exclude="bads", amplitude=False)

    X = []  # (num_bands x 2 channels x 1)
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].sum(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    X = numpy.array( X ).reshape(6,2) # get rid of the unneeded depth - only used when there are epochs

    return X[-1].tolist()

def features(npz_object):
    """
    Extract features from EEG data.

    :param npz_object: numpy array containing EEG data
    :return: tuple containing feature vectors and feature names
    """
    fs = npz_object["fs"][0,0]  # shape is (1,l) and holds the sampling rate (integer Hz)
    x = data_transform(npz_object["x"])
    info = mne.create_info(["AF7","AF8"], fs, ch_types="eeg", verbose="ERROR")

    feature_vectors = []
    for i in range(x.shape[0]):
        f = []
        data = mne.io.RawArray(x[i], info, verbose="ERROR")
        data.resample(300, npad="auto", verbose="ERROR")
        data.notch_filter([60, 120], picks="eeg", verbose="ERROR")
        f.extend(spectral_features(data))
        f.append(count_muscle_artifacts(data))
        feature_vectors.append(f)

    return numpy.array(feature_vectors), ["AF7HiFreq","AF8HiFreq","Artifacts"]