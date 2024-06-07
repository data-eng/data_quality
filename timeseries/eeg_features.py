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

def spectral_features(mneData, plot=False):
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

    if plot:
        spectrum.plot(average=True, picks="eeg", exclude="bads", amplitude=False)

    X = []  # (num_bands x 2 channels x 1)
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].sum(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    # Shape is (bands x channels x 1)
    # Depth is always 1 because we have only one epoch
    # Flatten to a feature vector in the shape shown in the feature_names
    # array below.
    # NOTE: if the code here changes, amend feature_names accordingly.
    return numpy.array( X ).reshape(1,12)


def features(npz_object):
    """
    Extract features from EEG data.

    :param npz_object: numpy array containing EEG data
    :return: tuple containing feature vectors, list of feature names, list of feature vector indexes that are relevant to quality estimation, and list of feature vector indexes that are relevant to classification 
    """
    fs = npz_object["fs"][0,0]  # shape is (1,l) and holds the sampling rate (integer Hz)
    x = data_transform(npz_object["x"])
    info = mne.create_info(["AF7","AF8"], fs, ch_types="eeg", verbose="ERROR")

    feat1 = None
    feat2 = []
    for i in range(x.shape[0]):
        data = mne.io.RawArray(x[i], info, verbose="ERROR")
        data.resample(300, npad="auto", verbose="ERROR")
        data.notch_filter([60, 120], picks="eeg", verbose="ERROR")
        if feat1 is None: feat1 = spectral_features(data)
        else: feat1 = numpy.append(feat1,spectral_features(data),axis=0)
        feat2.append(count_muscle_artifacts(data))
    a=numpy.expand_dims(numpy.array(feat2), axis=1)
    feature_vectors = numpy.concatenate((feat1,a), axis=1)

    feature_names = ["PSD_AF7DeltaBand","PSD_AF8DeltaBand",
                     "PSD_AF7_ThetaBand","PSD_AF8_ThetaBand",
                     "PSD_AF7_AlphaBand","PSD_AF8_AlphaBand",
                     "PSD_AF7_SigmaBand","PSD_AF8_SigmaBand",
                     "PSD_AF7_BetaBand","PSD_AF8_BetaBand",
                     "PSD_AF7_HighFreq","PSD_AF8_HighFreq",
                     "Artifacts"]
    return feature_vectors, feature_names, [10,11,12], [0,1,2,3,4,5,6,7,8,9]

