import numpy
import mne
import mne.preprocessing


def data_transform( x ):
    # In our dataset, shape is (n,len,2), where each row (epoch)
    # has a signal of length len, with two channels for each timestep.
    # Re-shape into (n,2,len) to feed into mne
    return numpy.transpose( x, (0, 2, 1) )


def count_muscle_artifacts( mneData ):
    threshold_muscle = 5  # z-score
    annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
        mneData, ch_type="eeg",
        threshold=threshold_muscle, min_length_good=0.2,
        filter_freq=[110, 140], verbose="ERROR" )
    #fig, ax = plt.subplots()
    #ax.plot( data.times, scores_muscle )
    #ax.axhline(y=threshold_muscle, color="r")
    #ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")

    # Number of samples above threshold_muscle
    # Why can't people always fill the dict, even for zero?
    try: bad_n = annot_muscle.count()["BAD_muscle"]
    except KeyError: bad_n = 0
    return bad_n


def spectral_features( mneData ):
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
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].sum(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    # X.shape is num_bands x 2 channels x 1
    # get rid of the unneeded depth
    # (I think it's only used when there are epochs)
    X = numpy.array( X ).reshape(6,2)
    # return the last band (two values)
    return X[-1].tolist()
    #spectrum.plot( average=True, picks="eeg", exclude="bads", amplitude=False )


def features( npz_object ):
    fs = npz_object["fs"][0,0]
    # shape is (1,l) and holds the sampling rate (integer Hz)
    x = data_transform( npz_object["x"] )
    info = mne.create_info( ["AF7","AF8"], fs, ch_types="eeg", verbose="ERROR" )
    feature_vectors = []
    for i in range(x.shape[0]):
        f = []
        data = mne.io.RawArray( x[i], info, verbose="ERROR" )
        data.resample( 300, npad="auto", verbose="ERROR" )
        data.notch_filter( [60, 120], picks="eeg", verbose="ERROR" )
        f.extend( spectral_features(data) )
        f.append( count_muscle_artifacts(data) )
        feature_vectors.append( f )
    return numpy.array( feature_vectors ), ["AF7HiFreq","AF8HiFreq","Artifacts"]
