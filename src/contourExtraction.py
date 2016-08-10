import contour_classification.contour_utils as cu


def compute_contour_data(contours_bins, contours_saliences, contours_start_times, stepNotes, minF0, hopsize,
                         normalize=True, extra_features=None):
    from pandas import DataFrame, concat
    from numpy import mean, std, array, Inf, zeros
    """ Create contour pandas dataframe uing contour information previouslly extracted with Essentia.
    Initializes DataFrame to have all future columns.
    Parameters
    ----------
    contours_bins: set of bins of the extracted contours
    contours_saliences:  set of saliences of the extracted contours
    contours_start_times:  set of starting times of the extracted contours
    stepNotes: number of bins per semitone
    minF0: minimum F0 in the salience functions
    hopsize: Hop size
    normalize: [True, False] to normalise the features, as performed in Bittner2015
    extra_features: Ncontours * N_features
    set of extra features apart from the ones used by Bittner2015 (pitch, duration, vibrato, salience)

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with all contour data, to be used for contour classification
    """

    contours_bins = array(contours_bins)
    contours_saliences = array(contours_saliences)
    contours_start_times = array(contours_start_times)
    contour_data = DataFrame
    headers = []

    # Set of headers, containing the first 12 features [0:11] and the first time for each of the contours
    headers[0:12] = ['onset', 'offset', 'duration', 'pitch mean', 'pitch std',
                     'salience mean', 'salience std', 'salience tot',
                     'vibrato', 'vib rate', 'vib extent', 'vib coverage', 'first_time']

    # Number of contours
    Ncont = len(contours_bins)

    # Find length of longest contour
    maxLen = 0
    for i in range(Ncont):
        maxLen = max(maxLen, len(contours_bins[i]))

    # Header "first_time" can be used to find where the contour features end,
    #  and when the contour info starts (time, bin, salience)

    # Just giving the extra headers some name
    headers[13:] = (array(range(maxLen * 3))).tolist()

    contour_data.num_end_cols = 4

    # Initialising dataset, following the format from the hacked VAMP MELODIA plugin from J. Salamon
    contour_data = DataFrame(Inf * zeros([Ncont, len(headers)]), columns=headers)

    for i in range(Ncont):
        #print i
        # Giving values for each row of the dataframe
        L = len(contours_saliences[i])
        # minF0 instead of 55
        pitches = 55 * 2 ** ((array(contours_bins[i]) / (12. * stepNotes)))
        contour_data.set_value(i, 'onset', contours_start_times[i])
        contour_data.set_value(i, 'offset', array(contours_start_times[i]) + len(pitches) * hopsize)
        contour_data.set_value(i, 'duration', len(pitches) * hopsize)
        contour_data.set_value(i, 'pitch mean', mean(pitches))
        contour_data.set_value(i, 'pitch std', std(pitches))
        contour_data.set_value(i, 'salience mean', mean(array(contours_saliences[i])))
        contour_data.set_value(i, 'salience std', std(array(contours_saliences[i])))
        contour_data.set_value(i, 'salience tot', sum(array(contours_saliences[i])))

        # In this case, we do not compute vibrato features, so we set them to 0.
        # This could be updated in order to use also vibrato features from contours extracted with Essentia
        contour_data.set_value(i, 'vibrato', 0)
        contour_data.set_value(i, 'vib rate', 0)
        contour_data.set_value(i, 'vib extent', 0)
        contour_data.set_value(i, 'vib coverage', 0)

        # After setting the features, we now give each contour the frame by frame information, e.g for frame0 (fr0), frame 1 (fr1)...
        # time_fr0, pitch_fr0, salience_fr0, time_fr1, pitch_fr1, salience_fr1, time_fr2, pitch_fr2, salience_fr2, ...

        contour_data.iloc[i, 12:12 + L * 3:3] = contours_start_times[i] + hopsize * array(range(L))
        contour_data.iloc[i, 13:13 + L * 3:3] = pitches
        contour_data.iloc[i, 14:14 + L * 3:3] = array(contours_saliences[i])

    # If extra features are used, they are set before the first_time
    if extra_features is not None:
        dfFeatures = concat([contour_data.ix[:, 0:12], extra_features], axis=1)
        contour_data = concat([dfFeatures, contour_data.ix[:, 12:]], axis=1)

    # All classification labels are initialised (will be updated while performing contour classification)
    contour_data['overlap'] = -1
    contour_data['labels'] = -1
    contour_data['melodiness'] = ""
    contour_data['mel prob'] = -1

    # Normalising features
    if normalize:
        contour_data = cu.normalize_features(contour_data)

    print "Contour dataframe created"

    return contour_data
