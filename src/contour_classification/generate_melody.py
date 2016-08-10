""" Module for generating melody output based on classifier scores """
import pandas as pd
import contour_utils as cc
import numpy as np
import mir_eval


def melody_from_clf(contour_data, prob_thresh=0.5, penalty=0, method='viterbi'):
    """ Compute output melody using classifier output.

    Parameters
    ----------
    contour_data : DataFrame or dict of DataFrames
        DataFrame containing labeled features.
    prob_thresh : float
        Threshold that determines positive class

    Returns
    -------
    mel_output : Series
        Pandas Series with time stamp as index and f0 as values
    """

    contour_threshed = contour_data[contour_data['mel prob'] >= prob_thresh]

    if len(contour_threshed) == 0:
        print "Warning: no contours above threshold."
        contour_times, _, _ = \
            cc.contours_from_contour_data(contour_data, n_end=4)

        hopsizeInSamples = 256.0
        step_size = hopsizeInSamples/44100.0  # contour time stamp step size
        mel_time_idx = np.arange(0, np.max(contour_times.values.ravel()) + 1,
                                 step_size)
        mel_output = pd.Series(np.zeros(mel_time_idx.shape),
                               index=mel_time_idx)
        return mel_output

    # get separate DataFrames of contour time, frequency, and probability
    contour_times, contour_freqs, _ = \
        cc.contours_from_contour_data(contour_threshed, n_end=4)

    # make frequencies below probability threshold negative
    #contour_freqs[contour_data['mel prob'] < prob_thresh] *= -1.0

    probs = contour_threshed['mel prob']
    contour_probs = pd.concat([probs]*contour_times.shape[1], axis=1,
                              ignore_index=True)

    contour_num = pd.DataFrame(np.array(contour_threshed.index))
    contour_nums = pd.concat([contour_num]*contour_times.shape[1], axis=1,
                             ignore_index=True)

    avg_freq = contour_freqs.mean(axis=1)

    # create DataFrame with all unwrapped [time, frequency, probability] values.
    mel_dat = pd.DataFrame(columns=['time', 'f0', 'probability', 'c_num'])
    mel_dat['time'] = contour_times.values.ravel()
    mel_dat['f0'] = contour_freqs.values.ravel()
    mel_dat['probability'] = contour_probs.values.ravel()
    mel_dat['c_num'] = contour_nums.values.ravel()

    # remove rows with NaNs
    mel_dat.dropna(inplace=True)

    # sort by probability then by time
    # duplicate times with have maximum probability value at the end
    mel_dat.sort(columns='probability', inplace=True)
    mel_dat.sort(columns='time', inplace=True)

    hopsizeInSamples = 256.0
    # compute evenly spaced time grid for output
    step_size = hopsizeInSamples/44100.0  # contour time stamp step size
    mel_time_idx = np.arange(0, np.max(mel_dat['time'].values) + 1, step_size)

    # find index in evenly spaced grid of estimated time values
    old_times = mel_dat['time'].values
    reidx = np.searchsorted(mel_time_idx, old_times)
    shift_idx = (np.abs(old_times - mel_time_idx[reidx - 1]) < \
                 np.abs(old_times - mel_time_idx[reidx]))
    reidx[shift_idx] = reidx[shift_idx] - 1

    # find duplicate time values
    mel_dat['reidx'] = reidx

    if method == 'max':
        print "using max decoding"
        mel_dat.drop_duplicates(subset='reidx', take_last=True, inplace=True)

        mel_output = pd.Series(np.zeros(mel_time_idx.shape), index=mel_time_idx)
        mel_output.iloc[mel_dat['reidx']] = mel_dat['f0'].values

    else:
        print "using viterbi decoding"
        duplicates = mel_dat.duplicated(subset='reidx') | \
                     mel_dat.duplicated(subset='reidx', take_last=True)

        not_duplicates = mel_dat[~duplicates]

        # initialize output melody
        mel_output = pd.Series(np.zeros(mel_time_idx.shape), index=mel_time_idx)

        # fill non-duplicate values
        mel_output.iloc[not_duplicates['reidx']] = not_duplicates['f0'].values

        dups = mel_dat[duplicates]
        dups['groupnum'] = (dups.loc[:, 'reidx'].diff() > 1).cumsum().copy()
        groups = dups.groupby('groupnum')

        for _, group in groups:
            states = np.unique(group['c_num'])
            center_freqs = avg_freq.loc[states]
            times = np.unique(group['reidx'])

            posterior = group[['probability', 'c_num', 'reidx']].pivot_table(
                'probability', index='reidx',
                columns='c_num',
                fill_value=0.0).as_matrix()

            f0_vals = group[['f0', 'c_num', 'reidx']].pivot_table(
                'f0', index='reidx',
                columns='c_num',
                fill_value=0.0).as_matrix()

            #posterior[np.where(f0_vals < prob_thresh)] = 0 #1e-10

            # build transition matrix from log distance between center frequency
            transition_matrix = np.log2(center_freqs.values)[np.newaxis, :] - \
                                np.log2(center_freqs.values)[:, np.newaxis]
            transition_matrix = 1 - normalize(np.abs(transition_matrix), axis=1)
            transition_matrix = normalize(transition_matrix, axis=1)

            path = viterbi(posterior, transition_matrix=transition_matrix,
                           prior=None, penalty=penalty)

            mel_output.iloc[times] = f0_vals[np.arange(len(path)), path]

    return mel_output


def score_melodies(mel_output_dict, test_annot_dict):
    """ Score melody output against ground truth.

    Parameters
    ----------
    mel_output_dict : dict of Series
        Dictionary of melody output series keyed by trackid
    test_annot_dict : dict of DataFrames
        Dictionary of DataFrames containing annotations.

    Returns
    -------
    melody_scores : dict
        melody evaluation metrics for each track
    """
    melody_scores = {}
    print "Scoring..."
    for key in mel_output_dict.keys():
        print key
        if mel_output_dict[key] is None:
            print "skipping..."
            continue
        ref = test_annot_dict[key]
        est = mel_output_dict[key]
        if isinstance(est,pd.DataFrame) or isinstance(est,pd.Series):
            melody_scores[key] = mir_eval.melody.evaluate(ref['time'].values,
                                                      ref['f0'].values,
                                                      est.index.values,
                                                      est.values)
        else:
            times, pitches = est
            melody_scores[key] = mir_eval.melody.evaluate(ref['time'].values,
                                          ref['f0'].values,
                                          times,
                                          pitches[:,0])

    return melody_scores


def viterbi(posterior, transition_matrix=None, prior=None, penalty=0,
            scaled=True):
    """Find the optimal Viterbi path through a posteriorgram.
    Ported closely from Tae Min Cho's MATLAB implementation.
    Parameters
    ----------
    posterior: np.ndarray, shape=(num_obs, num_states)
        Matrix of observations (events, time steps, etc) by the number of
        states (classes, categories, etc), e.g.
          posterior[t, i] = Pr(y(t) | Q(t) = i)
    transition_matrix: np.ndarray, shape=(num_states, num_states)
        Transition matrix for the viterbi algorithm. For clarity, each row
        corresponds to the probability of transitioning to the next state, e.g.
          transition_matrix[i, j] = Pr(Q(t + 1) = j | Q(t) = i)
    prior: np.ndarray, default=None (uniform)
        Probability distribution over the states, e.g.
          prior[i] = Pr(Q(0) = i)
    penalty: scalar, default=0
        Scalar penalty to down-weight off-diagonal states.
    scaled : bool, default=True
        Scale transition probabilities between steps in the algorithm.
        Note: Hard-coded to True in TMC's implementation; it's probably a bad
        idea to change this.
    Returns
    -------
    path: np.ndarray, shape=(num_obs,)
        Optimal state indices through the posterior.
    """

    # Infer dimensions.
    num_obs, num_states = posterior.shape

    # Define the scaling function
    scaler = normalize if scaled else lambda x: x
    # Normalize the posterior.
    posterior = normalize(posterior, axis=1)

    if transition_matrix is None:
        transition_matrix = np.ones([num_states]*2)

    transition_matrix = normalize(transition_matrix, axis=1)

    # Apply the off-axis penalty.
    offset = np.ones([num_states]*2, dtype=float)
    offset -= np.eye(num_states, dtype=np.float)
    penalty = offset * np.exp(penalty) + np.eye(num_states, dtype=np.float)
    transition_matrix = penalty * transition_matrix

    # Create a uniform prior if one isn't provided.
    prior = np.ones(num_states) / float(num_states) if prior is None else prior

    # Algorithm initialization
    delta = np.zeros_like(posterior)
    psi = np.zeros_like(posterior)
    path = np.zeros(num_obs, dtype=int)

    idx = 0
    delta[idx, :] = scaler(prior * posterior[idx, :])

    for idx in range(1, num_obs):
        res = delta[idx - 1, :].reshape(1, num_states) * transition_matrix
        delta[idx, :] = scaler(np.max(res, axis=1) * posterior[idx, :])
        psi[idx, :] = np.argmax(res, axis=1)

    path[-1] = np.argmax(delta[-1, :])
    for idx in range(num_obs - 2, -1, -1):
        path[idx] = psi[idx + 1, path[idx + 1]]
    return path


def normalize(x, axis=None):
    """Normalize the values of an ndarray to sum to 1 along the given axis.
    Parameters
    ----------
    x : np.ndarray
        Input multidimensional array to normalize.
    axis : int, default=None
        Axis to normalize along, otherwise performed over the full array.
    Returns
    -------
    z : np.ndarray, shape=x.shape
        Normalized array.
    """
    if not axis is None:
        shape = list(x.shape)
        shape[axis] = 1
        scalar = x.astype(float).sum(axis=axis).reshape(shape)
        scalar[scalar == 0] = 1.0
    else:
        scalar = x.sum()
        scalar = 1 if scalar == 0 else scalar
    return x / scalar
