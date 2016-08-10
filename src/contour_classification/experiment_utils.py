""" Helper functions for experiments """

from ShuffleLabelsOut import ShuffleLabelsOut
import contour_utils as cc
import json
from sklearn import metrics
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def create_splits(test_size=0.15):
    """ Split MedleyDB into train/test splits.

    Returns
    -------
    mdb_files : list
        List of sorted medleydb files.
    splitter : iterator
        iterator of train/test indices.
    """

    #index = json.load(open('medley_artist_index.json'))
    # EDIT: For Orchset
    index = json.load(open('orch_groups.json'))

    mdb_files = []
    keys = []

    for trackid, artist in sorted(index.items()):
        mdb_files.append(trackid)
        keys.append(artist)

    keys = np.asarray(keys)
    mdb_files = np.asarray(mdb_files)
    splitter = ShuffleLabelsOut(keys, random_state=1, test_size=test_size)

    return mdb_files, splitter


def get_data_files(track, meltype=1):
    """ Load all necessary data for a given track and melody type.

    Parameters
    ----------
    track : str
        Track identifier.
    meltype : int
        Melody annotation type. One of [1, 2, 3]

    Returns
    -------
    cdat : DataFrame
        Pandas DataFrame of contour data.
    adat : DataFrame
        Pandas DataFrame of annotation data.
    """
    contour_suffix = \
        "MIX_vamp_melodia-contours_melodia-contours_contoursall.csv"
    contours_path = "melodia_contours"

    # For ORCHSET with MELODIA --------------------------

    annot_path = os.path.join('/Users/jjb/Google Drive/data/segments/excerpts/GT')

    contour_suffix = \
        "_vamp_melodia-contours_melodia-contours_contoursall.csv"
    contours_path = "/Users/jjb/Google Drive/PhD/conferences/ISMIR2016/SIMM-PC/Orchset/contours_melodia"
    annot_suffix = "mel"
    contour_fname = "%s%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s.%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)


    # Fot ORCHSET with SIMM --------------------------

    contour_suffix = "pitch.ctr"
    contours_path = "/Users/jjb/Google Drive/PhD/conferences/ISMIR2016/SIMM-PC/Orchset/C4-Contours/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-50_mD-100"

    contours_path = "/Users/jjb/Google Drive/PhD/Tests/Orchset/ScContours/"

    annot_suffix = "mel"

    annot_path = os.path.join('/Users/jjb/Google Drive/data/segments/excerpts/GT')
    contour_fname = "%s.%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s.%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)

    # For MEDLEY with SIMM -------------------------
    contour_suffix = "MIX.pitch.ctr"
    contours_path = "/Users/jjb/Google Drive/PhD/conferences/ISMIR2016/SIMM-PC/MedleyDB/C4-Contours/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-50_mD-100"

    annot_suffix = "MELODY%s.csv" % str(meltype)
    mel_dir = "MELODY%s" % str(meltype)
    annot_path = os.path.join(os.environ['MEDLEYDB_PATH'], 'Annotations',
                              'Melody_Annotations', mel_dir)

    contour_fname = "%s_%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s_%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)

    # Fot ORCHSET with SIMM --------------------------

    contour_suffix = "pitch.ctr"
    contours_path = "/Users/jjb/Google Drive/PhD/conferences/ISMIR2016/SIMM-PC/Orchset/C4-Contours/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-50_mD-100"

    #contours_path = "/Users/jjb/Google Drive/PhD/Tests/Orchset/ScContours/"

    annot_suffix = "mel"

    annot_path = os.path.join('/Users/jjb/Google Drive/data/segments/excerpts/GT')
    contour_fname = "%s.%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s.%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)

    #################################################

    cdat = cc.load_contour_data(contour_fpath, normalize=True)
    adat = cc.load_annotation(annot_fpath)

    return cdat, adat


def compute_all_overlaps(track_list, meltype):
    """ Compute each contour's overlap with annotation.

    Parameters
    ----------
    track_list : list
        List of all trackids
    meltype : int
        One of [1,2,3]

    Returns
    -------
    dset_contour_dict : dict of DataFrames
        Dict of dataframes keyed by trackid
    dset_annot_dict : dict of dataframes
        dict of annotation dataframes keyed by trackid
    """

    dset_contour_dict = {}
    dset_annot_dict = {}

    msg = "Generating features..."
    num_spaces = len(track_list) - len(msg)
    print msg + ' '*num_spaces + '|'

    for track in track_list:
        cdat, adat = get_data_files(track, meltype=meltype)
        dset_annot_dict[track] = adat.copy()
        dset_contour_dict[track] = cc.compute_overlap(cdat, adat)
        sys.stdout.write('.')

    return dset_contour_dict, dset_annot_dict


def olap_stats(train_contour_dict):
    """ Compute overlap statistics.

    Parameters
    ----------
    train_contour_dict : dict of DataFrames
        Dict of train contour data frames

    Returns
    -------
    partial_olap_stats : DataFrames
        Description of overlap data.
    zero_olap_stats : DataFrames
        Description of non-overlap data.
    """
    # reduce for speed and memory
    red_list = []
    for cdat in train_contour_dict.values():
        red_list.append(cdat['overlap'])

    overlap_dat = cc.join_contours(red_list)
    non_zero_olap = overlap_dat[overlap_dat > 0]
    zero_olap = overlap_dat[overlap_dat == 0]
    partial_olap_stats = non_zero_olap.describe()
    zero_olap_stats = zero_olap.describe()

    return partial_olap_stats, zero_olap_stats


def label_all_contours(train_contour_dict, valid_contour_dict,
                       test_contour_dict, olap_thresh):
    """ Add labels to contours based on overlap_thresh.

    Parameters
    ----------
    train_contour_dict : dict of DataFrames
        dict of train contour data frames
    valid_contour_dict : dict of DataFrames
        dict of validation contour data frames
    test_contour_dict : dict of DataFrames
        dict of test contour data frames
    olap_thresh : float
        Value in [0, 1). Min overlap to be labeled as melody.

    Returns
    -------
    train_contour_dict : dict of DataFrames
        dict of train contour data frames
    test_contour_dict : dict of DataFrames
        dict of test contour data frames
    """
    for key in train_contour_dict.keys():
        train_contour_dict[key] = cc.label_contours(train_contour_dict[key],
                                                    olap_thresh=olap_thresh)

    for key in valid_contour_dict.keys():
        valid_contour_dict[key] = cc.label_contours(valid_contour_dict[key],
                                                    olap_thresh=olap_thresh)

    for key in test_contour_dict.keys():
        test_contour_dict[key] = cc.label_contours(test_contour_dict[key],
                                                   olap_thresh=olap_thresh)
    return train_contour_dict, valid_contour_dict, test_contour_dict


def contour_probs(clf, contour_data,idxStartFeatures=0,idxEndFeatures=11):
    """ Compute classifier probabilities for contours.

    Parameters
    ----------
    clf : scikit-learn classifier
        Binary classifier.
    contour_data : DataFrame
        DataFrame with contour information.

    Returns
    -------
    contour_data : DataFrame
        DataFrame with contour information and predicted probabilities.
    """
    contour_data['mel prob'] = -1
    features, _ = cc.pd_to_sklearn(contour_data,idxStartFeatures,idxEndFeatures)
    probs = clf.predict_proba(features)
    mel_probs = [p[1] for p in probs]
    contour_data['mel prob'] = mel_probs
    return contour_data


def get_best_threshold(y_ref, y_pred_score, plot=False):
    """ Get threshold on scores that maximizes f1 score.

    Parameters
    ----------
    y_ref : array
        Reference labels (binary).
    y_pred_score : array
        Predicted scores.
    plot : bool
        If true, plot ROC curve

    Returns
    -------
    best_threshold : float
        threshold on score that maximized f1 score
    max_fscore : float
        f1 score achieved at best_threshold
    """
    pos_weight = 1.0 - float(len(y_ref[y_ref == 1]))/float(len(y_ref))
    neg_weight = 1.0 - float(len(y_ref[y_ref == 0]))/float(len(y_ref))
    sample_weight = np.zeros(y_ref.shape)
    sample_weight[y_ref == 1] = pos_weight
    sample_weight[y_ref == 0] = neg_weight

    print "max prediction value = %s" % np.max(y_pred_score)
    print "min prediction value = %s" % np.min(y_pred_score)

    precision, recall, thresholds = \
            metrics.precision_recall_curve(y_ref, y_pred_score, pos_label=1,
                                           sample_weight=sample_weight)
    beta = 1.0
    btasq = beta**2.0
    fbeta_scores = (1.0 + btasq)*(precision*recall)/((btasq*precision)+recall)

    max_fscore = fbeta_scores[np.nanargmax(fbeta_scores)]
    best_threshold = thresholds[np.nanargmax(fbeta_scores)]

    if plot:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, '.b', label='PR curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right", frameon=True)
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, fbeta_scores[:-1], '.r', label='f1-score')
        plt.xlabel('Probability Threshold')
        plt.ylabel('F1 score')
        plt.show()

    plot_data = (recall, precision, thresholds, fbeta_scores[:-1])

    return best_threshold, max_fscore, plot_data
