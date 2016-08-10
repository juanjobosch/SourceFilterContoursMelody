""" Functions for doing scoring based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal
from sklearn import metrics


def transform_features(x_train, x_test):
    """ Transform features using a boxcox transform. Remove vibrato features.
    Comptes the optimal value of lambda on the training set and applies this
    lambda to the testing set.

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Untransformed training features.
    x_test : np.array [n_samples, n_features]
        Untransformed testing features.

    Returns
    -------
    x_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    x_test_boxcox : np.array [n_samples, n_features_trans]
        Transformed testing features.
    """
    x_train = x_train[:, 0:6]
    x_test = x_test[:, 0:6]

    _, n_feats = x_train.shape

    x_train_boxcox = np.zeros(x_train.shape)
    lmbda_opt = np.zeros((n_feats,))

    eps = 1.0  # shift features away from zero
    for i in range(n_feats):
        x_train_boxcox[:, i], lmbda_opt[i] = boxcox(x_train[:, i] + eps)

    x_test_boxcox = np.zeros(x_test.shape)
    for i in range(n_feats):
        x_test_boxcox[:, i] = boxcox(x_test[:, i] + eps, lmbda=lmbda_opt[i])

    return x_train_boxcox, x_test_boxcox


def fit_gaussians(x_train_boxcox, y_train):
    """ Fit class-dependent multivariate gaussians on the training set.

    Parameters
    ----------
    x_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    y_train : np.array [n_samples]
        Training labels.

    Returns
    -------
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class
    """
    pos_idx = np.where(y_train == 1)[0]
    mu_pos = np.mean(x_train_boxcox[pos_idx, :], axis=0)
    cov_pos = np.cov(x_train_boxcox[pos_idx, :], rowvar=0)

    neg_idx = np.where(y_train == 0)[0]
    mu_neg = np.mean(x_train_boxcox[neg_idx, :], axis=0)
    cov_neg = np.cov(x_train_boxcox[neg_idx, :], rowvar=0)
    rv_pos = multivariate_normal(mean=mu_pos, cov=cov_pos, allow_singular=True)
    rv_neg = multivariate_normal(mean=mu_neg, cov=cov_neg, allow_singular=True)
    return rv_pos, rv_neg


def melodiness(sample, rv_pos, rv_neg):
    """ Compute melodiness score for an example given trained distributions.

    Parameters
    ----------
    sample : np.array [n_feats]
        Instance of transformed data.
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class

    Returns
    -------
    melodiness: float
        score between 0 and inf. class cutoff at 1
    """
    return rv_pos.pdf(sample)/rv_neg.pdf(sample)


def compute_all_melodiness(x_train_boxcox, x_test_boxcox, rv_pos, rv_neg):
    """ Compute melodiness for all training and test examples.

    Parameters
    ----------
    x_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    x_test_boxcox : np.array [n_samples, n_features_trans]
        Transformed testing features.
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class

    Returns
    -------
    m_train : np.array [n_samples]
        melodiness scores for training set
    m_test : np.array [n_samples]
        melodiness scores for testing set
    """
    n_train = x_train_boxcox.shape[0]
    n_test = x_test_boxcox.shape[0]

    m_train = np.zeros((n_train, ))
    m_test = np.zeros((n_test, ))

    for i, sample in enumerate(x_train_boxcox):
        m_train[i] = melodiness(sample, rv_pos, rv_neg)

    for i, sample in enumerate(x_test_boxcox):
        m_test[i] = melodiness(sample, rv_pos, rv_neg)

    return m_train, m_test


def melodiness_metrics(m_train, m_test, y_train, y_test):
    """ Compute metrics on melodiness score

    Parameters
    ----------
    m_train : np.array [n_samples]
        melodiness scores for training set
    m_test : np.array [n_samples]
        melodiness scores for testing set
    y_train : np.array [n_samples]
        Training labels.
    y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    melodiness_scores : dict
        melodiness scores for training set
    """
    m_bin_train = 1*(m_train >= 1)
    m_bin_test = 1*(m_test >= 1)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(y_train, m_bin_train)
    test_scores['accuracy'] = metrics.accuracy_score(y_test, m_bin_test)

    train_scores['mcc'] = metrics.matthews_corrcoef(y_train, m_bin_train)
    test_scores['mcc'] = metrics.matthews_corrcoef(y_test, m_bin_test)

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_train,
                                                           m_bin_train)
    train_scores['precision'] = p
    train_scores['recall'] = r
    train_scores['f1'] = f
    train_scores['support'] = s

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_test,
                                                           m_bin_test)
    test_scores['precision'] = p
    test_scores['recall'] = r
    test_scores['f1'] = f
    test_scores['support'] = s

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_train, m_bin_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_test, m_bin_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(y_train, m_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(y_test, m_test + 1, average='weighted')

    melodiness_scores = {'train': train_scores, 'test': test_scores}

    return melodiness_scores

