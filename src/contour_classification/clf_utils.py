""" Utilities for classifier experiments """
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def cross_val_sweep(x_train, y_train, max_search=100,
                    step=5, plot=True):
    """ Choose best parameter by performing cross fold validation

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    y_train : np.array [n_samples]
        Training labels
    max_search : int
        Maximum depth value to sweep
    step : int
        Step size in parameter sweep
    plot : bool
        If true, plot error bars and cv accuracy

    Returns
    -------
    best_depth : int
        Optimal max_depth parameter
    max_cv_accuracy : DataFrames
        Best accuracy achieved on hold out set with optimal parameter.
    """
    scores = []
    for max_depth in np.arange(5, max_search, step):
        print "training with max_depth=%s" % max_depth
        clf = RFC(n_estimators=100, max_depth=max_depth, n_jobs=-1,
                  class_weight='auto', max_features=None)
        all_scores = cross_validation.cross_val_score(clf, x_train, y_train,
                                                      cv=5)
        scores.append([max_depth, np.mean(all_scores), np.std(all_scores)])

    depth = [score[0] for score in scores]
    accuracy = [score[1] for score in scores]
    std_dev = [score[2] for score in scores]

    if plot:
        plt.errorbar(depth, accuracy, std_dev, linestyle='-', marker='o')
        plt.title('Mean cross validation accuracy')
        plt.xlabel('max depth')
        plt.ylabel('mean accuracy')
        plt.show()

    best_depth = depth[np.argmax(accuracy)]
    max_cv_accuracy = np.max(accuracy)
    plot_data = (depth, accuracy, std_dev)

    return best_depth, max_cv_accuracy, plot_data


def train_clf(x_train, y_train, best_depth):
    """ Train classifier.

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    y_train : np.array [n_samples]
        Training labels
    best_depth : int
        Optimal max_depth parameter

    Returns
    -------
    clf : classifier
        Trained scikit-learn classifier
    """
    clf = RFC(n_estimators=100, max_depth=best_depth, n_jobs=-1,
              class_weight='auto', max_features=None)
    clf = clf.fit(x_train, y_train)
    return clf


def clf_predictions(x_train, x_valid, x_test, clf):
    """ Compute probability predictions for all training and test examples.

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    x_test : np.array [n_samples, n_features]
        Testing features.
    clf : classifier
        Trained scikit-learn classifier

    Returns
    -------
    p_train : np.array [n_samples]
        predicted probabilities for training set
    p_test : np.array [n_samples]
        predicted probabilities for testing set
    """
    p_train = clf.predict_proba(x_train)[:, 1]
    p_valid = clf.predict_proba(x_valid)[:, 1]
    p_test = clf.predict_proba(x_test)[:, 1]
    return p_train, p_valid, p_test


def clf_metrics(p_train, p_test, y_train, y_test):
    """ Compute metrics on classifier predictions

    Parameters
    ----------
    p_train : np.array [n_samples]
        predicted probabilities for training set
    p_test : np.array [n_samples]
        predicted probabilities for testing set
    y_train : np.array [n_samples]
        Training labels.
    y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    clf_scores : dict
        classifier scores for training set
    """
    y_pred_train = 1*(p_train >= 0.5)
    y_pred_test = 1*(p_test >= 0.5)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores['accuracy'] = metrics.accuracy_score(y_test, y_pred_test)

    train_scores['mcc'] = metrics.matthews_corrcoef(y_train, y_pred_train)
    test_scores['mcc'] = metrics.matthews_corrcoef(y_test, y_pred_test)

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_train,
                                                           y_pred_train)
    train_scores['precision'] = p
    train_scores['recall'] = r
    train_scores['f1'] = f
    train_scores['support'] = s

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_test,
                                                           y_pred_test)
    test_scores['precision'] = p
    test_scores['recall'] = r
    test_scores['f1'] = f
    test_scores['support'] = s

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_train, y_pred_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_test, y_pred_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(y_train, p_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(y_test, p_test + 1, average='weighted')

    clf_scores = {'train': train_scores, 'test': test_scores}

    return clf_scores

