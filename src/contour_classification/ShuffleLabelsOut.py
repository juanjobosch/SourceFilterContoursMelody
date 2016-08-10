#!/usr/bin/env python
'''Generate train/test splits by random shuffling of labels'''
"""Taken from
https://github.com/bmcfee/ml_scraps/blob/master/ShuffleLabelsOut.py"""

import numpy as np
from sklearn.cross_validation import ShuffleSplit


class ShuffleLabelsOut(ShuffleSplit):
    '''Shuffle- Labels-Out cross-validation iterator

    Parameters
    ----------
    y :  array, [n_samples]
        Labels of samples

    n_iter : int (default 5)
        Number of shuffles to generate

    test_size : float (default 0.2), int, or None

    train_size : float, int, or None (default is None)

    random_state : int or RandomState
    '''

    def __init__(self, y, n_iter=5, test_size=0.2, train_size=None,
                 random_state=None):

        classes, y_indices = np.unique(y, return_inverse=True)

        super(ShuffleLabelsOut, self).__init__(
            len(classes), n_iter=n_iter, test_size=test_size, train_size=train_size,
            random_state=random_state)

        self.classes = classes
        self.y_indices = y_indices

    def __repr__(self):
        return ('%s(labels=%s, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.y_indices,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter

    def _iter_indices(self):

        for y_train, y_test in super(ShuffleLabelsOut, self)._iter_indices():
            # these are the indices of classes in the partition
            # invert them into data indices

            train = np.flatnonzero(np.in1d(self.y_indices, y_train))
            test = np.flatnonzero(np.in1d(self.y_indices, y_test))

            yield train, test
