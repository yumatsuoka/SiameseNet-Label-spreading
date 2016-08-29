#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Label-Spreading
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
from sklearn.semi_supervised import label_propagation

class Label_spreading:

    def __init__(self, array, labels, target, n_testlabel):
        self.array = array
        self.labels = labels
        self.target = target
        self.predict_labels = None
        self.accuracy = 0
        self.n_testlabel = n_testlabel
        self.label_spread = label_propagation.LabelSpreading(
            kernel='knn', alpha=1.0)

    def fit(self):
        self.label_spread.fit(self.array, self.labels)

    def get_predict_labels(self):
        self.predict_labels = np.asarray(self.label_spread.transduction_)
        
    def evaluate(self):
        for i in six.moves.range(self.n_testlabel):
            if self.predict_labels[i] == self.target[i]:
                self.accuracy += 1
        self.accuracy = self.accuracy *100 / self.n_testlabel
