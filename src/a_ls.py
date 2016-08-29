#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply Label-Spreading
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import six
import six.moves.cPickle as pickle
import numpy as np

from label_spreading import Label_spreading

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='Label-spreading')
    parser.add_argument('--d_name', type=str, default='hoge')
    args = parser.parse_args()
    
    print('label_spreading Load dataset')
    label_data = pickle.load(open('../dump/{}_label.vector'.format(args.d_name)))
    label_target = pickle.load(open('../dump/{}_label.target'.format(args.d_name)))
    nolabel_data = pickle.load(open('../dump/{}_unlabel.vector'.format(args.d_name)))
    nolabel_target = pickle.load(open('../dump/{}_unlabel.target'.format(args.d_name)))

    print('Make dataset and target')
    n_testlabel = len(nolabel_target)
    array = np.concatenate((nolabel_data, label_data), axis=0)
    target = np.concatenate((nolabel_target, label_target), axis=0)
    
    print('Make labels')
    labels = -np.ones(len(nolabel_target) + len(label_target))
    for i in six.moves.range(len(label_target)):
        labels[len(nolabel_target)+i] = label_target[i]
    
    ####################################    
    #need array, label, target, labeled_dim
    print("Create object of Label Spreading...")
    ls = Label_spreading(array, labels, target, n_testlabel)
    print("Start label spreading...")
    ls.fit()
    print("Get results")
    ls.get_predict_labels()
    print('Evaluate result...')
    ls.evaluate()
    print('accuracy=', ls.accuracy)   
    print("end all process...")
