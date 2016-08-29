#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dump feature vector on siamese net or triplet net
"""
from __future__ import print_function

import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda


def dump_feature_vector(model, d_name, dic, outputdim, batchsize, xp, gpu, n_plot=None):
    n_data = len(dic['data'])
    results = xp.empty((n_data, outputdim))
    for i in range(0, n_data, batchsize):
        x_batch = xp.asarray(dic['data'][i:i + batchsize], dtype=xp.float32)
        y = model.forward_one(x_batch)
        results[i:i + batchsize] = y.data

    if gpu >= 0:
        results = xp.asnumpy(results)
    
    print('save feature vector')
    pickle.dump(results, open('{}.vector'.format(d_name), 'wb'), -1)
    pickle.dump(dic['target'], open('{}.target'.format(d_name), 'wb'), -1)
    make_plot(results, dic['target'], d_name, n_plot)

def make_plot(data, target, d_name, n_plot=None):
    print('plot vectors')
    c = ['#ff0000','#0000ff','#d16b16','#00984b','#0074bf',
            '#c30068','#6d1782', '#546474', '#244765', '#8f253b']
    c_markers = ['s', 'v', 'o', 'd', '*', '+', 'H', 'p', 'x', 'D']
    for i in range(len(list(set(target)))):
        feat = data[np.where(target == i)]
	if n_plot != None:
	    feat = feat[:n_plot]
    	plt.plot(feat[:, 0], feat[:, 1], c_markers[i], markeredgecolor=c[i], markerfacecolor='#ffffff', markeredgewidth=2, markersize=10)
    plt.legend(['0', '1','2','3','4','5','6','7','8','9'], numpoints=1, borderaxespad=0, bbox_to_anchor=(1.15, 1))
    plt.subplots_adjust(left=0.1, right=0.85)
    plt.savefig('{}_vector.png'.format(d_name))
    print('plot end')
