#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train siamese net and get feature vectors
"""

from __future__ import print_function

import six
import time
import argparse
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.datasets import fetch_mldata
import numpy as np

import chainer
from chainer import cuda, optimizers, serializers

import dump_vec
import siamese_fc1


def train_and_dump(model, optimizer, labeled_data_dict, unlabeled_data_dict,\
        xp, batchsize, epoch, plot_dim, gpu, outputdim):
    x_train = labeled_data_dict['data']
    y_train = labeled_data_dict['target']
    n_train = len(y_train)
    loss_list = np.empty(epoch)

    for itr in six.moves.range(1, epoch + 1):
    # for itr in tqdm(six.moves.range(1, epoch + 1)):
        print('epoch', itr)
        perm = np.random.permutation(n_train)
        sum_train_loss = 0
        for i in six.moves.range(0, n_train, batchsize):
            x0 = x_train[perm[i:i + batchsize]]
            x1 = x_train[perm[i:i + batchsize]][::-1]
            y0 = y_train[perm[i:i + batchsize]]
            y1 = y_train[perm[i:i + batchsize]][::-1]
            label = xp.array(y0 == y1, dtype=xp.int32)

            x0 = xp.asarray(x0, dtype=xp.float32)
            x1 = xp.asarray(x1, dtype=xp.float32)
            real_batchsize = len(x0)
            
            optimizer.zero_grads()
            loss = model(x0, x1, label)
            loss.backward()
            optimizer.update()

            sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize

        print('train mean loss={}'.format(sum_train_loss / n_train))
        loss_list[epoch-1] = sum_train_loss / n_train

    # print('dump model & optimizer')
    # serializers.save_hdf5('../dump/{}_shiamese.model'.format(d_name), model)
    # serializers.save_hdf5('../dump/{}_shiamese.state'.format(d_name), optimizer)
    
    print('Make loss graph')
    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_list)
    plt.savefig('../dump/{}_loss.png'.format(d_name))

    print('dump feature vector')
    dump_vec.dump_feature_vector(model, '../dump/{}_label'.format(d_name),\
            labeled_data_dict, outputdim, batchsize, xp, gpu)
    dump_vec.dump_feature_vector(model, '../dump/{}_unlabel'.format(d_name),\
            unlabeled_data_dict, outputdim, batchsize, xp, gpu, plot_dim)


def get_mnist(n_data):
    mnist = fetch_mldata('MNIST original')
    r_data = mnist['data'].astype(np.float32)
    r_label = mnist['target'].astype(np.int32)
    # former 60,000 samples which is training data in MNIST.
    perm = np.random.permutation(60000)
    data = r_data[perm]
    label = r_label[perm]
    # split the data to training data(labeled data) and test data(unlabeled data)
    ld_dict = {'data':data[:n_data].reshape((n_data, 1, 28, 28)) / 255.0,\
            'target':label[:n_data]}
    unld_dict = {'data':data[n_data:].reshape((60000-n_data, 1, 28, 28)) / 255.0,\
            'target':label[n_data:]}
    
    return ld_dict, unld_dict


if __name__ == '__main__':
    
    st = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default=-1, type=int)
    parser.add_argument('--epoch',  default=300, type=int)
    parser.add_argument('--batchsize', default=20, type=int)
    parser.add_argument('--initmodel', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--outputdim', default=100, type=int)
    parser.add_argument('--wd',     default=0.0001, type=float)
    parser.add_argument('--lr',     default=0.00015, type=float)
    parser.add_argument('--n_train',    default=1000, type=int)
    parser.add_argument('--plot_dim',   default=1000, type=int)
    parser.add_argument('--d_name', default='hoge', type=str)
    args = parser.parse_args()

    print('Create model')
    model = siamese_fc1.Siamese_net(args.outputdim)

    print('Check gpu')
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('Load dataset')
    ld_dict, unld_dict = get_mnist(args.n_train)

    print('Setup optimizer')
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.wd))

    if args.initmodel:
        model_dir = '../dump/{}_siamese.model'.format(args.d_name)
        serializers.load_hdf5(model_dir, model)
    if args.resume:
        state_dir = '../dump/{}_siamese.state'.format(args.d_name)
        serializers.load_hdf5(state_dir, optimizer)

    print('training and test')
    train_and_dump(model, optimizer, ld_dict, unld_dict, xp, args.batchsize,\
            args.epoch, args.plot_dim, args.gpu, args.outputdim)
    print('end')
    print('elapsed time[m]:', (time.clock() - st)/60.0)
