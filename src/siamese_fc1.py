#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make siamese net using chainer
"""

import chainer
import chainer.functions as F
import chainer.links as L

class Siamese_net(chainer.Chain):

    def __init__(self, output_dim):
        super(Siamese_net, self).__init__(
            conv1=L.Convolution2D(1, 32, 3, pad=1, stride=1),
            conv2=L.Convolution2D(32, 32, 3, pad=1, stride=1),
            conv3=L.Convolution2D(32, 64, 3, pad=1, stride=1),
            conv4=L.Convolution2D(64, 64, 3, pad=1, stride=1),
            fc1=L.Linear(3136, output_dim),
        )
        self.train = True

    def clear(self):
        self.loss = None

    def __call__(self, x0, x1, t):
        self.clear()
        y0 = self.forward_one(x0)
        y1 = self.forward_one(x1)
        t = chainer.Variable(t, volatile='off')

        self.loss = F.contrastive(y0, y1, t)
        return self.loss

    def forward_one(self, x_data):
        x = chainer.Variable(x_data, volatile='off')
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv3(h))), 3, stride=2)
        h = F.relu(self.conv4(h))
        h = self.fc1(h)
        return h
