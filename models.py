#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d)
from tensorlayer.models import Model
from data import flags

# def get_G(name=None):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#
#     nx = Input((flags.batch_size, 256, 256, 3))
#
#     n = Conv2d(64, (7, 7), (1, 1), W_init=w_init)(nx)
#     n = InstanceNorm2d(act=tf.nn.relu)(n)
#
#     n = Conv2d(128, (3, 3), (2, 2), W_init=w_init)(n)
#     n = InstanceNorm2d(act=tf.nn.relu)(n)
#
#     n = Conv2d(256, (3, 3), (2, 2), W_init=w_init)(n)
#     n = InstanceNorm2d(act=tf.nn.relu)(n)
#
#     for i in range(9):
#         _n = Conv2d(256, (3, 3), (1, 1), W_init=w_init)(n)
#         _n = InstanceNorm2d(act=tf.nn.relu)(_n)
#         _n = Conv2d(256, (3, 3), (1, 1), W_init=w_init)(_n)
#         _n = InstanceNorm2d()(_n)
#         n = Elementwise(tf.add)([n, _n])
#
#     n = DeConv2d(128, (3, 3), (2, 2), W_init=w_init)(n)
#     n = InstanceNorm2d(act=tf.nn.relu)(n)
#
#     n = DeConv2d(64, (3, 3), (2, 2), W_init=w_init)(n)
#     n = InstanceNorm2d(act=tf.nn.relu)(n)
#
#     n = Conv2d(3, (7, 7), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
#
#     M = Model(inputs=nx, outputs=n, name=name)
#     return M

def get_G(name=None): # follow the paper
    w_init = tf.random_normal_initializer(stddev=0.02)

    nx = Input((flags.batch_size, 256, 256, 3))

    n = Conv2d(32, (7, 7), (1, 1), W_init=w_init)(nx)
    n = InstanceNorm2d(act=tf.nn.relu)(n)

    n = Conv2d(64, (3, 3), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=tf.nn.relu)(n)

    n = Conv2d(128, (3, 3), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=tf.nn.relu)(n)

    for i in range(9):
        _n = Conv2d(128, (3, 3), (1, 1), W_init=w_init)(n)
        _n = InstanceNorm2d(act=tf.nn.relu)(_n)
        _n = Conv2d(128, (3, 3), (1, 1), W_init=w_init)(_n)
        _n = InstanceNorm2d()(_n)
        n = Elementwise(tf.add)([n, _n])

    n = DeConv2d(64, (3, 3), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=tf.nn.relu)(n)

    n = DeConv2d(32, (3, 3), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=tf.nn.relu)(n)

    n = Conv2d(3, (7, 7), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)

    M = Model(inputs=nx, outputs=n, name=name)
    return M

def get_D(name=None):
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nx = Input((flags.batch_size, 256, 256, 3))
            # n = Lambda(lambda x: tf.image.random_crop(x, [flags.batch_size, 70, 70, 3]))(nx)

    n = Conv2d(64, (4, 4), (2, 2), act=lrelu, W_init=w_init)(nx)
    n = Conv2d(128, (4, 4), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu)(n)

    n = Conv2d(256, (4, 4), (2, 2), W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu)(n)

    n = Conv2d(512, (4, 4), (1, 1), W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu)(n)

    n = Conv2d(1, (4, 4), (1, 1), padding='VALID', W_init=w_init)(n)
    n = Flatten()(n)
    M = Model(inputs=nx, outputs=n, name=name)
    return M
