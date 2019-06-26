#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm)
from tensorlayer.models import Model
from data import flags

def get_G(name=None):
    gf_dim = 32
    w_init = tf.random_normal_initializer(stddev=0.02)

    nx = Input((flags.batch_size, 256, 256, 3))
    nn = Conv2d(gf_dim, (7, 7), (1, 1), W_init=w_init)(nx)
    nn = InstanceNorm(act=tf.nn.relu)(nn)

    nn = Conv2d(gf_dim * 2, (3, 3), (2, 2), W_init=w_init)(nn)
    nn = InstanceNorm(act=tf.nn.relu)(nn)

    nn = Conv2d(gf_dim * 4, (3, 3), (2, 2), W_init=w_init)(nn)
    nn = InstanceNorm(act=tf.nn.relu)(nn)

    n = nn
    for i in range(9):
        _nn = Conv2d(gf_dim * 4, (3, 3), (1, 1), W_init=w_init)(n)
        _nn = InstanceNorm(act=tf.nn.relu)(_nn)
        _nn = Conv2d(gf_dim * 4, (3, 3), (1, 1), W_init=w_init)(_nn)
        _nn = InstanceNorm(act=tf.nn.relu)(_nn)
        _nn = Elementwise(tf.add)([n, _nn])
        n = _nn

    nn = DeConv2d(gf_dim * 2, (3, 3), (2, 2))(n)
    nn = InstanceNorm(act=tf.nn.relu)(nn)

    nn = DeConv2d(gf_dim, (3, 3), (2, 2))(nn)
    nn = InstanceNorm(act=tf.nn.relu)(nn)

    nn = Conv2d(3, (7, 7), (1, 1), act=tf.nn.tanh)(nn)

    M = Model(inputs=nx, outputs=nn, name=name)
    return M

def get_D(name=None):
    df_dim = 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nx = Input((flags.batch_size, 256, 256, 3))

    n = Lambda(lambda x: tf.image.random_crop(x, [flags.batch_size, 70, 70, 3]))(nx) # patchGAN
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, W_init=w_init)(n)
    n = Conv2d(df_dim * 2, (4, 4), (2, 2), W_init=w_init)(n)
    n = InstanceNorm(act=lrelu)(n)

    n = Conv2d(df_dim * 4, (4, 4), (2, 2), W_init=w_init)(n)
    n = InstanceNorm(act=lrelu)(n)

    n = Conv2d(df_dim * 8, (4, 4), (2, 2), W_init=w_init)(n)
    n = InstanceNorm(act=lrelu)(n)

    n = Conv2d(1, (4, 4), (1, 1), W_init=w_init)(n)

    M = Model(inputs=nx, outputs=n, name=name)
    return M
