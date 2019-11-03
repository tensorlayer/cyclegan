#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import models
from data import flags, data_A, data_B, im_test_A, im_test_B, n_step_per_epoch

im_test_A = np.asarray(im_test_A, dtype=np.float32) / 127.5 - 1
im_test_B = np.asarray(im_test_B, dtype=np.float32)  / 127.5 - 1

sample_A = im_test_A[0:25] # some images for visualization
sample_B = im_test_B[0:25]
# tl.prepro.threading_data(sample_A, prep)
# ni = int(np.sqrt(flags.batch_size))
tl.vis.save_images(sample_A, [5, 5], flags.sample_dir+'/_sample_A.png')
tl.vis.save_images(sample_B, [5, 5], flags.sample_dir+'/_sample_B.png')

def train():
    Gab = models.get_G(name='Gab')
    Gba = models.get_G(name='Gba')
    Da  = models.get_D(name='Da')
    Db  = models.get_D(name='Db')

    Gab.train()
    Gba.train()
    Da.train()
    Db.train()

    lr_v = tf.Variable(flags.lr_init)
    optimizer_Gab = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Gba = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Da = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Db = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)

    # Gab.load_weights(flags.model_dir + '/Gab.h5')
    # Gba.load_weights(flags.model_dir + '/Gba.h5')
    # Da.load_weights(flags.model_dir + '/Da.h5')
    # Db.load_weights(flags.model_dir + '/Db.h5')

    for epoch in range(0, flags.n_epoch):
        # reduce lr linearly after 100 epochs, from lr_init to 0
        if epoch >= 100:
            new_lr = flags.lr_init - flags.lr_init * (epoch - 100) / 100
            lr_v.assign(lr_v, new_lr)
            print("New learning rate %f" % new_lr)

        # train 1 epoch
        for step, (image_A, image_B) in enumerate(zip(data_A, data_B)):
            if image_A.shape[0] != flags.batch_size or image_B.shape[0] != flags.batch_size : # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # print(image_A.numpy().max())
                fake_B = Gab(image_A)
                fake_A = Gba(image_B)
                cycle_A = Gba(fake_B)
                cycle_B = Gab(fake_A)
                logits_fake_B = Db(fake_B)    # TODO: missing image buffer (pool)
                logits_real_B = Db(image_B)
                logits_fake_A = Da(fake_A)
                logits_real_A = Da(image_A)
                # loss_Da = (tl.cost.mean_squared_error(logits_real_A, tf.ones_like(logits_real_A), is_mean=True) + \  # LSGAN
                #     tl.cost.mean_squared_error(logits_fake_A, tf.ones_like(logits_fake_A), is_mean=True)) / 2.
                loss_Da = tf.reduce_mean(tf.math.squared_difference(logits_fake_A, tf.zeros_like(logits_fake_A))) + \
                    tf.reduce_mean(tf.math.squared_difference(logits_real_A, tf.ones_like(logits_real_A)))
                # loss_Da = tl.cost.sigmoid_cross_entropy(logits_fake_A, tf.zeros_like(logits_fake_A)) + \
                    # tl.cost.sigmoid_cross_entropy(logits_real_A, tf.ones_like(logits_real_A))
                # loss_Db = (tl.cost.mean_squared_error(logits_real_B, tf.ones_like(logits_real_B), is_mean=True) + \ # LSGAN
                #     tl.cost.mean_squared_error(logits_fake_B, tf.ones_like(logits_fake_B), is_mean=True)) / 2.
                loss_Db = tf.reduce_mean(tf.math.squared_difference(logits_fake_B, tf.zeros_like(logits_fake_B))) + \
                    tf.reduce_mean(tf.math.squared_difference(logits_real_B, tf.ones_like(logits_real_B)))
                # loss_Db = tl.cost.sigmoid_cross_entropy(logits_fake_B, tf.zeros_like(logits_fake_B)) + \
                #     tl.cost.sigmoid_cross_entropy(logits_real_B, tf.ones_like(logits_real_B))
                # loss_Gab = tl.cost.mean_squared_error(logits_fake_B, tf.ones_like(logits_fake_B), is_mean=True) # LSGAN
                loss_Gab = tf.reduce_mean(tf.math.squared_difference(logits_fake_B, tf.ones_like(logits_fake_B)))
                # loss_Gab = tl.cost.sigmoid_cross_entropy(logits_fake_B, tf.ones_like(logits_fake_B))
                # loss_Gba = tl.cost.mean_squared_error(logits_fake_A, tf.ones_like(logits_fake_A), is_mean=True) # LSGAN
                loss_Gba = tf.reduce_mean(tf.math.squared_difference(logits_fake_A, tf.ones_like(logits_fake_A)))
                # loss_Gba = tl.cost.sigmoid_cross_entropy(logits_fake_A, tf.ones_like(logits_fake_A))
                # loss_cyc = 10 * (tl.cost.absolute_difference_error(image_A, cycle_A, is_mean=True) + \
                #     tl.cost.absolute_difference_error(image_B, cycle_B, is_mean=True))
                loss_cyc = 10. * (tf.reduce_mean(tf.abs(image_A - cycle_A)) + tf.reduce_mean(tf.abs(image_B - cycle_B)))
                loss_Gab_total = loss_Gab + loss_cyc
                loss_Gba_total = loss_Gba + loss_cyc
            grad = tape.gradient(loss_Gab_total, Gab.trainable_weights)
            optimizer_Gab.apply_gradients(zip(grad, Gab.trainable_weights))
            grad = tape.gradient(loss_Gba_total, Gba.trainable_weights)
            optimizer_Gba.apply_gradients(zip(grad, Gba.trainable_weights))
            grad = tape.gradient(loss_Da, Da.trainable_weights)
            optimizer_Da.apply_gradients(zip(grad, Da.trainable_weights))
            grad = tape.gradient(loss_Db, Db.trainable_weights)
            optimizer_Db.apply_gradients(zip(grad, Db.trainable_weights))
            # del tape
            print("Epoch[{}/{}] step[{}/{}] time:{} Gab:{} Gba:{} cyc:{} Da:{} Db:{}".format(\
                epoch, flags.n_epoch, step, n_step_per_epoch, time.time()-step_time, \
                loss_Gab, loss_Gba, loss_cyc, loss_Da, loss_Db))

        # visualization
        outb = Gab(sample_A)
        outa = Gba(sample_B)
        tl.vis.save_images(outb.numpy(), [5, 5], flags.sample_dir+'/{}_a2b.png'.format(epoch))
        tl.vis.save_images(outa.numpy(), [5, 5], flags.sample_dir+'/{}_b2a.png'.format(epoch))

        # save models
        if epoch % 5:
            Gab.save_weights(flags.model_dir + '/Gab.h5')
            Gba.save_weights(flags.model_dir + '/Gba.h5')
            Da.save_weights(flags.model_dir + '/Da.h5')
            Db.save_weights(flags.model_dir + '/Db.h5')

def parallel_train(kungfu_option):
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer

    Gab = models.get_G(name='Gab')
    Gba = models.get_G(name='Gba')
    Da  = models.get_D(name='Da')
    Db  = models.get_D(name='Db')

    Gab.train()
    Gba.train()
    Da.train()
    Db.train()

    lr_v = tf.Variable(flags.lr_init)
    optimizer_Gab = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Gba = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Da = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    optimizer_Db = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)

    # KungFu: wrap the optimizers
    if kungfu_option == 'sync-sgd':
        optimizer_Gab = SynchronousSGDOptimizer(optimizer_Gab)
        optimizer_Gba = SynchronousSGDOptimizer(optimizer_Gba)
        optimizer_Da = SynchronousSGDOptimizer(optimizer_Da)
        optimizer_Db = SynchronousSGDOptimizer(optimizer_Db)
    elif kungfu_option == 'async-sgd':
        optimizer_Gab = PairAveragingOptimizer(optimizer_Gab)
        optimizer_Gba = PairAveragingOptimizer(optimizer_Gba)
        optimizer_Da = PairAveragingOptimizer(optimizer_Da)
        optimizer_Db = PairAveragingOptimizer(optimizer_Db)
    elif kungfu_option == 'sma':
        optimizer_Gab = SynchronousAveragingOptimizer(optimizer_Gab)
        optimizer_Gba = SynchronousAveragingOptimizer(optimizer_Gba)
        optimizer_Da = SynchronousAveragingOptimizer(optimizer_Da)
        optimizer_Db = SynchronousAveragingOptimizer(optimizer_Db)
    else:
        raise RuntimeError('Unknown distributed training optimizer.')

    # Gab.load_weights(flags.model_dir + '/Gab.h5')
    # Gba.load_weights(flags.model_dir + '/Gba.h5')
    # Da.load_weights(flags.model_dir + '/Da.h5')
    # Db.load_weights(flags.model_dir + '/Db.h5')

    # KungFu: shard the data
    data_A_shard = []
    data_B_shard = []
    for step, (image_A, image_B) in enumerate(zip(data_A, data_B)):
        if step % current_cluster_size() == current_rank():
            data_A_shard.append(image_A)
            data_B_shard.append(image_B)

    for epoch in range(0, flags.n_epoch):
        # reduce lr linearly after 100 epochs, from lr_init to 0
        if epoch >= 100:
            new_lr = flags.lr_init - flags.lr_init * (epoch - 100) / 100
            lr_v.assign(lr_v, new_lr)
            print("New learning rate %f" % new_lr)

        # train 1 epoch
        for step, (image_A, image_B) in enumerate(zip(data_A_shard, data_B_shard)):
            if image_A.shape[0] != flags.batch_size or image_B.shape[0] != flags.batch_size : # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # print(image_A.numpy().max())
                fake_B = Gab(image_A)
                fake_A = Gba(image_B)
                cycle_A = Gba(fake_B)
                cycle_B = Gab(fake_A)
                logits_fake_B = Db(fake_B)    # TODO: missing image buffer (pool)
                logits_real_B = Db(image_B)
                logits_fake_A = Da(fake_A)
                logits_real_A = Da(image_A)
                # loss_Da = (tl.cost.mean_squared_error(logits_real_A, tf.ones_like(logits_real_A), is_mean=True) + \  # LSGAN
                #     tl.cost.mean_squared_error(logits_fake_A, tf.ones_like(logits_fake_A), is_mean=True)) / 2.
                loss_Da = tf.reduce_mean(tf.math.squared_difference(logits_fake_A, tf.zeros_like(logits_fake_A))) + \
                    tf.reduce_mean(tf.math.squared_difference(logits_real_A, tf.ones_like(logits_real_A)))
                # loss_Da = tl.cost.sigmoid_cross_entropy(logits_fake_A, tf.zeros_like(logits_fake_A)) + \
                    # tl.cost.sigmoid_cross_entropy(logits_real_A, tf.ones_like(logits_real_A))
                # loss_Db = (tl.cost.mean_squared_error(logits_real_B, tf.ones_like(logits_real_B), is_mean=True) + \ # LSGAN
                #     tl.cost.mean_squared_error(logits_fake_B, tf.ones_like(logits_fake_B), is_mean=True)) / 2.
                loss_Db = tf.reduce_mean(tf.math.squared_difference(logits_fake_B, tf.zeros_like(logits_fake_B))) + \
                    tf.reduce_mean(tf.math.squared_difference(logits_real_B, tf.ones_like(logits_real_B)))
                # loss_Db = tl.cost.sigmoid_cross_entropy(logits_fake_B, tf.zeros_like(logits_fake_B)) + \
                #     tl.cost.sigmoid_cross_entropy(logits_real_B, tf.ones_like(logits_real_B))
                # loss_Gab = tl.cost.mean_squared_error(logits_fake_B, tf.ones_like(logits_fake_B), is_mean=True) # LSGAN
                loss_Gab = tf.reduce_mean(tf.math.squared_difference(logits_fake_B, tf.ones_like(logits_fake_B)))
                # loss_Gab = tl.cost.sigmoid_cross_entropy(logits_fake_B, tf.ones_like(logits_fake_B))
                # loss_Gba = tl.cost.mean_squared_error(logits_fake_A, tf.ones_like(logits_fake_A), is_mean=True) # LSGAN
                loss_Gba = tf.reduce_mean(tf.math.squared_difference(logits_fake_A, tf.ones_like(logits_fake_A)))
                # loss_Gba = tl.cost.sigmoid_cross_entropy(logits_fake_A, tf.ones_like(logits_fake_A))
                # loss_cyc = 10 * (tl.cost.absolute_difference_error(image_A, cycle_A, is_mean=True) + \
                #     tl.cost.absolute_difference_error(image_B, cycle_B, is_mean=True))
                loss_cyc = 10. * (tf.reduce_mean(tf.abs(image_A - cycle_A)) + tf.reduce_mean(tf.abs(image_B - cycle_B)))
                loss_Gab_total = loss_Gab + loss_cyc
                loss_Gba_total = loss_Gba + loss_cyc
            grad = tape.gradient(loss_Gab_total, Gab.trainable_weights)
            optimizer_Gab.apply_gradients(zip(grad, Gab.trainable_weights))
            grad = tape.gradient(loss_Gba_total, Gba.trainable_weights)
            optimizer_Gba.apply_gradients(zip(grad, Gba.trainable_weights))
            grad = tape.gradient(loss_Da, Da.trainable_weights)
            optimizer_Da.apply_gradients(zip(grad, Da.trainable_weights))
            grad = tape.gradient(loss_Db, Db.trainable_weights)
            optimizer_Db.apply_gradients(zip(grad, Db.trainable_weights))
            # del tape
            print("Epoch[{}/{}] step[{}/{}] time:{} Gab:{} Gba:{} cyc:{} Da:{} Db:{}".format(\
                epoch, flags.n_epoch, step, n_step_per_epoch, time.time()-step_time, \
                loss_Gab, loss_Gba, loss_cyc, loss_Da, loss_Db))

            # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
            if step == 0:
                from kungfu.tensorflow.initializer import broadcast_variables

                # Broadcast model variables
                broadcast_variables(Gab.trainable_weights)
                broadcast_variables(Gba.trainable_weights)
                broadcast_variables(Da.trainable_weights)
                broadcast_variables(Db.trainable_weights)

                # Broadcast optimizer variables
                broadcast_variables(optimizer_Gab.variables())
                broadcast_variables(optimizer_Gba.variables())
                broadcast_variables(optimizer_Da.variables())
                broadcast_variables(optimizer_Db.variables())

        # visualization
        outb = Gab(sample_A)
        outa = Gba(sample_B)
        tl.vis.save_images(outb.numpy(), [5, 5], flags.sample_dir+'/{}_a2b.png'.format(epoch))
        tl.vis.save_images(outa.numpy(), [5, 5], flags.sample_dir+'/{}_b2a.png'.format(epoch))

        # save models
        if epoch % 5:
            Gab.save_weights(flags.model_dir + '/Gab.h5')
            Gba.save_weights(flags.model_dir + '/Gba.h5')
            Da.save_weights(flags.model_dir + '/Da.h5')
            Db.save_weights(flags.model_dir + '/Db.h5')

def eval():
    Gab = models.get_G()
    Gba = models.get_G()
    Gab.eval()
    Gba.eval()
    Gab.load_weights(flags.model_dir + '/Gab.h5')
    Gba.load_weights(flags.model_dir + '/Gba.h5')
    for i, (x, _) in enumerate(tl.iterate.minibatches(inputs=im_test_A, targets=im_test_A, batch_size=25, shuffle=False)):
        o = Gab(x)
        tl.vis.save_images(x, [5, 5], flags.sample_dir+'/eval_{}_a.png'.format(i))
        tl.vis.save_images(o.numpy(), [5, 5], flags.sample_dir+'/eval_{}_a2b.png'.format(i))
    for i, (x, _) in enumerate(tl.iterate.minibatches(inputs=im_test_B, targets=im_test_B, batch_size=25, shuffle=False)):
        o = Gba(x)
        tl.vis.save_images(x, [5, 5], flags.sample_dir+'/eval_{}_b.png'.format(i))
        tl.vis.save_images(o.numpy(), [5, 5], flags.sample_dir+'/eval_{}_b2a.png'.format(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenPose-plus.')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sma',
                        help='available options: sync-sgd, async-sgd, sma')
    parser.add_argument('--parallel',
                        action='store_true',
                        default=False,
                        help='enable parallel training')
    args = parser.parse_args()

    if args.parallel:
        parallel_train(args.kf_optimizer)
    else:
        train()

    eval()
