#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs
from matplotlib import pyplot as plt

from examples import conf
from examples.utils import dataset, save_image_collections


@zs.reuse('model')
def generator(observed, n, n_h, n_x, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        # [n, n_h]
        minval = tf.ones(shape=[n, n_h]) * (-1.0)
        maxval = tf.ones(shape=[n, n_h])
        z = zs.Uniform('z', minval=minval, maxval=maxval)
        lx_z = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        # [n, n_x]
        x = tf.sigmoid(x_logits)

    return model, x, x_logits


def mmd(samples1, n1, samples2, n2, sigmas=(1, 2, 5, 10, 20, 40, 80)):
    samples = tf.concat([samples1, samples2], axis=0)
    samples_X2 = tf.reduce_sum(samples * samples, axis=1, keep_dims=True)
    samples_XX = tf.matmul(samples, samples, transpose_b=True)
    samples_val = samples_XX - 0.5 * samples_X2 - \
                  0.5 * tf.transpose(samples_X2)
    weights = tf.concat([tf.constant(1.0 / n1, shape=(n1, 1), dtype=tf.float32),
                         tf.constant(-1.0 / n2, shape=(n2, 1), dtype=tf.float32)], axis=0)
    exception = tf.concat([tf.constant(1.0 / n1 / n1, shape=(n1,), dtype=tf.float32),
                         tf.constant(1.0 / n2 / n2, shape=(n2,), dtype=tf.float32)], axis=0)
    weights = tf.matmul(weights, weights, transpose_b=True) - \
              tf.matrix_diag(exception)
    cost = 0.0
    for sigma in sigmas:
        kernel_val = tf.exp(1.0 / sigma * samples_val)
        cost += tf.reduce_sum(weights * kernel_val)
    return tf.sqrt(tf.maximum(cost, 0.0)), weights, samples_val


if __name__ == '__main__':
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    n_h = 450
    epoches = 1000
    batch_size = 1000
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.03
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    save_freq = 10
    image_freq = 1
    result_path = "results/gmmn"

    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x2 = tf.placeholder(tf.float32, shape=[None, n_x], name='x2')
    n = tf.shape(x)[0]

    model, samples, x_logits = generator({}, n, n_h, n_x, is_training)
    cost, weights, _ = mmd(x, batch_size, samples, batch_size)
    cost_opt, _, val = mmd(x2, batch_size, x, batch_size)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    n_gen = 100
    _, samples, _ = generator({}, n_gen, n_h, n_x, False)
    x_gen = tf.reshape(samples, [-1, 28, 28, 1])

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # get optimal cost
        opt_csts = []
        np.random.shuffle(x_train)
        for t in range(iters - 1):
            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
            x_batch2 = x_train[(t + 1) * batch_size:(t + 2) * batch_size]
            opt_cst, v = sess.run([cost_opt, val],
                                   feed_dict={x: x_batch,
                                              x2: x_batch2})
            opt_csts.append(opt_cst)
        print('Emperical Optimal Cost = {}'.format(np.mean(opt_csts)))

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            csts = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, cst = sess.run([infer, cost],
                                  feed_dict={x: x_batch,
                                             learning_rate_ph: learning_rate,
                                             is_training: True})
                #print('Cost = {}'.format(cst))
                csts.append(cst)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): MMD Cost = {}'.format(
                epoch, time_epoch, np.mean(csts)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

            if epoch % image_freq == 0:
                images = sess.run(x_gen)
                name = "results/gmmn/gmmn.epoch.{}.png".format(epoch)
                save_image_collections(images, name)