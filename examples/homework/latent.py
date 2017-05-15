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
def mixture_gaussian(observed, n, n_x, cls, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        z_logits = tf.get_variable('z_logits', shape=[1, cls],
                                   initializer=tf.random_normal_initializer(
                                       0, 1))
        para = z_logits
        z_logits = tf.tile(z_logits, [n, 1])
        z = zs.OnehotCategorical('z', logits=z_logits, n_samples=n_particles,
                                 group_event_ndims=0, dtype=tf.float32)
        # [n_particles, n, cls]
        z = tf.identity(z)
        x_mean = tf.get_variable('x_mean', shape=[cls, n_x],
                                 initializer=tf.random_normal_initializer(
                                     0, 1.0))
        m_para = x_mean
        x_logstd = tf.get_variable('x_logstd', shape=[cls, n_x],
                                   initializer=tf.random_normal_initializer(
                                        0, 0.1))
        s_para = x_logstd
        # [n_particles, cls, n_x]
        x_mean = tf.tile(tf.expand_dims(x_mean, 0), [n_particles, 1, 1])
        x_mean = tf.matmul(z, x_mean)
        x_logstd = tf.tile(tf.expand_dims(x_logstd, 0), [n_particles, 1, 1])
        x_logstd = tf.matmul(z, x_logstd)
        x = zs.Normal('x', x_mean, x_logstd, group_event_ndims=1)
    return model, [tf.squeeze(tf.nn.softmax(para, 1), 0), m_para, s_para], \
           tf.identity(x)


@zs.reuse('variational')
def q_net(observed, x, cls, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = layers.fully_connected(
            tf.to_float(x), 20, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_logits = layers.fully_connected(lz_x, cls, activation_fn=None)
        z = zs.OnehotCategorical('z', z_logits,
                                 n_samples=n_particles, group_event_ndims=0,
                                 dtype=tf.float32)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1237)

    N = 400
    D = n_x = 2
    cls = 3
    lb_samples = 10
    ll_samples = 100
    epoches = 1000
    batch_size = 100
    iters = N // batch_size
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75
    save_freq = 100
    image_freq = 2
    result_path = "results/mixture_gaussian"

    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model, _, _ = mixture_gaussian(observed, n, n_x, cls, n_particles,
                                       is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net({}, x, cls, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    print('--')
    print(qz_samples.get_shape())
    print(log_qz.get_shape())
    print('--')

    cost, lower_bound = zs.rws(log_joint, {'x': x_obs},
                                 {'z': [qz_samples, log_qz]}, axis=0)
    lower_bound = tf.reduce_mean(lower_bound)
    cost = tf.reduce_mean(cost)

    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        # generate data
        sess.run(tf.global_variables_initializer())
        print(N)
        _, paras, samples = mixture_gaussian({}, N, D, cls, 1, False)
        samples = tf.squeeze(samples)
        true_category, true_mean, true_std, x_train = \
            sess.run(paras + [samples])

        print(x_train)
        print(true_category)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=0.3)
        plt.scatter(true_mean[:, 0], true_mean[:, 1])

        sess.run(tf.global_variables_initializer())

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
            lbs = []
            cnt = 0
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            estimate_category, estimate_mean, estimate_std = sess.run(paras)
            m = np.mean((estimate_mean - true_mean) * (estimate_mean - true_mean))
            s = np.mean((estimate_std - true_std) * (estimate_std - true_std))
            print(estimate_category)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}, '
                  'MSE = {} (mean), {} (log_std)'.format(
                  epoch, time_epoch, np.mean(lbs), m, s))

        plt.scatter(estimate_mean[:, 0], estimate_mean[:, 1], c='r')
        plt.show()
