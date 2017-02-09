#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vae 1 latent z for cifar. Prepare for Semi supervised learning.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
import six
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import zhusuan as zs
    from zhusuan.distributions import logistic
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    import utils
except:
    raise ImportError()


@zs.reuse('model')
def vae_flat(observed, n, n_x, n_z, n_particles, is_training):
    with zs.StochasticGraph(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n_particles, n_z])
        z_logstd = tf.zeros([n_particles, n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=1, n_samples=n)
        y_logits = tf.zeros([n_particles, n_y])
        y = zs.Discrete('y', y_logits, sample_dim=1, n_samples=n)

        lz = layers.fully_connected(z, 4*4*512, normalizer_fn=layers.batch_norm,
                                    normalizer_params=normalizer_params)
        lz = tf.reshape(lz, [-1, 4, 4, 512])
        lz = layers.conv2d_transpose(lz, 256, kernel_size=5, stride=2,
                                     normalizer_fn=layers.batch_norm,
                                     normalizer_params=normalizer_params)
        lz = layers.conv2d_transpose(lz, 128, kernel_size=5, stride=2,
                                     normalizer_fn=layers.batch_norm,
                                     normalizer_params=normalizer_params)
        lz = layers.conv2d_transpose(lz, 3, kernel_size=5, stride=2,
                                     activation_fn=None)
        x_mean = tf.reshape(lz, [n_particles, n, -1])
        x_logsd = tf.get_variable("x_logsd", (),
                                  initializer=tf.constant_initializer(0.0))
    return model, x_mean, x_logsd


def q_net(x, n_xl, n_z, n_particles, is_training):
    with zs.StochasticGraph() as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        lz_x = layers.conv2d(
            lz_x, 128, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 256, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 512, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.dropout(lz_x, keep_prob=0.9, is_training=is_training)
        lz_x = layers.flatten(lz_x)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, lz_logstd, sample_dim=0,
                      n_samples=n_particles)
    return variational


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    _, n_xl, _, n_channels = x_train.shape
    n_x = n_xl * n_xl * n_channels
    x_train = x_train.reshape((-1, n_x))
    x_train -= np.mean(x_train, 0)
    x_test = x_test.reshape((-1, n_x))
    x_test -= np.mean(x_test, 0)
    n_y = t_train.shape[1]

    # Define model parameters
    n_z = 100

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 3000
    batch_size = 20
    test_batch_size = 40
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    print_freq = 100
    test_freq = 2000
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)

    def log_joint(observed):
        obs_ = observed.copy()
        x = obs_.pop('x')
        model, x_mean, x_logsd = vae_flat(
            obs_, n, n_x, n_z, n_particles, is_training)
        log_pz = model.local_log_prob(['z'])
        log_px_z = tf.log(logistic.cdf(x + 1. / 256, x_mean, x_logsd) -
                          logistic.cdf(x, x_mean, x_logsd) + 1e-8)
        return tf.reduce_sum(log_pz, -1) + tf.reduce_sum(log_px_z, -1)

    variational = q_net(x, n_xl, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    log_qz = tf.reduce_sum(log_qz, -1)
    lower_bound = tf.reduce_mean(
        zs.advi(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))
    log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]},
                            axis=0))
    bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
    grads = optimizer.compute_gradients(bits_per_dim)
    infer = optimizer.apply_gradients(grads)

    total_size = 0
    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())
        total_size += np.prod([int(s) for s in i.get_shape()])
    print("Num trainable variables: %d" % total_size)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs, test_lbs, test_lls = [], [], []
            bits, test_bits = [], []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                try:
                    _, lb, bit = sess.run(
                        [infer, lower_bound, bits_per_dim],
                        feed_dict={x: x_batch,
                                   learning_rate_ph: learning_rate,
                                   n_particles: lb_samples,
                                   is_training: True})
                except tf.errors.InvalidArgumentError as error:
                    if ("NaN" in error.message) or ("Inf" in error.message):
                        continue
                    raise
                lbs.append(lb)
                bits.append(bit)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Lower bound = {} bits = {}'.format(
                        epoch, iter, (time.time() + time_train) / print_freq,
                        np.mean(lbs), np.mean(bits)))
                    lbs = []
                    bits = []
                    time_train = -time.time()

                if iter % test_freq == 0:
                    time_test = -time.time()
                    test_lbs = []
                    test_lls = []
                    test_bits = []
                    for t in range(test_iters):
                        test_x_batch = x_test[t * test_batch_size:
                        (t + 1) * test_batch_size]
                        test_lb, test_bit = sess.run(
                            [lower_bound, bits_per_dim],
                            feed_dict={x: test_x_batch,
                                       n_particles: lb_samples,
                                       is_training: False})
                        test_ll = sess.run(log_likelihood,
                                           feed_dict={x: test_x_batch,
                                                      n_particles: ll_samples,
                                                      is_training: False})
                        test_lbs.append(test_lb)
                        test_lls.append(test_ll)
                        test_bits.append(test_bit)
                    time_test += time.time()
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test lower bound = {} bits = {}'.format(
                        np.mean(test_lbs), np.mean(test_bits)))
                    print('>> Test log likelihood = {}'.format(
                        np.mean(test_lls)))

