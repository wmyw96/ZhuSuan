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
try:
    from zhusuan.model import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
    from zhusuan.distributions import logistic
    from zhusuan.optimization.adamax import AdamaxOptimizer
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_x, n, n_particles, is_training):
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        with StochasticGraph() as model:
            z3_mean = tf.zeros([n_particles*n, 4, 4, 256])
            z3_logstd = tf.zeros([n_particles*n, 4, 4, 256])
            z3 = Normal(z3_mean, z3_logstd)
            l_z2_z3 = layers.conv2d_transpose(z3.value, 256, 3)
            l_z2_z3 = layers.conv2d_transpose(l_z2_z3, 256, 3)
            l_z2_z3 = layers.conv2d_transpose(l_z2_z3, 128, 5, stride=2)

            z2_mean = layers.conv2d_transpose(l_z2_z3, 128, 3, activation_fn=None)
            z2_logstd = layers.conv2d_transpose(l_z2_z3, 128, 3, activation_fn=None)
            z2 = Normal(z2_mean, z2_logstd)
            l_z1_z2 = layers.conv2d_transpose(z2.value, 128, 3)
            l_z1_z2 = layers.conv2d_transpose(l_z1_z2, 128, 3)
            l_z1_z2 = layers.conv2d_transpose(l_z1_z2, 64, 5, stride=2)

            z1_mean = layers.conv2d_transpose(l_z1_z2, 64, 3, activation_fn=None)
            z1_logstd = layers.conv2d_transpose(l_z1_z2, 64, 3, activation_fn=None)
            z1 = Normal(z1_mean, z1_logstd)

            l_z0_z1 = layers.conv2d_transpose(z1.value, 64, 3)
            l_z0_z1 = layers.conv2d_transpose(l_z0_z1, 64, 3)
            z0_mean = layers.conv2d_transpose(l_z0_z1, 64, 3, activation_fn=None)
            z0_logstd = layers.conv2d_transpose(l_z0_z1, 64, 3, activation_fn=None)

            z0 = Normal(z0_mean, z0_logstd)

            lx_z0 = layers.conv2d_transpose(z0.value, 64, 5, stride=2)
            x = layers.conv2d_transpose(lx_z0, 3, 3, activation_fn=None)
            x = tf.reshape(x, [n_particles, n, n_x])
        self.model = model
        self.x = x
        self.zs = [z0, z1, z2, z3]
        self.n_particles = n_particles
        self.n_x = n_x
        self.x_logsd = tf.get_variable("x_logsd", (),
                                       initializer=tf.constant_initializer(
                                           0.0))

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        z0, z1, z2, z3 = latent['z0'], latent['z1'], latent['z2'], latent['z3']
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        z0_out, z1_out, z2_out, z3_out, x_out = self.model.get_output(
            self.zs + [self.x],
            inputs={self.zs[0]: z0,
                    self.zs[1]: z1,
                    self.zs[2]: z2,
                    self.zs[3]: z3
                    })
        x_mean = x_out[0]
        log_px_z0 = tf.log(logistic.cdf(x + 1. / 256, x_mean, self.x_logsd) -
                           logistic.cdf(x, x_mean, self.x_logsd) + 1e-8)
        log_px_z0 = tf.reduce_sum(log_px_z0, -1)
        log_pz0_z1 = tf.reduce_sum(
            tf.reshape(z0_out[1], [n_particles, -1, 16, 16, 64]), [2, 3, 4])
        log_pz1_z2 = tf.reduce_sum(
            tf.reshape(z1_out[1], [n_particles, -1, 16, 16, 64]), [2, 3, 4])
        log_pz2_z3 = tf.reduce_sum(
            tf.reshape(z2_out[1], [n_particles, -1, 8, 8, 128]), [2, 3, 4])
        log_pz3 = tf.reduce_sum(
            tf.reshape(z3_out[1], [n_particles, -1, 4, 4, 256]), [2, 3, 4])

        return log_px_z0 + log_pz0_z1 + log_pz1_z2 + log_pz2_z3 + log_pz3

    def reconst(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        z0, z1, z2, z3 = latent['z0'], latent['z1'], latent['z2'], latent['z3']
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        z0_out, z1_out, z2_out, z3_out, x_out = self.model.get_output(
            self.zs + [self.x],
            inputs={self.zs[0]: z0,
                    self.zs[1]: z1,
                    self.zs[2]: z2,
                    self.zs[3]: z3
                    })
        x_mean = x_out[0]
        log_px_z0 = tf.log(logistic.cdf(x + 1. / 256, x_mean, self.x_logsd) -
                           logistic.cdf(x, x_mean, self.x_logsd) + 1e-8)
        log_px_z0 = tf.reduce_sum(log_px_z0, -1)
        log_pz0_z1 = tf.reduce_sum(
            tf.reshape(z0_out[1], [n_particles, -1, 16, 16, 64]), [2, 3, 4])
        log_pz1_z2 = tf.reduce_sum(
            tf.reshape(z1_out[1], [n_particles, -1, 16, 16, 64]), [2, 3, 4])
        log_pz2_z3 = tf.reduce_sum(
            tf.reshape(z2_out[1], [n_particles, -1, 8, 8, 128]), [2, 3, 4])
        log_pz3 = tf.reduce_sum(
            tf.reshape(z3_out[1], [n_particles, -1, 4, 4, 256]), [2, 3, 4])
        return [log_px_z0, log_pz0_z1, log_pz1_z2, log_pz2_z3, log_pz3]


def q_net(x, n_xl, n_particles, is_training):
    """
        Build the recognition network (Q-net) used as variational posterior.

        :param x: A Tensor.
        :param n_xl: A Tensor or int. The dimension of observed variables (x) width.
        :param n_particles: A Tensor or int. Number of samples of latent variables.
        """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    with StochasticGraph() as variational:
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        lz0_x = layers.conv2d(x, 64, 3)
        lz0_x = layers.conv2d(lz0_x, 64, 3)
        lz0_x = layers.conv2d(lz0_x, 64, 5, stride=2)
        lz0_mean = layers.conv2d(lz0_x, 64, 3, activation_fn=None)
        lz0_logstd = layers.conv2d(lz0_x, 64, 3, activation_fn=None)
        lz0 = Normal(lz0_mean, lz0_logstd)
        lz1_z0 = layers.conv2d(lz0.value, 64, 3)
        lz1_z0 = layers.conv2d(lz1_z0, 64, 3)
        lz1_mean = layers.conv2d(lz1_z0, 64, 3, activation_fn=None)
        lz1_logstd = layers.conv2d(lz1_z0, 64, 3, activation_fn=None)
        lz1 = Normal(lz1_mean, lz1_logstd)
        lz2_z1 = layers.conv2d(lz1.value, 64, 3)
        lz2_z1 = layers.conv2d(lz2_z1, 64, 3)
        lz2_z1 = layers.conv2d(lz2_z1, 128, 5, stride=2)
        lz2_mean = layers.conv2d(lz2_z1, 128, 3, activation_fn=None)
        lz2_logstd = layers.conv2d(lz2_z1, 128, 3, activation_fn=None)
        lz2 = Normal(lz2_mean, lz2_logstd)
        lz3_z2 = layers.conv2d(lz2.value, 128, 3)
        lz3_z2 = layers.conv2d(lz3_z2, 128, 3)
        lz3_z2 = layers.conv2d(lz3_z2, 256, 5, stride=2)
        lz3_mean = layers.conv2d(lz3_z2, 256, 3, activation_fn=None)
        lz3_logstd = layers.conv2d(lz3_z2, 256, 3, activation_fn=None)
        lz3 = Normal(lz3_mean, lz3_logstd)
        return variational, lz0, lz1, lz2, lz3


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    print(x_train.max(), x_train.min())
    _, n_xl, _, n_channels = x_train.shape
    n_x = n_xl * n_xl * n_channels
    x_train = x_train.reshape((-1, n_x))
    x_train -= np.mean(x_train, 0)
    x_test = x_test.reshape((-1, n_x))
    x_test -= np.mean(x_test, 0)
    n_y = t_train.shape[1]

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 3000
    batch_size = 16
    test_batch_size = 64
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    # optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-8)
    optimizer = AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)
    model = M1(n_x, n, n_particles, is_training)
    variational, lz0, lz1, lz2, lz3 = q_net(x, n_xl, n_particles, is_training)
    z0_output, z1_output, z2_output, z3_output = variational.get_output([lz0, lz1, lz2, lz3])
    z0_logpdf = tf.reduce_sum(
        tf.reshape(z0_output[1], [n_particles, n, 16, 16, 64]), [2, 3, 4])
    z1_logpdf = tf.reduce_sum(
        tf.reshape(z1_output[1], [n_particles, n, 16, 16, 64]), [2, 3, 4])
    z2_logpdf = tf.reduce_sum(
        tf.reshape(z2_output[1], [n_particles, n, 8, 8, 128]), [2, 3, 4])
    z3_logpdf = tf.reduce_sum(
        tf.reshape(z3_output[1], [n_particles, n, 4, 4, 256]), [2, 3, 4])
    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, {'z0': [z0_output[0], z0_logpdf],
                          'z1': [z1_output[0], z1_logpdf],
                          'z2': [z2_output[0], z2_logpdf],
                          'z3': [z3_output[0], z3_logpdf]},
        reduction_indices=0))
    reconst = [tf.reduce_mean(i) for i in model.reconst(
        {'z0': z0_output[0], 'z1': z1_output[0], 'z2': z2_output[0], 'z3': z3_output[0]},
        {'x': x}, {})]

    log_likelihood = tf.reduce_mean(is_loglikelihood(
        model, {'x': x}, {'z0': [z0_output[0], z0_logpdf],
                          'z1': [z1_output[0], z1_logpdf],
                          'z2': [z2_output[0], z2_logpdf],
                          'z3': [z3_output[0], z3_logpdf]},
        reduction_indices=0))
    bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
    grads = optimizer.compute_gradients(bits_per_dim)

    def l2_norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x)))


    update_ratio = learning_rate_ph * tf.reduce_mean(tf.pack(list(
        (l2_norm(k) / (l2_norm(v) + 1e-8)) for k, v in grads
        if k is not None)))

    infer = optimizer.apply_gradients(grads)

    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    total_size = 0
    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())
        total_size += np.prod([int(s) for s in i.get_shape()])
    print("Num trainable variables: %d" % total_size)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            bits = []
            update_ratios = []
            recs = []
            recs1 = []
            recs2 = []
            recs3 = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                try:
                    _, lb, bit, update_ratio_, rec0, rec1, rec2, rec3 = sess.run(
                        [infer, lower_bound, bits_per_dim, update_ratio, reconst[0], reconst[1], reconst[2], reconst[3]],
                        feed_dict={x: x_batch,
                                   learning_rate_ph: learning_rate,
                                   n_particles: lb_samples,
                                   is_training: True})

                except tf.errors.InvalidArgumentError as error:
                    if ("NaN" in error.message) or ("Inf" in error.message):
                        continue
                    raise error
                lbs.append(lb)
                bits.append(bit)
                update_ratios.append(update_ratio_)
                recs.append(rec0)
                recs1.append(rec1)
                recs2.append(rec2)
                recs3.append(rec3)
                # print('Iter {} ({:.1f}s): Lower bound = {} bits = {}'.format(
                #     t, time.time()+time_epoch, lb, bit))
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {} bits = {} rec = {} logpz0 = {} logpz1 = {} logpz2 = {}'.format(
                epoch, time_epoch, np.mean(lbs), np.mean(bits), np.mean(recs), np.mean(recs1), np.mean(recs2), np.mean(recs3)))
            print('update ratio = {}'.format(np.mean(update_ratios)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                test_bits = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                                   t * test_batch_size: (
                                                        t + 1) * test_batch_size]
                    test_lb, test_bit = sess.run([lower_bound, bits_per_dim],
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
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))