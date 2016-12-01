#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    from zhusuan.variational import advi, rws
    from zhusuan.evaluation import is_loglikelihood
    from zhusuan.distributions import logistic
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    import utils
except:
    raise ImportError()


def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_x, n, n_particles, groups, is_training):
        self.n_particles = n_particles
        self.n = n
        self.n_x = n_x
        self.x_logsd = tf.get_variable("x_logsd", (),
                                       initializer=tf.constant_initializer(
                                           0.0))
        with StochasticGraph() as model:
            h_top = tf.get_variable('h_top', [1, 1, 1, groups[-1].num_filters],
                                    initializer=tf.zeros_initializer)
            h = tf.tile(h_top,
                        [n * n_particles, groups[-1].map_size,
                         groups[-1].map_size, 1])
            prior = {}
            for group_i, group in reversed(list(enumerate(groups))):
                for block_i in reversed(range(group.num_blocks)):
                    name = 'group_%d/block_%d' % (group_i, block_i)
                    stride = 1
                    if group_i > 0 and block_i == 0:
                        stride = 2
                    h1 = tf.nn.elu(h)
                    h1 = layers.conv2d_transpose(
                        h1, group.n_z * 2, 3,
                        activation_fn=None, scope=name + '_down_conv1')
                    pz_mean, pz_logstd = split(
                        h1, -1, [group.n_z] * 2
                    )

                    prior[name] = Normal(pz_mean, pz_logstd)
                    z = prior[name].value
                    z = tf.reshape(z, [-1, group.map_size, group.map_size,
                                       group.n_z])
                    h1 = z
                    h1 = tf.nn.elu(h1)
                    if stride == 2:
                        h = layers.conv2d_transpose(h, group.num_filters, 3, 2,
                                                    scope=name + '_resize_down')
                    h1 = layers.conv2d_transpose(h1, group.num_filters, 3,
                                                 stride=stride,
                                                 activation_fn=None,
                                                 scope=name + '_down_conv2')
                    h += 0.1 * h1
            h = tf.nn.elu(h)
            x = layers.conv2d_transpose(h, 3, kernel_size=5, stride=2,
                                        activation_fn=None, scope='x_mean')
            x = tf.reshape(x, [n_particles, n, -1])
        self.model = model
        self.x = x
        self.prior = prior

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """

        zs = list(six.itervalues(self.prior))
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        outputs = self.model.get_output(
            zs + [self.x],
            inputs=dict(zip(zs, six.itervalues(latent))))
        zs_out = outputs[:-1]
        x_mean = outputs[-1][0]
        log_px_z0 = tf.log(logistic.cdf(x + 1. / 256, x_mean, self.x_logsd) -
                           logistic.cdf(x, x_mean, self.x_logsd) + 1e-8)
        log_px_z0 = tf.reduce_sum(log_px_z0, -1)

        log_pzs = sum(tf.reduce_sum(
            tf.reshape(z[1], [n_particles, n, -1]), -1) for z in zs_out)

        return log_px_z0 + log_pzs


def q_net(x, n_xl, n_particles, groups, is_training):
    """
        Build the recognition network (Q-net) used as variational posterior.

        :param x: A Tensor.
        :param n_xl: A Tensor or int. The dimension of observed variables (x) width.
        :param n_particles: A Tensor or int. Number of samples of latent variables.
        """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    with StochasticGraph() as variational:
        x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        h = layers.conv2d(x, groups[0].num_filters, 5, 2, activation_fn=None)
        qz_mean = {}
        qz_logstd = {}
        posterior = {}

        for group_i, group in enumerate(groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)
                stride = 1
                if group_i > 0 and block_i == 0:
                    stride = 2
                h1 = tf.nn.elu(h)
                h1 = layers.conv2d(h1, group.num_filters + 2 * group.n_z, 3,
                                   stride=stride, activation_fn=None,
                                   scope=name + '_up_conv1')
                qz_mean[name], qz_logstd[name], h1 = split(
                    h1, -1, [group.n_z] * 2 + [group.num_filters])
                h1 = tf.nn.elu(h1)
                h1 = layers.conv2d(h1, group.num_filters, kernel_size=3,
                                   activation_fn=None, scope=name + '_up_conv2')
                if stride == 2:
                    h = layers.conv2d(h, group.num_filters, 3, stride=2,
                                      activation_fn=None,
                                      scope=name+'_resize_up')
                h += 0.1 * h1

        h_top = tf.get_variable('h_top', [1, 1, 1, groups[-1].num_filters],
                                initializer=tf.zeros_initializer)
        h = tf.tile(h_top, [tf.shape(x)[0] * n_particles, groups[-1].map_size,
                            groups[-1].map_size, 1])

        for group_i, group in reversed(list(enumerate(groups))):
            for block_i in reversed(range(group.num_blocks)):
                name = 'group_%d/block_%d' % (group_i, block_i)
                stride = 1
                if group_i > 0 and block_i == 0:
                    stride = 2
                h1 = tf.nn.elu(h)
                h1 = layers.conv2d_transpose(
                    h1, group.n_z * 2, 3,
                    activation_fn=None, scope=name + '_down_conv1')
                rz_mean, rz_logstd = split(
                    h1, -1, [group.n_z] * 2
                )
                rz_mean, rz_logstd = map(
                    lambda k: tf.reshape(k, [n_particles, -1,
                                             group.map_size, group.map_size,
                                             group.n_z]),
                    [rz_mean, rz_logstd])
                post_z_mean = rz_mean + tf.expand_dims(qz_mean[name], 0)
                post_z_logstd = rz_logstd + tf.expand_dims(qz_logstd[name], 0)
                post_z_mean, post_z_logstd = map(
                    lambda k: tf.reshape(k, [-1,
                                             group.map_size, group.map_size,
                                             group.n_z]),
                    [post_z_mean, post_z_logstd])
                posterior[name] = Normal(post_z_mean, post_z_logstd)
                z = posterior[name].value
                z = tf.reshape(z, [-1, group.map_size, group.map_size, group.n_z])
                h1 = z
                h1 = tf.nn.elu(h1)
                if stride == 2:
                    h = layers.conv2d_transpose(h, group.num_filters, 3, 2,
                                                scope=name+'_resize_down')
                h1 = layers.conv2d_transpose(h1, group.num_filters, 3,
                                             stride=stride, activation_fn=None,
                                             scope=name+'_down_conv2')
                h += 0.1 * h1
        return variational, posterior

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
    test_batch_size = 24
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    bottle_neck_group = namedtuple(
        'bottle_neck_group',
        ['num_blocks', 'num_filters', 'map_size', 'n_z'])
    groups = [
        bottle_neck_group(4, 64, 16, 64),
        bottle_neck_group(4, 64, 8, 64),
        bottle_neck_group(4, 64, 4, 64)
    ]

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    # optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-8)
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)
    with tf.variable_scope('model'):
        model = M1(n_x, n, n_particles, groups, is_training)
    with tf.variable_scope('variational'):
        variational, post = q_net(x, n_xl, n_particles, groups, is_training)
    zs_outputs = variational.get_output(list(six.itervalues(post)))
    zs_ = [[z_output[0],
            tf.reduce_sum(tf.reshape(z_output[1], [n_particles, n, -1]), -1)]
           for z_output in zs_outputs]

    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, latent=dict(zip(list(six.iterkeys(post)), zs_)),
        reduction_indices=0))

    log_likelihood = tf.reduce_mean(is_loglikelihood(
        model, {'x': x}, latent=dict(zip(six.iterkeys(post), zs_)),
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
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                try:
                    _, lb, bit, update_ratio_ = sess.run(
                        [infer, lower_bound, bits_per_dim, update_ratio],
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
                # print('Iter {} ({:.1f}s): Lower bound = {} bits = {}'.format(
                #     t, time.time()+time_epoch, lb, bit))
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {} bits = {}'.format(
                epoch, time_epoch, np.mean(lbs), np.mean(bits)))
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