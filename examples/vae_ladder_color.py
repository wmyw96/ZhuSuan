#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vae_ladder for CIFAR10. Without autoregressive connections. Separate p(x|z)
and q(z|x).
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
from collections import namedtuple
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


@zs.reuse('model')
def vae_iaf(observed, n, n_particles, groups, is_training):
    with zs.StochasticGraph(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        h_top = tf.get_variable(name='h_top_p', shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
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
                    h1, group.n_z * 2 + group.num_filters, 3,
                    activation_fn=None, scope=name + '_down_conv1')
                h1, pz_mean, pz_logstd = split(
                    h1, -1, [group.num_filters] + [group.n_z] * 2
                )

                prior[name] = zs.Normal(name, pz_mean, pz_logstd)
                z = prior[name]
                z = tf.reshape(z, [-1, group.map_size, group.map_size,
                                   group.n_z])
                h1 = tf.concat([h1, z], 3)
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
        x_logsd = tf.get_variable("x_logsd", (),
                                  initializer=tf.constant_initializer(0.0))
    return model, x, x_logsd


def q_net(x, n_xl, n_particles, groups, is_training):
    with zs.StochasticGraph() as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
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
                                      scope=name + '_resize_up')
                h += 0.1 * h1

        h_top = tf.get_variable(name='h_top_q', shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
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
                    h1, group.n_z * 2 + group.num_filters, 3,
                    activation_fn=None, scope=name + '_down_conv1')
                rz_mean, rz_logstd, h1 = split(
                    h1, -1, [group.n_z] * 2 + [group.num_filters]
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
                posterior[name] = zs.Normal(name, post_z_mean, post_z_logstd)
                z = posterior[name]
                z = tf.reshape(z,
                               [-1, group.map_size, group.map_size, group.n_z])
                h1 = tf.concat([h1, z], 3)
                h1 = tf.nn.elu(h1)
                if stride == 2:
                    h = layers.conv2d_transpose(h, group.num_filters, 3, 2,
                                                scope=name + '_resize_down')
                h1 = layers.conv2d_transpose(h1, group.num_filters, 3,
                                             stride=stride, activation_fn=None,
                                             scope=name + '_down_conv2')
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
        bottle_neck_group(2, 64, 16, 64),
        bottle_neck_group(2, 64, 8, 64),
        bottle_neck_group(2, 64, 4, 64)
    ]

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)

    def log_joint(latent, observed, given):
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        model, x_mean, x_logsd = vae_iaf(latent, n, n_particles, groups,
                                         is_training)
        log_pzs = model.local_log_prob(list(six.iterkeys(latent)))
        log_pz = sum(tf.reduce_sum(
            tf.reshape(z, [n_particles, n, -1]), -1) for z in log_pzs)
        log_px_z = tf.log(logistic.cdf(x + 1. / 256, x_mean, x_logsd) -
                          logistic.cdf(x, x_mean, x_logsd) + 1e-8)
        log_px_z = tf.reduce_sum(log_px_z, -1)
        return log_px_z + log_pz

    variational, post = q_net(x, n_xl, n_particles, groups, is_training)
    qzs_samples_logs = variational.query(list(six.iterkeys(post)), outputs=True,
                                         local_log_prob=True)
    zs_ = [[z_output[0],
            tf.reduce_sum(tf.reshape(z_output[1], [n_particles, n, -1]), -1)]
           for z_output in qzs_samples_logs]
    latents = dict(zip(list(six.iterkeys(post)), zs_))
    lower_bound = tf.reduce_mean(
        zs.advi(log_joint, {'x': x}, latents,
                axis=0))
    log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x}, latents,
                            axis=0))
    bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
    grads = optimizer.compute_gradients(bits_per_dim)
    infer = optimizer.apply_gradients(grads)

    def l2_norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    update_ratio = learning_rate_ph * tf.reduce_mean(tf.stack(list(
        (l2_norm(k) / (l2_norm(v) + 1e-8)) for k, v in grads
        if k is not None)))

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
        sess.run(tf.global_variables_initializer())
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
                                   t * test_batch_size:
                                   (t + 1) * test_batch_size]
                    test_lb, test_bit = sess.run([lower_bound, bits_per_dim],
                                                 feed_dict={
                                                     x: test_x_batch,
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