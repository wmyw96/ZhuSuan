#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ladder Variational Autoencoder on CIFAR10. (Casper, 2016)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import layers
import six
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import zhusuan as zs
    from zhusuan.distributions import logistic
    from zhusuan.utils import merge_dicts
except:
    raise ImportError()

import dataset
import utils
import multi_gpu
from multi_gpu import FLAGS


@zs.reuse('model')
def ladder_vae(observed, n, n_particles, groups):
    with zs.StochasticGraph(observed=observed) as model:
        h_top = tf.get_variable(name='h_top_p',
                                shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
        h, _ = downward(h_top, n, n_particles, groups)
        h = tf.nn.elu(h)
        x = layers.conv2d_transpose(h, 3, kernel_size=5, stride=2,
                                    activation_fn=None, scope='x_mean')
        x = tf.reshape(x, [n_particles, n, -1])
        x_logsd = tf.get_variable("x_logsd", (),
                                  initializer=tf.constant_initializer(0.0))
    return model, x, x_logsd


@zs.reuse('variational')
def q_net(x, n_xl, n_particles, groups):
    with zs.StochasticGraph() as variational:
        x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        n = tf.shape(x)[0]
        h = layers.conv2d(x, groups[0].num_filters, 5, 2, activation_fn=None)
        h, qz_mean, qz_logstd, qz_up_context = upward(h, n, n_particles, groups)
        h_top = tf.get_variable(name='h_top_q',
                                shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
        h, ar_z = downward(h_top, n, n_particles, groups,
                           lateral_inputs=[qz_mean, qz_logstd, qz_up_context])
    return variational, ar_z


def upward(h, n, n_particles, groups):
    qz_mean = {}
    qz_logstd = {}
    qz_up_context = {}
    for group_i, group in enumerate(groups):
        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)
            stride = 1
            if group_i > 0 and block_i == 0:
                stride = 2
            h1 = tf.nn.elu(h)
            h1 = layers.conv2d(h1, 2 * group.num_filters + 2 * group.n_z, 3,
                               stride=stride, activation_fn=None,
                               scope=name + '_up_conv1')
            qz_mean[name], qz_logstd[name], h1, qz_up_context[name] = tf.split(
                h1, [group.n_z] * 2 + [group.num_filters] * 2, axis=3)
            h1 = tf.nn.elu(h1)
            h1 = layers.conv2d(h1, group.num_filters, kernel_size=3,
                               activation_fn=None, scope=name + '_up_conv2')
            if stride == 2:
                h = layers.conv2d(h, group.num_filters, 3, stride=2,
                                  activation_fn=None,
                                  scope=name + '_resize_up')
            h += 0.1 * h1
    return h, qz_mean, qz_logstd, qz_up_context


def downward(h_top, n, n_particles, groups, lateral_inputs=None):
    h = tf.tile(h_top, [n * n_particles, groups[-1].map_size,
                        groups[-1].map_size, 1])
    ar_z = {}
    for group_i, group in reversed(list(enumerate(groups))):
        for block_i in reversed(range(group.num_blocks)):
            name = 'group_%d/block_%d' % (group_i, block_i)
            stride = 1
            if group_i > 0 and block_i == 0:
                stride = 2
            h1 = tf.nn.elu(h)
            if lateral_inputs:
                h1 = layers.conv2d_transpose(
                    h1, group.num_filters * 2 + group.n_z * 2, 3,
                    activation_fn=None, scope=name + '_down_conv1')
                h1, pz_down_context, pz_mean, pz_logstd = tf.split(
                    h1, [group.num_filters] * 2 + [group.n_z] * 2, axis=3)

                pz_mean, pz_logstd = [
                    tf.reshape(x, [n_particles, n, group.map_size,
                                   group.map_size, group.n_z])
                    for x in [pz_mean, pz_logstd]]
                pz_down_context = tf.reshape(pz_down_context,
                                             [n_particles, n, group.map_size,
                                              group.map_size, group.num_filters])
                qz_mean, qz_logstd, qz_up_context = lateral_inputs
                post_z_mean = pz_mean + tf.expand_dims(qz_mean[name], 0)
                post_z_logstd = pz_logstd + tf.expand_dims(qz_logstd[name], 0)
                post_z_context = pz_down_context + \
                                 tf.expand_dims(qz_up_context[name], 0)
                post_z_mean, post_z_logstd = [
                    tf.reshape(x, [n*n_particles, group.map_size,
                                   group.map_size, group.n_z])
                    for x in [post_z_mean, post_z_logstd]]
                post_z_context = tf.reshape(post_z_context,
                                            [n*n_particles, group.map_size,
                                             group.map_size, group.num_filters])
                z_i = zs.Normal(name, post_z_mean, post_z_logstd)
                z_i_samples = z_i.tensor
                log_qz_i = z_i.log_prob(z_i_samples)
                arw_mean, arw_logstd = ar_multiconv2d(
                    z_i_samples, post_z_context, [group.num_filters,
                                                  group.num_filters],
                    [group.n_z, group.n_z], name=name + 'ar_multi_conv')
                z_i_samples = (z_i_samples - arw_mean) / tf.exp(arw_logstd)
                log_qz_i += arw_logstd
                log_qz_i = tf.reduce_sum(
                    tf.reshape(log_qz_i, [n_particles, n, -1]), -1)
                ar_z[name] = [z_i_samples, log_qz_i]
            else:
                h1 = layers.conv2d_transpose(
                    h1, group.num_filters + group.n_z * 2, 3,
                    activation_fn=None, scope=name + '_down_conv1')
                h1, pz_mean, pz_logstd = tf.split(
                    h1, [group.num_filters] + [group.n_z] * 2, axis=3)
                z_i = zs.Normal(name, pz_mean, pz_logstd)

            h1 = tf.concat([h1, z_i], 3)
            h1 = tf.nn.elu(h1)
            if stride == 2:
                h = layers.conv2d_transpose(h, group.num_filters, 3, 2,
                                            scope=name + '_resize_down')
            h1 = layers.conv2d_transpose(h1, group.num_filters, 3,
                                         stride=stride,
                                         activation_fn=None,
                                         scope=name + '_down_conv2')
            h += 0.1 * h1
    return h, ar_z


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = int(n_out / n_in)
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = int(n_in / n_out)
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    l = int((h - 1) / 2)
    m = int((w - 1) / 2)
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


def ar_conv2d(name, x, num_filters, filter_size=(3, 3), stride=(1, 1),
              pad="SAME", zerodiagonal=True):
    h = filter_size[0]
    w = filter_size[1]
    n_in = int(x.get_shape()[-1])
    n_out = num_filters
    filter_shape = [h, w, n_in, n_out]
    mask = tf.constant(get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal))
    with tf.variable_scope(name):
        v = tf.get_variable("V", filter_shape, tf.float32)
    w = v * mask
    return tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=pad,
                        use_cudnn_on_gpu=True)


def ar_multiconv2d(input, context, n_h, n_out, name, nl=tf.nn.elu):
    with tf.variable_scope(name):
        for i, size in enumerate(n_h):
            input = ar_conv2d("layer_%d" % i, input, size, zerodiagonal=False)
            if i == 0:
                input += context
            input = nl(input)
        return [
            ar_conv2d("layer_out_%d" % i, input, size, zerodiagonal=True) * 0.1
            for i, size in enumerate(n_out)]

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
    epoches = 2000
    batch_size = 16 * FLAGS.num_gpus
    test_batch_size = 16 * FLAGS.num_gpus
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    print_freq = 100
    test_freq = iters
    learning_rate = 0.002
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    bottle_neck_group = namedtuple(
        'bottle_neck_group',
        ['num_blocks', 'num_filters', 'map_size', 'n_z'])
    groups = [
        bottle_neck_group(10, 160, 16, 32),
        bottle_neck_group(10, 160, 8, 32)
    ]

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9,
                                      beta2=0.999)

    def build_tower_graph(x, id_):
        x_part = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
                   (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(x_part)[0]
        x_obs = tf.tile(tf.expand_dims(x_part, 0), [n_particles, 1, 1])

        def log_joint(observed):
            obs_ = observed.copy()
            x = obs_.pop('x')
            model, x_mean, x_logsd = ladder_vae(obs_, n, n_particles, groups)
            log_pzs = model.local_log_prob(list(six.iterkeys(obs_)))
            log_pz = sum(tf.reduce_sum(
                tf.reshape(z, [n_particles, n, -1]), -1) for z in log_pzs)
            log_px_z = tf.log(logistic.cdf(x + 1. / 256, x_mean, x_logsd) -
                              logistic.cdf(x, x_mean, x_logsd) + 1e-8)
            log_px_z = tf.reduce_sum(log_px_z, -1)
            return log_px_z + log_pz

        variational, latents = q_net(x_part, n_xl, n_particles, groups)
        lower_bound = tf.reduce_mean(
            zs.advi(log_joint, {'x': x_obs}, latents, axis=0))
        bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
        grads = optimizer.compute_gradients(bits_per_dim)
        return lower_bound, bits_per_dim, grads

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                lower_bound, bits_per_dim, grads = build_tower_graph(x, i)
                tower_losses.append([lower_bound, bits_per_dim])
                tower_grads.append(grads)
    lower_bound, bits_per_dim = multi_gpu.average_losses(tower_losses)
    grads = multi_gpu.average_gradients(tower_grads)
    infer = optimizer.apply_gradients(grads)

    total_size = 0
    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())
        total_size += np.prod([int(s) for s in i.get_shape()])
    print("Num trainable variables: %d" % total_size)

    # Run the inference
    with multi_gpu.create_session() as sess:
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
                                   n_particles: lb_samples})
                except tf.errors.InvalidArgumentError as error:
                    if ("NaN" in error.message) or ("Inf" in error.message):
                        continue
                    raise
                lbs.append(lb)
                bits.append(bit)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Lower bound = {} bits = {}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(lbs), np.mean(bits)))
                    lbs = []
                    bits = []

                if iter % test_freq == 0:
                    time_test = -time.time()
                    test_lbs = []
                    test_bits = []
                    for tt in range(test_iters):
                        test_x_batch = x_test[tt * test_batch_size:
                                              (tt + 1) * test_batch_size]
                        test_lb, test_bit = sess.run(
                            [lower_bound, bits_per_dim],
                            feed_dict={x: test_x_batch,
                                       n_particles: lb_samples})
                        test_lbs.append(test_lb)
                        test_bits.append(test_bit)
                    time_test += time.time()
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test lower bound = {} bits = {}'.format(
                        np.mean(test_lbs), np.mean(test_bits)))

                if iter % print_freq == 0:
                    time_train = -time.time()
