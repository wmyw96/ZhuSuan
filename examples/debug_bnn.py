#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range, zip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dataset
import matplotlib.pyplot as plt


# MADE
def random_weights(n_in, n_out):
    return tf.random_normal(shape=(n_in, n_out), mean=0,
                            stddev=np.sqrt(2 / n_in), dtype=tf.float32)


def random_bias(n_out):
    return tf.constant([0] * n_out, dtype=tf.float32)


def get_linear_mask(input_pri, output_pri, hidden):
    layers = [len(input_pri)] + hidden + [len(output_pri)]
    max_pri = max(input_pri)
    print((hidden[0] / len(input_pri)))
    priority = [input_pri]
    for units in hidden:
        priority += [[j for j in range(max_pri)
                      for k in range(int(units / len(input_pri)))]]
    priority += [output_pri]
    mask = []
    for l in range(len(layers) - 1):
        # z_{j} = z_{i} * W_{ij}
        maskl = np.zeros((layers[l], layers[l + 1]))
        for i in range(layers[l]):
            for j in range(layers[l + 1]):
                maskl[i][j] = (priority[l][i] <= priority[l + 1][j]) * 1.0
        mask.append(maskl)
    return mask


def made(name, id, z, hidden, multiple=3, hidden_layers=2):
    static_z_shape = z.get_shape()
    if not static_z_shape[-1:].is_fully_defined():
        raise ValueError('Inputs {} has undefined last dimension.'.format(z))
    d = int(static_z_shape[-1])

    units = multiple * d
    layer_unit = [d] + [units] * hidden_layers + [2 * d]
    mask = get_linear_mask([i + 1 for i in range(d)],
                           [i for i in range(d)] * 2, [units] * hidden_layers)

    with tf.name_scope(name + '%d' % id):
        #layer = tf.concat([z, hidden], static_z_shape.ndims - 1,
        #                  name='layer_0')
        layer = z
        layer = tf.reshape(layer, [-1, d])
        for i in range(hidden_layers):
            w = tf.Variable(random_weights(layer_unit[i], layer_unit[i + 1]))
            w = w * tf.constant(mask[i], dtype=tf.float32)
            b = tf.Variable(random_bias(layer_unit[i + 1]))
            linear = tf.matmul(layer, w) + b
            layer = tf.nn.relu(linear, name='layer_%d' % (i + 1))

        m_w = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        m_w = m_w * tf.constant(mask[hidden_layers][:, :d], dtype=tf.float32)
        m_b = tf.Variable(random_bias(d))
        m = tf.matmul(layer, m_w) + m_b

        s_w = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        s_w = s_w * tf.constant(mask[hidden_layers][:, d:], dtype=tf.float32)
        s_b = tf.Variable(random_bias(d))
        s = tf.matmul(layer, s_w) + s_b

    m = tf.reshape(m, tf.shape(z))
    s = tf.reshape(s, tf.shape(z))

    return m, s

def matrixiaf(name, sample, hidden, log_prob, autoregressiveNN, iters, update='normal'):
    joint_prob = log_prob
    z = sample
    m = s = None

    static_z_shape = z.get_shape()
    ndim = static_z_shape.ndims
    D2 = int(static_z_shape[ndim - 2])
    D1 = int(static_z_shape[ndim - 1])

    perm = [0, 1, 3, 2]

    for iter in range(iters):
        if update == 'normal':
            if D1 > 1:
                m, s = autoregressiveNN(name + 'siaf_col', iter, z, hidden)
                z = s * z + m
                joint_prob = joint_prob - tf.reduce_sum(tf.log(s), axis=[-1,-2])
                z = tf.reverse(z, [-1])

            if D2 > 1:
                z = tf.transpose(z, perm)
                m, s = autoregressiveNN(name + 'siaf_row', iter, z, hidden)
                z = s * z + m
                joint_prob = joint_prob - tf.reduce_sum(tf.log(s), axis=[-1,-2])
                z = tf.reverse(z, [-1])
                z = tf.transpose(z, perm)

        if update == 'gru':
            if D1 > 1:
                m, s = autoregressiveNN(name + 'siaf_col', iter, z, hidden)
                sigma = tf.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                joint_prob = joint_prob - tf.reduce_sum(tf.log(sigma), axis=[-1,-2])
                z = tf.reverse(z, [-1])

            if D2 > 1:
                z = tf.transpose(z, perm)
                m, s = autoregressiveNN(name + 'siaf_row', iter, z, hidden)
                sigma = tf.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                joint_prob = joint_prob - tf.reduce_sum(tf.log(sigma), axis=[-1,-2])
                z = tf.reverse(z, [-1])
                z = tf.transpose(z, perm)

    return z, joint_prob


def flat_flow(sample, hidden, log_prob):
    sample_shape = tf.shape(sample)
    log_prob_shape = tf.shape(log_prob)
    static_sample_shape = sample.get_shape()
    ndim = static_sample_shape.ndims
    D2 = int(static_sample_shape[ndim - 2])
    D1 = int(static_sample_shape[ndim - 1])
    z = tf.reshape(sample, [-1, D1 * D2])
    log_prob = tf.reshape(log_prob, [-1])
    z, log_prob = zs.planar_normalizing_flow(z, log_prob, 20)
    #z, log_prob = zs.inv_autoregressive_flow(z, None, log_prob, made, 2,
    #                                         'gru')

    z = tf.reshape(z, sample_shape)
    log_prob = tf.reshape(log_prob, log_prob_shape)
    return z, log_prob


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles, y_sd):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            w_logstd = tf.ones([1, n_out, n_in + 1]) * tf.log(10.0)
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                n_samples=n_particles, group_event_ndims=2))

        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_std = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(1.))
        y_log_std = tf.log(y_std)
        log_y_sd = tf.log(tf.convert_to_tensor(y_sd, dtype=tf.float32))
        y = zs.Normal('y', y_mean, log_y_sd)

    return model, y_mean, y_std


def mean_field_variational(layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, w_logstd,
                          n_samples=n_particles, group_event_ndims=2))
    return variational


def toy_data(data_size):
    x = np.random.uniform(-4, 4, data_size)
    epsilon = np.random.normal(0, 3, data_size)
    y = x * x * x + epsilon
    x = x.reshape((data_size, 1))
    y = y.reshape((data_size))
    return x, y


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load UCI Boston housing data
    # Load toy data set
    x_train, y_train = toy_data(20)
    x_test = np.arange(-6, 6, 0.01).reshape((-1, 1))
    y_test = x_test * x_test * x_test
    y_test = y_test.reshape((-1))
    N, n_x = x_train.shape
    plt.plot(x_train.reshape(-1), y_train, 'ro')
    plt.plot(x_test.reshape(-1), y_test, color='black')
    plt.axis([-6, 6, -100, 100])
    # plt.show()

    # Standardize data
    x_train, x_test, mean_x_train, std_x_train = dataset.standardize(x_train,
                                                                     x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train.reshape((-1, 1)), y_test.reshape((-1, 1)))
    y_sd = 3 / std_y_train
    print(y_sd)
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    std_y_train = std_y_train.squeeze()

    # Define model parameters
    n_hiddens = [100]

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epoches = 1000
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]


    def log_joint(observed):
        model, _, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles, y_sd)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N, \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N


    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))

    # add flow for bnn
    latentf = {}
    tot = 0
    valist = []
    for (key, value) in latent.items():
        name = key
        qsample = value[0]
        qlog_prob = value[1]
        qsample, qlog_prob = matrixiaf(name, qsample, None, qlog_prob,
                                       made, 5, update='gru')
        #qsample, qlog_prob = flat_flow(qsample, None, qlog_prob)
        va = tf.reduce_mean(qsample * qsample, [-1, -2])
        tot = tot + 1
        valist.append(va)
        latentf[name] = [qsample, qlog_prob]
    tva = sum(valist) / len(valist)
    tva = tf.reduce_mean(tva, [0, 1])
    latent = latentf

    lower_bound, reconstruction = \
        zs.sgvb(log_joint, {'y': y_obs}, latent, axis=0)
    lower_bound = tf.reduce_mean(lower_bound)
    reconstruction = tf.reduce_mean(reconstruction)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y_obs})
    model, y_mean, para_std = bayesianNN(observed, x, n_x, layer_sizes, n_particles, y_sd)
    y_pred = tf.reduce_mean(y_mean, 0)
    y_var = tf.sqrt(tf.reduce_mean(y_mean * y_mean, 0) - \
                    tf.reduce_mean(y_mean, 0) * tf.reduce_mean(y_mean, 0)) \
            * tf.convert_to_tensor(std_y_train, dtype=tf.float32)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * \
           tf.convert_to_tensor(std_y_train, dtype=tf.float32)
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
                     tf.log(tf.convert_to_tensor(std_y_train,
                                                 dtype=tf.float32))

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            res = []
            errs = []
            vas = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb, re, error, va, psd = sess.run(
                    [infer, lower_bound, rmse, reconstruction, tva, para_std],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
                res.append(re)
                errs.append(error)
                vas.append(va)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}, RMSE = {}, '
                  'RC ERROR = {}, Weight Variations = {}, KL = {}'.format(epoch,
                                         time_epoch,
                                         np.mean(lbs),
                                         np.mean(res),
                                         np.mean(errs),
                                         np.mean(vas),
                                         np.mean(errs) - np.mean(lbs)))
        ym, yv, test_re, sd = sess.run([y_pred, y_var, rmse, para_std],
                                   feed_dict={n_particles: ll_samples,
                                              x: x_test,
                                              y: y_test})
        print('Test RMSE = {}, PARA std = {}'.format(test_re, sd))
        ym = ym.reshape(-1)
        fx = x_test.reshape(-1) * std_x_train + mean_x_train
        fy = ym * std_y_train + mean_y_train
        fyl = fy - 3 * yv
        fyh = fy + 3 * yv
        plt.plot(fx, fy, color='red')
        plt.fill_between(fx, fyl, fyh, where=fyl < fyh,
                         facecolor='grey', interpolate=True)
        plt.show()
