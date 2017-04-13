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


lda_mean0 = 4.0
lda_log_std0 = 0.0


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


def made(name, id, z, hidden, multiple=1, hidden_layers=1):
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
                joint_prob = joint_prob - \
                             tf.reduce_sum(tf.log(s), axis=[-1,-2])
                z = tf.reverse(z, [-1])

            if D2 > 1:
                z = tf.transpose(z, perm)
                m, s = autoregressiveNN(name + 'siaf_row', iter, z, hidden)
                z = s * z + m
                joint_prob = joint_prob - \
                             tf.reduce_sum(tf.log(s), axis=[-1,-2])
                z = tf.reverse(z, [-1])
                z = tf.transpose(z, perm)

        if update == 'gru':
            if D1 > 1:
                m, s = autoregressiveNN(name + 'siaf_col', iter, z, hidden)
                sigma = tf.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                joint_prob = joint_prob - \
                             tf.reduce_sum(tf.log(sigma), axis=[-1,-2])
                z = tf.reverse(z, [-1])

            if D2 > 1:
                z = tf.transpose(z, perm)
                m, s = autoregressiveNN(name + 'siaf_row', iter, z, hidden)
                sigma = tf.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                joint_prob = joint_prob - \
                             tf.reduce_sum(tf.log(sigma), axis=[-1,-2])
                z = tf.reverse(z, [-1])
                z = tf.transpose(z, perm)

    return z, joint_prob


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles, tau, lda):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            w_logstd = tf.ones([1, n_out, n_in + 1]) * tf.log(lda)
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                n_samples=n_particles, group_event_ndims=2))
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
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, tf.log(tau))
        #y = zs.Normal('y', y_mean, tf.log(1.0))

    return model, y_mean


def mean_field_variational(x, y_obs, n_x, layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            #w_mean = tf.zeros([1, n_out, n_in + 1])
            #w_logstd = tf.zeros([1, n_out, n_in + 1])
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


def mean(x):
    return sum(x) / len(x)


def var(x):
    return (mean([y * y for y in x]) - mean(x)**2) * (len(x) - 1) / (len(x))


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1237)

    # Load UCI Boston housing data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.hstack([y_train, y_valid]).astype('float32')
    x_test = x_test.astype('float32')
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, _, _ = dataset.standardize(
        y_train.reshape((-1, 1)), y_test.reshape((-1, 1)))
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    # Define model parameters
    n_hiddens = [50]

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75
    para_taus = [0.35, 0.36, 0.34, 0.37, 0.33, 0.38, 0.32, 0.39, 0.31, 0.30]
    para_ldas = [1.0, 0.8, 1.2, 0.6, 1.4, 0.4, 1.6, 0.2, 1.8, 2.0]
    seed_np = [453984654, 61091275, 135739564, 57753681, 1590583398]
    seed_tf = [1317197089, 1324626939, 1518482692, 825093942, 1080703132]

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    std_y_train = tf.placeholder(tf.float32, shape=[])
    tau = tf.placeholder(tf.float32, shape=[])
    lda = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles,
                              tau, lda)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N, \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N

    variational = mean_field_variational(x, y_obs, n_x, layer_sizes, n_particles)
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
        #qsample, qlog_prob = matrixiaf(name, qsample, None, qlog_prob,
        #                               made, 1, update='gru')
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
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles,
                               tau, lda)
    y_pred = tf.reduce_mean(y_mean, 0)
    y_var = tf.sqrt(tf.reduce_mean(y_mean * y_mean, 0) - \
                    tf.reduce_mean(y_mean, 0) * tf.reduce_mean(y_mean, 0)) \
            * std_y_train
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
                     tf.log(std_y_train)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    for para_tau in para_taus:
        for para_lda in para_ldas:
            print('Starting evaluate hyperparameters (Tau={:.1f}, Lda={:.1f})'
                  .format(para_tau, para_lda))
            results = []
            for t in range(len(seed_np)):
                report = 0.0
                # train
                tf.set_random_seed(seed_tf[t])
                np.random.seed(seed_np[t])

                # Load UCI Boston housing data
                data_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'data', 'housing.data')
                x_train, y_train, x_valid, y_valid, x_test, y_test = \
                    dataset.load_uci_boston_housing(data_path)
                x_train = np.vstack([x_train, x_valid]).astype('float32')
                y_train = np.hstack([y_train, y_valid]).astype('float32')
                x_test = x_test.astype('float32')
                N, n_x = x_train.shape

                # Standardize data
                x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
                y_train, y_test, _, sd_y_train = dataset.standardize(
                    y_train.reshape((-1, 1)), y_test.reshape((-1, 1)))
                y_train, y_test = y_train.squeeze(), y_test.squeeze()
                sd_y_train = sd_y_train.squeeze()
                print(y_train[1])

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for epoch in range(1, epoches + 1):
                        time_epoch = -time.time()
                        if epoch % anneal_lr_freq == 0:
                            learning_rate *= anneal_lr_rate
                        lbs = []
                        res = []
                        errs = []
                        for t in range(iters):
                            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                            y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                            _, lb, re, error = sess.run(
                                [infer, lower_bound, rmse, reconstruction],
                                feed_dict={n_particles: lb_samples,
                                           learning_rate_ph: learning_rate,
                                           x: x_batch, y: y_batch,
                                           std_y_train: sd_y_train,
                                           tau: para_tau,
                                           lda: para_lda})
                            lbs.append(lb)
                            res.append(re)
                            errs.append(error)
                        time_epoch += time.time()
                        print('Epoch {} ({:.1f}s): Lower bound = {}, RMSE = {}, '
                              'RC ERROR = {}, KL = {}'.format(epoch,
                                                     time_epoch,
                                                     np.mean(lbs),
                                                     np.mean(res),
                                                     np.mean(errs),
                                                     np.mean(errs) - np.mean(lbs)))
                        if epoch % test_freq == 0:
                            time_test = -time.time()
                            test_lb, test_rmse, test_ll = sess.run(
                                [lower_bound, rmse, log_likelihood],
                                feed_dict={n_particles: lb_samples,
                                           learning_rate_ph: learning_rate,
                                           x: x_test, y: y_test,
                                           std_y_train: sd_y_train,
                                           tau: para_tau,
                                           lda: para_lda})
                            time_test += time.time()
                            print('>>> TEST ({:.1f}s)'.format(time_test))
                            print('>> Test lower bound = {}'.format(test_lb))
                            print('>> Test rmse = {}'.format(test_rmse))
                            print('>> Test log_likelihood = {}'.format(test_ll))
                            report = test_rmse
                print('Test #{} (Tau={:.1f}, Lda={:.1f}): rmse {}'.
                      format(t, para_tau, para_lda, report))

                results.append(report)
            print('(Tau={:.1f}, Lda={:.1f}): mean rmse {} +/- {}'.
                      format(para_tau, para_lda, mean(results), sd(results)))

