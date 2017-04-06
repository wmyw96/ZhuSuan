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


tau_alpha0 = 1000.0
tau_beta0 = 10.0
#tau_alpha0 = tf.abs(tf.get_variable('tau_alpha0', shape=[],
#                                    initializer=tf.constant_initializer(100.)))
#tau_beta0 = tf.abs(tf.get_variable('tau_beta0', shape=[],
#                                    initializer=tf.constant_initializer(1.)))
lda_alpha0 = tf.abs(tf.get_variable('lda_alpha0', shape=[],
                                    initializer=tf.constant_initializer(6.)))
lda_beta0 = tf.abs(tf.get_variable('lda_beta0', shape=[],
                                    initializer=tf.constant_initializer(6.)))

@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        tau_alpha = tf.expand_dims(tau_alpha0, 0)
        tau_beta = tf.expand_dims(tau_beta0, 0)

        tau = zs.Gamma('tau', alpha=tau_alpha, beta=tau_beta,
                       n_samples=n_particles, group_event_ndims=0)
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_particles, 1, n_out, n_in + 1])
            w_logstd = tf.ones([n_particles, 1,
                                n_out, n_in + 1]) * tf.log(3.0)
            # [n_particles, 1, n_out, n_in + 1]
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                n_samples=None, group_event_ndims=2))
        # forward
        # ?[(0), (1), (2), (3)]
        # [n_particles, N, n_x, 1]
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            # [n_particles, N, n_out, n_in + 1]
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y = zs.Normal('y', y_mean, tf.log(1.0 / tau))

    return model, y_mean, tf.reduce_mean((1.0 / tau))


def mean_field_variational(data_size, x, y_obs, n_x, layer_sizes, n_particles):
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

        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            # [n_particles, N, n_out, n_in + 1]
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        tau_alpha = tau_alpha0 + tf.to_float(data_size) * 0.5
        tau_alpha = tf.expand_dims(tau_alpha, 0)
        tau_alpha = tf.tile(tf.expand_dims(tau_alpha, 0), [n_particles, 1])
        tau_beta = tau_beta0 + tf.reduce_mean((y_mean - y_obs) ** 2, -1) * 0.5 * \
            data_size

        tau_beta = tf.expand_dims(tau_beta, 1)
        tau = zs.Gamma('tau', alpha=tau_alpha, beta=tau_beta,
                       n_samples=None, group_event_ndims=0)

    return variational, tf.reduce_mean(tau)


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

    data_size = 20
    # Load UCI Boston housing data
    # Load toy data set
    x_train, y_train = toy_data(data_size)
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
    lb_samples = 20
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
    #n_particles = 5
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_tau = model.local_log_prob(['tau'])
        log_pws = model.local_log_prob(w_names)
        print(tf.add_n(log_pws).get_shape())
        log_py_xw = model.local_log_prob('y')
        return log_tau + tf.add_n(log_pws) + \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N, \
               tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N

    variational, v_tau = mean_field_variational(data_size, x, y_obs, n_x, layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    tau_samples, tau_log_probs = variational.query('tau', outputs=True,
                                                   local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))
    latent.update({'tau': [tau_samples, tau_log_probs]})

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
    model, y_mean, para_std = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
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
            pcns = []
            #vas = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb, re, error, pcn = sess.run(
                    [infer, lower_bound, rmse, reconstruction, v_tau],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
                res.append(re)
                errs.append(error)
                pcns.append(pcn)
                #vas.append(va)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}, RMSE = {}, '
                  'RC ERROR = {}, KL = {}, Precision = {}'.format(epoch,
                                         time_epoch,
                                         np.mean(lbs),
                                         np.mean(res),
                                         np.mean(errs),
                                         np.mean(errs) - np.mean(lbs),
                                         np.mean(pcns)))
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
