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

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

import utils


@zs.reuse('model')
def blr(observed, N, D, x, n_particles):
    with zs.StochasticGraph(observed=observed) as model:
        # w: [D, n_particles]
        w = zs.Normal('w', tf.zeros([D]), tf.ones([D]),
                      sample_dim=1, n_samples=n_particles)
        pred = tf.matmul(x, w)
        # y: [N, n_particles]
        y = zs.Bernoulli('y', pred)
    return model


@zs.reuse('variational')
def q_net(observed, var_mean, var_logstd):
    with zs.StochasticGraph(observed=observed) as variational:
        w = zs.Normal('w', var_mean, var_logstd,
                      sample_dim=1, n_samples=n_particles)
    return variational


if __name__ == "__main__":
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Define model parameters
    N = 5
    D = 2
    learning_rate = 1
    learning_rate_g = 0.001
    learning_rate_d = 0.01
    t0 = 100
    t0_d = 100
    t0_g = 10
    epoches = 1000
    epoches_d = 100
    epoches_d0 = 1000
    epoches_g = 1
    lower_box = -5
    upper_box = 5


    def draw_decision_boundary(x, w, y):
        positive_x = x[y == 1, :]
        negative_x = x[y == 0, :]

        x0_points = np.linspace(lower_box, upper_box, num=100)
        x1_points = np.linspace(lower_box, upper_box, num=100)
        grid_x, grid_y = np.meshgrid(x0_points, x1_points)
        points = np.hstack((np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])))
        pred = np.sum(points * w, axis=1)
        pred = 1.0 / (1 + np.exp(-pred))
        grid_pred = np.reshape(pred, grid_x.shape)

        plt.pcolormesh(grid_x, grid_y, grid_pred)
        plt.colorbar()
        CS = plt.contour(grid_x, grid_y, grid_pred, colors='k', levels=np.array([0.25, 0.5, 0.75]))
        plt.clabel(CS)
        plt.plot(positive_x[:, 0], positive_x[:, 1], 'x')
        plt.plot(negative_x[:, 0], negative_x[:, 1], 'o')


    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[N, D], name='x')
    y = tf.placeholder(tf.float32, shape=[N], name='y')
    w_ph = tf.placeholder(tf.float32, shape=[D, None], name='w_ph')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    y_rep = tf.tile(tf.expand_dims(y, axis=1), [1, n_particles])
    var_mean = tf.Variable(tf.zeros([D]), name='var_mean')
    var_logstd = tf.Variable(tf.zeros([D]), name='var_logstd')

    # Variational inference
    def log_joint(observed):
        model = blr(observed, N, D, x, n_particles)
        # log_pw: [D, n_particles]; log_py_w: [N, n_particles]
        log_pw, log_py_w = model.local_log_prob(['w', 'y'])
        return tf.reduce_sum(log_pw, 0) + tf.reduce_sum(log_py_w, 0), \
               tf.reduce_sum(log_pw, 0), tf.reduce_sum(log_py_w, 0)

    variational = q_net({}, var_mean, var_logstd)
    # [D, n_particles]
    qw_samples, log_qw = variational.query('w', outputs=True,
                                           local_log_prob=True)
    log_qw = tf.reduce_sum(log_qw, 0)
    lower_bound = tf.reduce_mean(log_joint({'w': qw_samples, 'y': y_rep})[0] - log_qw)

    lj, lprior, _ = log_joint({'w': w_ph, 'y': y_rep})
    lp = q_net({'w': w_ph}, var_mean, var_logstd).local_log_prob('w')
    lp = tf.reduce_sum(lp, 0)

    # Data generator
    model = blr({}, N, D, x, n_particles)
    w_samples, y_samples = model.query(['w', 'y'], outputs=True,
                                       local_log_prob=True)
    w_samples, w_prior = w_samples
    y_samples = y_samples[0]

    # Adverserial training
    # Generator
    with tf.name_scope('generator'):
        epsilon = tf.random_normal((n_particles, D))
        h = layers.fully_connected(epsilon, 10, scope="generator1")
        h = layers.fully_connected(h, 10, scope="generator2")
        generated_w = layers.fully_connected(h, 2, activation_fn=None,
                                             scope="generator3")
        generated_w = tf.transpose(generated_w)

    # Generator: using tractable q
    generated_w = qw_samples
    log_density_ratio = lp - lprior
    true_d = 1.0 / (1.0 + tf.exp(-log_density_ratio))

    # Discriminator
    def discriminator(w):
        with tf.name_scope('discriminator'):
            # [n_particles, D]
            w = tf.transpose(w)
            h = layers.fully_connected(w, 50, scope="disc1", activation_fn=tf.nn.relu)
            h = layers.fully_connected(h, 50, scope="disc2", activation_fn=tf.nn.relu)
            h = layers.fully_connected(h, 50, scope="disc3", activation_fn=tf.nn.relu)
            # [n_particles]
            d = tf.squeeze(layers.fully_connected(h, 1, scope="disc4", activation_fn=None))

        return d

    # Objective
    d_qw = discriminator(generated_w)
    d_pw = discriminator(w_samples)
    # eq_d = -tf.nn.sigmoid_cross_entropy_with_logits(logits=d_qw,
    #                                                 labels=tf.ones_like(d_qw))
    # eq_1_d = -tf.nn.sigmoid_cross_entropy_with_logits(logits=d_qw,
    #                                         labels=tf.zeros_like(d_qw))
    # ep_1_d = -tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pw,
    #                                             labels=tf.zeros_like(d_pw))
    eq_d = d_qw - tf.log(1 + tf.exp(d_qw))
    eq_1_d = -tf.log(1 + tf.exp(d_qw))
    ep_1_d = -tf.log(1 + tf.exp(d_pw))
    eq_ll = log_joint({'w': generated_w, 'y': y_rep})[2]
    disc_obj = tf.reduce_mean(eq_d + ep_1_d)
    gen_obj = tf.reduce_mean(eq_d - eq_1_d - eq_ll)

    estimated_d = tf.nn.sigmoid(discriminator(w_ph))
    estimated_q = lprior + discriminator(w_ph)

    eq_log_q = log_qw
    eq_log_p = log_joint({'w': generated_w})[1]
    ep_log_q = tf.reduce_sum(q_net({'w': w_samples}, var_mean, var_logstd).local_log_prob('w'), 0)
    ep_log_p = w_prior
    optimal_disc_obj = tf.reduce_mean(-tf.log(1 + tf.exp(-eq_log_q + eq_log_p))
                                      -tf.log(1 + tf.exp(ep_log_q - ep_log_p)))

    # Optimizer
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(-lower_bound, var_list=[var_mean, var_logstd])

    d_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="disc")
    g_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="generator")
    g_parameters = [var_mean, var_logstd]
    print('D parameters')
    for i in d_parameters:
        print(i.name, i.get_shape())
    print('G parameters')
    for i in g_parameters:
        print(i.name, i.get_shape())

    d_optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    d_infer = optimizer.minimize(-disc_obj, var_list=d_parameters)
    g_optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    g_infer = optimizer.minimize(gen_obj, var_list=g_parameters)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Generate data
        # Generate x
        nx = np.random.rand(N, D) * (upper_box - lower_box) + lower_box

        # Generate w, y
        nw, ny = sess.run([w_samples, y_samples], feed_dict={x: nx, n_particles: 1})
        nw = np.squeeze(nw)
        ny = np.squeeze(ny)

        # Draw the decision boundary
        plt.subplot(3, 3, 1)
        draw_decision_boundary(nx, nw, ny)
        print('W = {}'.format(nw))

        # Generate a w grid
        w0 = np.linspace(-10, 10, 100)
        w1 = np.linspace(-10, 10, 100)
        w0_grid, w1_grid = np.meshgrid(w0, w1)
        w_points = np.hstack((np.reshape(w0_grid, [-1, 1]), np.reshape(w1_grid, [-1, 1])))
        # Evaluate log_joint
        lj_points = np.transpose(sess.run(lj, feed_dict={x: nx,
                                            y: ny,
                                            n_particles: w_points.shape[0],
                                            w_ph: np.transpose(w_points)}))
        lj_grid = np.reshape(lj_points, w0_grid.shape)

        plt.subplot(3, 3, 2)
        plt.pcolormesh(w0_grid, w1_grid, lj_grid)
        plt.colorbar()
        CS = plt.contour(w0_grid, w1_grid, lj_grid, colors='k')
        plt.clabel(CS)

        # Run the inference
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            _, lb = sess.run([infer, lower_bound],
                             feed_dict={x: nx,
                                        y: ny,
                                        learning_rate_ph: learning_rate * t0 / (t0 + epoch),
                                        n_particles: 100})
            time_epoch += time.time()
            if epoch % 100 == 0:
                print(sess.run([var_mean, var_logstd]))
                print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                    epoch, time_epoch, lb))

        # Plot the prior
        # Evaluate log_joint
        lp_points = np.transpose(sess.run(lp, feed_dict={x: nx,
                                                         y: ny,
                                                         n_particles: w_points.shape[0],
                                                         w_ph: np.transpose(w_points)}))
        lp_grid = np.reshape(lp_points, w0_grid.shape)

        plt.subplot(3, 3, 4)
        plt.pcolormesh(w0_grid, w1_grid, lp_grid)
        plt.colorbar()
        CS = plt.contour(w0_grid, w1_grid, lp_grid, colors='k')
        plt.clabel(CS)

        # Run the adverserial inference
        for i in range(epoches_d0):
            _, do, dp, dq, sp, sq, eqd, ep1d = sess.run([d_infer, disc_obj, d_pw,
                                                         d_qw, w_samples, generated_w, eq_d, ep_1_d],
                                                        feed_dict={x: nx,
                                                                   y: ny,
                                                                   learning_rate_ph: learning_rate_d * t0_d / (
                                                                   t0_d + i),
                                                                   n_particles: 10000})
            if i % 100 == 0:
                print('Discriminator obj = {}'.format(do))

        for epoch in range(1, epoches_g + 1):
            # lr = learning_rate * t0 / (t0 + epoch)
            objs = []
            for i in range(epoches_d):
                _, do, dp, dq, sp, sq, eqd, ep1d = sess.run([d_infer, disc_obj, d_pw,
                                                  d_qw, w_samples, generated_w, eq_d, ep_1_d],
                                 feed_dict={x: nx,
                                            y: ny,
                                            learning_rate_ph: learning_rate_d * t0_d / (t0_d + i),
                                            n_particles: 10000})
                objs.append(do)
            print('Discriminator obj = {}, {}'.format(objs[0], objs[-1]))

            # sess.run(g_infer,
            #          feed_dict={x: nx,
            #                     y: ny,
            #                     learning_rate_ph: lr,
            #                     n_particles: 100})

            print('Epoch {}: Discriminator obj = {}'.format(epoch, do))

        # Generate samples from the generator
        # samples = sess.run(generated_w,
        #                    feed_dict={n_particles: 10000})
        # plt.subplot(2, 2, 4)
        # plt.plot(samples[0,:], samples[1,:], '.')

        print('Optimal discriminator obj = {}'.format(
            sess.run(optimal_disc_obj, feed_dict={n_particles: 10000})))

        plt.subplot(3, 3, 3)
        plt.plot(objs, '.')

        lp_points = np.transpose(sess.run(estimated_q, feed_dict={x: nx,
                                                         y: ny,
                                                         n_particles: w_points.shape[0],
                                                         w_ph: np.transpose(w_points)}))
        lp_grid = np.reshape(lp_points, w0_grid.shape)

        plt.subplot(3, 3, 5)
        plt.pcolormesh(w0_grid, w1_grid, lp_grid)
        plt.colorbar()
        CS = plt.contour(w0_grid, w1_grid, lp_grid, colors='k')
        plt.clabel(CS)


        lp_points = np.transpose(sess.run(true_d, feed_dict={n_particles: w_points.shape[0],
                                                             w_ph: np.transpose(w_points)}))
        lp_grid = np.reshape(lp_points, w0_grid.shape)

        plt.subplot(3, 3, 7)
        plt.pcolormesh(w0_grid, w1_grid, lp_grid)
        plt.colorbar()
        CS = plt.contour(w0_grid, w1_grid, lp_grid, colors='k')
        plt.clabel(CS)

        lp_points = np.transpose(sess.run(estimated_d, feed_dict={n_particles: w_points.shape[0],
                                                             w_ph: np.transpose(w_points)}))
        lp_grid = np.reshape(lp_points, w0_grid.shape)

        plt.subplot(3, 3, 8)
        plt.pcolormesh(w0_grid, w1_grid, lp_grid)
        plt.colorbar()
        CS = plt.contour(w0_grid, w1_grid, lp_grid, colors='k')
        plt.clabel(CS)

        plt.show()