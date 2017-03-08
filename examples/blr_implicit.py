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
#import seaborn as sns

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors

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


def kde(samples, points, kernel_stdev):
    # samples: [n D]
    # points: [m D]
    # return [m]
    samples = tf.expand_dims(samples, 1)
    points = tf.expand_dims(points, 0)
    # [n m D]
    Z = np.sqrt(2 * np.pi) * kernel_stdev
    log_probs = -np.log(Z) + (-0.5 / kernel_stdev**2) * (samples - points)**2
    log_probs = tf.reduce_sum(log_probs, -1)
    log_probs = zs.log_mean_exp(log_probs, 0)

    return log_probs


def compute_tickers(probs, n_bins=10):
    # Sort
    flat_probs = list(probs.flatten())
    flat_probs.sort()
    flat_probs.reverse()

    num_intervals = n_bins * (n_bins - 1) // 2
    interval_size = len(flat_probs) // num_intervals

    tickers = []
    cnt = 0
    for i in range(n_bins - 1):
        tickers.append(flat_probs[cnt])
        cnt += interval_size * (i + 1)
    tickers.append(flat_probs[-1])
    tickers.reverse()
    return tickers


def contourf(x, y, z):
    tickers = compute_tickers(z)
    palette = cm.PuBu
    plt.contourf(x, y, z, tickers,
                 cm=palette, norm=colors.BoundaryNorm(tickers, ncolors=palette.N))
    plt.colorbar()


if __name__ == "__main__":
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # samples = tf.constant(np.array([[2, 2], [-2, -2]]), dtype=tf.float32)
    # points = tf.placeholder(tf.float32, [10000, 2])
    # probs = kde(samples, points, 0.1)
    #
    # # Generate a w grid
    # w0 = np.linspace(-10, 10, 100)
    # w1 = np.linspace(-10, 10, 100)
    # w0_grid, w1_grid = np.meshgrid(w0, w1)
    # w_points = np.hstack(
    #     (np.reshape(w0_grid, [-1, 1]), np.reshape(w1_grid, [-1, 1])))
    #
    # sess = tf.Session()
    # vals = sess.run(probs, feed_dict={points: w_points})
    # vals = np.reshape(vals, w0_grid.shape)
    #
    # plt.subplot(111)
    # contourf(w0_grid, w1_grid, vals)
    # plt.show()
    #
    # sys.exit(0)

    # Define model parameters
    N = 5
    D = 2
    learning_rate = 1
    learning_rate_g = 0.01
    learning_rate_d = 0.01
    t0 = 100
    t0_d = 100
    t0_g = 10
    epoches = 100
    epoches_d = 20
    epoches_d0 = 1000
    epoches_g = 200
    discriminator_num_samples = 10000
    generator_num_samples = 100000
    lower_box = -5
    upper_box = 5
    kde_batch_size = 2000
    kde_num_samples = 100000
    kde_stdev = 0.2


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
    samples_ph = tf.placeholder(tf.float32, shape=[None, D], name='samples')
    points_ph = tf.placeholder(tf.float32, shape=[None, D], name='points')
    grid_prob_op = kde(samples_ph, points_ph, kde_stdev)

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
        h = layers.fully_connected(epsilon, 20, scope="generator1",
                                   weights_initializer=tf.contrib.layers.xavier_initializer())
        h = layers.fully_connected(h, 20, scope="generator2",
                                   weights_initializer=tf.contrib.layers.xavier_initializer())
        generated_w = layers.fully_connected(h, 2, activation_fn=None,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             scope="generator3")
        generated_w = tf.transpose(generated_w)

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
    sigmoid = tf.nn.sigmoid_cross_entropy_with_logits
    eq_d = -sigmoid(logits=d_qw, labels=tf.ones_like(d_qw))
    eq_1_d = -sigmoid(logits=d_qw, labels=tf.zeros_like(d_qw))
    ep_1_d = -sigmoid(logits=d_pw, labels=tf.zeros_like(d_pw))
    eq_ll = log_joint({'w': generated_w, 'y': y_rep})[2]
    disc_obj = tf.reduce_mean(eq_d + ep_1_d)
    prior_term = tf.reduce_mean(eq_d - eq_1_d)
    ll_term = tf.reduce_mean(-eq_ll)
    gen_obj = prior_term + ll_term

    estimated_d = tf.nn.sigmoid(discriminator(w_ph))
    estimated_q = lprior + discriminator(w_ph)

    # Optimizer
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(-lower_bound, var_list=[var_mean, var_logstd])

    d_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="disc")
    g_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="generator")
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

        # Run the adverserial inference
        for i in range(epoches_d0):
            _, do, dp, dq, sp, sq, eqd, ep1d = sess.run([d_infer, disc_obj, d_pw,
                                                         d_qw, w_samples, generated_w, eq_d, ep_1_d],
                                                        feed_dict={x: nx,
                                                                   y: ny,
                                                                   learning_rate_ph: learning_rate_d * t0_d / (
                                                                   t0_d + i),
                                                                   n_particles: discriminator_num_samples})
            if i % 100 == 0:
                print('Discriminator obj = {}'.format(do))

        gen_objs = []
        disc_cnt = 0
        for epoch in range(1, epoches_g + 1):
            lg = learning_rate_g * t0_g / (t0_g + epoch)
            objs = []
            disc_cnt += 1
            for i in range(epoches_d):
                _, do, dp, dq, sp, sq, eqd, ep1d = sess.run([d_infer, disc_obj, d_pw,
                                                  d_qw, w_samples, generated_w, eq_d, ep_1_d],
                                 feed_dict={x: nx,
                                            y: ny,
                                            learning_rate_ph: lg * 3 / (i + 3),
                                            n_particles: discriminator_num_samples})
                objs.append(do)
            print('Discriminator obj = {}, {}'.format(objs[0], objs[-1]))

            _, go, pv, lv = sess.run([g_infer, gen_obj, prior_term, ll_term],
                     feed_dict={x: nx,
                                y: ny,
                                learning_rate_ph: lg,
                                n_particles: generator_num_samples})
            gen_objs.append(go)
            print('Generator obj = {}, prior = {}, ll = {}'.format(go, pv, lv))

            if epoch % 200 == 0:
                # Draw the decision boundary
                plt.subplot(3, 3, 1)
                draw_decision_boundary(nx, nw, ny)
                plt.title('Decision boundary')

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
                contourf(w0_grid, w1_grid, lj_grid)
                plt.plot(nw[0], nw[1], 'x')
                plt.title('True posterior')

                # Plot the prior
                # Evaluate log_joint
                lp_points = np.transpose(sess.run(lp, feed_dict={x: nx, y: ny,
                        n_particles: w_points.shape[0],
                        w_ph: np.transpose(w_points)}))
                lp_grid = np.reshape(lp_points, w0_grid.shape)

                plt.subplot(3, 3, 3)
                plt.plot(gen_objs, '.')
                plt.title('Generator objective')

                plt.subplot(3, 3, 6)
                plt.plot(objs, '.')
                plt.title('Discriminator objective')

                plt.subplot(3, 3, 4)
                contourf(w0_grid, w1_grid, lp_grid)
                plt.title('Mean field posterior')

                # Generate samples from the generator
                samples = sess.run(generated_w,
                                   feed_dict={n_particles: kde_num_samples})
                ax = plt.subplot(3, 3, 7)
                ax.plot(samples[0,:], samples[1,:], '.')
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                plt.title('Implicit posterior samples')

                # Compute kde
                point_prob = np.zeros((w_points.shape[0]))
                kde_num_batches = kde_num_samples // kde_batch_size
                for b in range(kde_num_batches):
                    sample_batch = samples[:,\
                                   b*kde_batch_size:(b+1)*kde_batch_size]
                    point_prob += sess.run(grid_prob_op,
                            feed_dict={samples_ph: np.transpose(sample_batch),
                                       points_ph: w_points})
                point_prob /= kde_num_batches
                point_prob = np.reshape(point_prob, w0_grid.shape)

                plt.subplot(3, 3, 5)
                contourf(w0_grid, w1_grid, point_prob)
                plt.title('Implicit posterior KDE')

                # Plot estimated q
                lp_points = np.transpose(sess.run(estimated_q, feed_dict={x: nx,
                                                                 y: ny,
                                                                 n_particles: w_points.shape[0],
                                                                 w_ph: np.transpose(w_points)}))
                lp_grid = np.reshape(lp_points, w0_grid.shape)

                plt.subplot(3, 3, 8)
                contourf(w0_grid, w1_grid, lp_grid)
                plt.title('Estimated implicit posterior using discriminator')

                # Plot estimated D
                lp_points = np.transpose(sess.run(estimated_d, feed_dict={n_particles: w_points.shape[0],
                                                                     w_ph: np.transpose(w_points)}))
                lp_grid = np.reshape(lp_points, w0_grid.shape)

                plt.subplot(3, 3, 9)
                plt.pcolormesh(w0_grid, w1_grid, lp_grid)
                plt.colorbar()
                CS = plt.contour(w0_grid, w1_grid, lp_grid, colors='k')
                plt.clabel(CS)
                plt.title('Discriminator')

                plt.show()
