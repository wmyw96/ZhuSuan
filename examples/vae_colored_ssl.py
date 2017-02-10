#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vae_ladder ssl for CIFAR10. Without autoregressive connections. Separate p(x|z)
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


@zs.reuse('model')
def M2(observed, n, n_y, groups, n_particles):
    with zs.StochasticGraph(observed=observed) as model:
        y_logits = tf.zeros([n_particles, n_y])
        y = zs.Discrete('y', y_logits, sample_dim=1, n_samples=n)
        y = tf.reshape(y, [-1, 1, 1, n_y])
        h_top = tf.get_variable(name='h_top_p',
                                shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
        h = tf.tile(h_top,
                    [n * n_particles, groups[-1].map_size,
                     groups[-1].map_size, 1])
        y_ = tf.tile(y, [1, groups[-1].map_size, groups[-1].map_size, 1])
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

                h1, pz_mean, pz_logstd = tf.split(
                    h1, [group.num_filters] + [group.n_z] * 2, axis=3
                )
                z = zs.Normal(name, pz_mean, pz_logstd)
                z = tf.reshape(z, [-1, group.map_size, group.map_size,
                                   group.n_z])
                if group_i == len(group)-1 and block_i == group.num_blocks-1:
                    y_ = tf.tile(y, [1, group.map_size, group.map_size, 1])
                    h1 = tf.concat([h1, z, y_], 3)
                else:
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


@zs.reuse('variational')
def qz_xy(x, y, n_y, n_xl, groups, n_particles):
    with zs.StochasticGraph() as variational:
        x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        y = tf.reshape(y, [-1, 1, 1, n_y])
        y_ = tf.tile(y, [1, n_xl, n_xl, 1])
        h = layers.conv2d(tf.concat([x, y_], 3), groups[0].num_filters, 5, 2,
                          activation_fn=None)
        qz_mean = {}
        qz_logstd = {}
        posterior = {}

        for group_i, group in enumerate(groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)
                stride = 1
                if group_i > 0 and block_i == 0:
                    stride = 2
                # y_ = tf.tile(y, [1, tf.shape(h)[1], tf.shape(h)[1], 1])
                # h1 = tf.concat([h, y_], 3)
                h1 = h
                h1 = tf.nn.elu(h1)
                h1 = layers.conv2d(h1, group.num_filters + 2 * group.n_z, 3,
                                   stride=stride, activation_fn=None,
                                   scope=name + '_up_conv1')
                qz_mean[name], qz_logstd[name], h1 = tf.split(
                    h1, [group.n_z] * 2 + [group.num_filters], axis=3)
                h1 = tf.nn.elu(h1)
                h1 = layers.conv2d(h1, group.num_filters, kernel_size=3,
                                   activation_fn=None, scope=name + '_up_conv2')
                if stride == 2:
                    h = layers.conv2d(h, group.num_filters, 3, stride=2,
                                      activation_fn=None,
                                      scope=name + '_resize_up')
                h += 0.1 * h1

        h_top = tf.get_variable(name='h_top_q',
                                shape=[1, 1, 1, groups[-1].num_filters],
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
                rz_mean, rz_logstd, h1 = tf.split(
                    h1, [group.n_z] * 2 + [group.num_filters], axis=3
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
                # y_ = tf.tile(y, [1, group.map_size, group.map_size, 1])
                # h1 = tf.concat([h1, z, y_], 3)
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


@zs.reuse('classifier')
def qy_x(x, n_xl, n_y):
    ly_x = tf.reshape(x, [-1, n_xl, n_xl, 3])
    ly_x = layers.conv2d(ly_x, 32, 3)
    ly_x = layers.conv2d(ly_x, 32, 3)
    ly_x = layers.max_pool2d(ly_x, 2)
    ly_x = layers.conv2d(ly_x, 64, 3)
    ly_x = layers.conv2d(ly_x, 64, 3)
    ly_x = layers.max_pool2d(ly_x, 2)
    ly_x = layers.conv2d(ly_x, 128, 3)
    ly_x = layers.conv2d(ly_x, 128, 3)
    ly_x = layers.max_pool2d(ly_x, 2)
    ly_x = layers.flatten(ly_x)
    ly_x = layers.fully_connected(ly_x, 500)
    ly_x = layers.fully_connected(ly_x, n_y, activation_fn=None)
    return ly_x


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_labeled, t_labeled, x_unlabeled, t_unlabeled, x_test, t_test = \
        dataset.load_cifar10_semi_supervised(data_path, normalize=True,
                                             one_hot=True)

    _, n_xl, _, n_channels = x_labeled.shape
    n_x = n_xl * n_xl * n_channels
    x_labeled, x_unlabeled, x_test = map(lambda xs: xs.reshape(
        (-1, n_x)), [x_labeled, x_unlabeled, x_test])
    n_labeled = x_labeled.shape[0]
    n_y = t_labeled.shape[1]

    # Define training/evaluation parameters
    lb_samples = 1
    beta = 100.
    epoches = 3000
    batch_size = 20
    test_batch_size = 20
    iters = x_unlabeled.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size

    print_freq = 100
    test_freq = 500
    learning_rate = 0.0003
    anneal_lr_freq = 50
    anneal_lr_rate = 0.75
    bottle_neck_group = namedtuple(
        'bottle_neck_group',
        ['num_blocks', 'num_filters', 'map_size', 'n_z'])
    groups = [
        bottle_neck_group(2, 64, 16, 32),
        bottle_neck_group(2, 64, 8, 32),
        bottle_neck_group(2, 64, 4, 32)
    ]

    # Build the computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)

    def log_joint(observed):
        x = observed['x']
        n = tf.shape(x)[1]
        model, x_mean, x_logstd = M2(observed, n, n_y, groups, n_particles)
        z_names = list(six.iterkeys(observed))
        z_names.remove('x')
        z_names.remove('y')
        log_pzs = model.local_log_prob(z_names)
        log_pz = sum(tf.reduce_sum(
            tf.reshape(z, [n_particles, n, -1]), -1) for z in log_pzs)
        log_px_zy = tf.log(logistic.cdf(x + 1./256., x_mean, x_logstd) -
                           logistic.cdf(x, x_mean, x_logstd) + 1e-8)
        log_px_zy = tf.reduce_sum(log_px_zy, -1)
        log_py = model.local_log_prob('y')
        return log_px_zy + log_pz + log_py

    # Labeled
    x_labeled_ph = tf.placeholder(tf.float32, shape=[None, n_x], name='x_l')
    x_labeled_obs = tf.tile(tf.expand_dims(x_labeled_ph, 0),
                            [n_particles, 1, 1])
    y_labeled_ph = tf.placeholder(tf.float32, shape=[None, n_y], name='y_l')
    y_labeled_obs = tf.tile(tf.expand_dims(y_labeled_ph, 0),
                            [n_particles, 1, 1])
    n = tf.shape(x_labeled_ph)[0]
    variational, posterior = qz_xy(x_labeled_ph, y_labeled_ph, n_y, n_xl,
                                   groups, n_particles)
    qzs_samples_logs = variational.query(list(six.iterkeys(posterior)),
                                         outputs=True, local_log_prob=True)
    zs_ = [[z_output[0],
            tf.reduce_sum(tf.reshape(z_output[1], [n_particles, n, -1]), -1)]
           for z_output in qzs_samples_logs]
    latents = dict(zip(list(six.iterkeys(posterior)), zs_))
    labeled_lower_bound = tf.reduce_mean(
        zs.advi(log_joint, {'x': x_labeled_obs, 'y': y_labeled_obs}, latents,
                axis=0))
    labeled_bits_per_dim = -labeled_lower_bound / n_x * 1. / np.log(2.)

    # Unlabeled
    x_unlabeled_ph = tf.placeholder(tf.float32, shape=[None, n_x], name='x_u')
    n = tf.shape(x_unlabeled_ph)[0]
    y_diag = tf.diag(tf.ones(n_y))
    y_u = tf.reshape(tf.tile(tf.expand_dims(y_diag, 0), [n, 1, 1]), [-1, n_y])
    x_u = tf.reshape(tf.tile(tf.expand_dims(x_unlabeled_ph, 1), [1, n_y, 1]),
                     [-1, n_x])
    x_unlabeled_obs = tf.tile(tf.expand_dims(x_u, 0), [n_particles, 1, 1])
    y_unlabeled_obs = tf.tile(tf.expand_dims(y_u, 0), [n_particles, 1, 1])
    variational, posterior = qz_xy(x_u, y_u, n_y, n_xl, groups, n_particles)

    qzs_samples_logs = variational.query(list(six.iterkeys(posterior)),
                                         outputs=True, local_log_prob=True)
    zs_ = [[z_output[0],
            tf.reduce_sum(tf.reshape(z_output[1],
                                     [n_particles, tf.shape(x_u)[0], -1]), -1)]
           for z_output in qzs_samples_logs]
    latents = dict(zip(list(six.iterkeys(posterior)), zs_))
    lb_z = zs.advi(log_joint, {'x': x_unlabeled_obs, 'y': y_unlabeled_obs},
                   latents, axis=0)

    # sum over y
    lb_z = tf.reshape(lb_z, [-1, n_y])
    qy_logits_u = qy_x(x_unlabeled_ph, n_xl, n_y)
    qy_u = tf.reshape(tf.nn.softmax(qy_logits_u), [-1, n_y])
    qy_u += 1e-8
    qy_u /= tf.reduce_sum(qy_u, 1, keep_dims=True)
    log_qy_u = tf.log(qy_u)
    unlabeled_lower_bound = tf.reduce_mean(
        tf.reduce_sum(qy_u * (lb_z - log_qy_u), 1))
    unlabeled_bits_per_dim = - unlabeled_lower_bound / n_x * 1. / np.log(2.)

    # Build classifier
    qy_logits_l = qy_x(x_labeled_ph, n_xl, n_y)
    qy_l = tf.nn.softmax(qy_logits_l)
    pred_y = tf.argmax(qy_l, 1)
    acc = tf.reduce_mean(
        tf.cast(tf.equal(pred_y, tf.argmax(y_labeled_ph, 1)), tf.float32))
    log_qy_x = zs.discrete.logpmf(y_labeled_ph, qy_logits_l)
    log_qy_x = tf.reduce_mean(log_qy_x)
    classifier_cost = -beta * log_qy_x

    # Gather gradients
    cost = -(labeled_lower_bound + unlabeled_lower_bound - classifier_cost) / 2.
    grads = optimizer.compute_gradients(cost)
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
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            indices = np.random.permutation(x_unlabeled.shape[0])
            x_unlabeled = x_unlabeled[indices]
            t_unlabeled = t_unlabeled[indices]
            lbs_labeled, lbs_unlabeled, train_accs, train_costs = [], [], [], []
            unlabeled_accs, unlabeled_costs = [], []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                labeled_indices = np.random.randint(0, n_labeled,
                                                    size=batch_size)
                x_labeled_batch = x_labeled[labeled_indices]
                y_labeled_batch = t_labeled[labeled_indices]
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                y_unlabeled_batch = t_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                _, lb_labeled, lb_unlabeled, cost, train_acc = sess.run(
                    [infer, labeled_bits_per_dim, unlabeled_bits_per_dim,
                     log_qy_x, acc],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch,
                               x_unlabeled_ph: x_unlabeled_batch,
                               learning_rate_ph: learning_rate,
                               n_particles: lb_samples})
                unlabeled_acc, unlabeled_cost = sess.run([acc, log_qy_x],
                                                         feed_dict={
                                             x_labeled_ph: x_unlabeled_batch,
                                             y_labeled_ph: y_unlabeled_batch,
                                         })
                lbs_labeled.append(lb_labeled)
                lbs_unlabeled.append(lb_unlabeled)
                train_accs.append(train_acc)
                train_costs.append(cost)
                unlabeled_accs.append(unlabeled_acc)
                unlabeled_costs.append(unlabeled_cost)
                if iter % print_freq == 0:
                    print('Ep{} Iter{:04d}({:.2f}s): '
                          'LB_l: {:.2f}, LB_u: {:.2f} '
                          'log q(y|x): {:.2f} Acc: {:.2f}% '
                          'log q(y|x)_u: {:.2f} Acc_u: {:.2f}%'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(lbs_labeled), np.mean(lbs_unlabeled),
                                 np.mean(train_costs),
                                 np.mean(train_accs) * 100.,
                                 np.mean(unlabeled_costs),
                                 np.mean(unlabeled_accs) * 100.))
                    lbs_labeled, lbs_unlabeled, train_accs, train_costs = \
                        [], [], [], []
                    unlabeled_accs = []
                    time_train = -time.time()
                if iter % test_freq == 0:
                    time_test = -time.time()
                    test_lls_labeled = []
                    test_lls_unlabeled = []
                    test_accs = []
                    test_costs = []
                    for k in range(test_iters):
                        test_x_batch = \
                            x_test[
                            k * test_batch_size: test_batch_size * (k + 1)]
                        test_y_batch = \
                            t_test[
                            k * test_batch_size: (k + 1) * test_batch_size]
                        test_ll_labeled, test_ll_unlabeled, test_cost, \
                        test_acc = sess.run(
                            [labeled_bits_per_dim, unlabeled_bits_per_dim,
                             log_qy_x, acc],
                            feed_dict={x_labeled_ph: test_x_batch,
                                       y_labeled_ph: test_y_batch,
                                       x_unlabeled_ph: test_x_batch,
                                       n_particles: lb_samples})
                        test_lls_labeled.append(test_ll_labeled)
                        test_lls_unlabeled.append(test_ll_unlabeled)
                        test_accs.append(test_acc)
                        test_costs.append(test_cost)
                    time_test += time.time()
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test LB: labeled = {:.3f}, unlabeled = {:.3f} '
                          'Log q(y|x) = {:.3f}'.
                          format(np.mean(test_lls_labeled),
                                 np.mean(test_lls_unlabeled),
                                 np.mean(test_costs)))
                    print('>> Test accuracy: {:.2f}%'.format(
                        100. * np.mean(test_accs)))

                    time_un = -time.time()
                    un_lls_labeled = []
                    un_lls_unlabeled = []
                    un_accs = []
                    un_costs = []
                    for k in range(iters):
                        un_x_batch = \
                            x_unlabeled[
                            k * test_batch_size: test_batch_size * (k + 1)]
                        un_y_batch = \
                            t_unlabeled[
                            k * test_batch_size: (k + 1) * test_batch_size]
                        un_ll_labeled, un_ll_unlabeled, un_cost, \
                        un_acc = sess.run(
                            [labeled_bits_per_dim, unlabeled_bits_per_dim,
                             log_qy_x, acc],
                            feed_dict={x_labeled_ph: un_x_batch,
                                       y_labeled_ph: un_y_batch,
                                       x_unlabeled_ph: un_x_batch,
                                       n_particles: lb_samples})
                        un_lls_labeled.append(un_ll_labeled)
                        un_lls_unlabeled.append(un_ll_unlabeled)
                        un_accs.append(un_acc)
                        un_costs.append(un_cost)
                    time_un += time.time()
                    print('>>> TRAIN ({:.1f}s)'.format(time_un))
                    print('>> TRAIN LB: labeled = {:.3f}, unlabeled = {:.3f} '
                          'Log q(y|x) = {:.3f}'.
                          format(np.mean(un_lls_labeled),
                                 np.mean(un_lls_unlabeled),
                                 np.mean(un_costs)))
                    print('>> TRAIN accuracy: {:.2f}%'.format(
                        100. * np.mean(un_accs)))

