#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import prettytensor as pt
from six.moves import range, reduce, zip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli, logistic
    from zhusuan.layers import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    from deconv import deconv2d
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_x):
        self.n_x = n_x
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            self.l_z1_z2 = (pt.template('z2').
                            reshape([-1, 16, 16, 64]).
                            deconv2d(3, 64))

            self.l_z1_mean = (pt.template('z1_hid').
                              deconv2d(3, 64, activation_fn=None))

            self.l_z1_logvar = (pt.template('z1_hid').
                                deconv2d(3, 64, activation_fn=None))

            self.l_x_z1 = (pt.template('z1').
                           reshape([-1, 16, 16, 64]).
                           deconv2d(5, 64, stride=2).
                           deconv2d(3, 3, activation_fn=None))

            self.x_logsd = tf.get_variable("x_logsd",  (),
                                           initializer=tf.constant_initializer(
                                             0.0))

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        z2 = latent['z2']
        z1 = latent['z1']
        x = observed['x']

        log_pz2 = tf.reduce_sum(norm.logpdf(z2), [2, 3, 4])

        n_samples = tf.shape(z1)[1]
        z1_hid = self.l_z1_z2.construct(z2=z2).tensor
        z1_mean = self.l_z1_mean.construct(z1_hid=z1_hid).reshape(
            [-1, n_samples, 16, 16, 64]).tensor
        z1_logvar = self.l_z1_logvar.construct(z1_hid=z1_hid).reshape(
            [-1, n_samples, 16, 16, 64]).tensor
        log_pz1_z2 = norm.logpdf(z1, z1_mean, tf.exp(0.5 * z1_logvar))
        log_pz1_z2 = tf.reduce_sum(log_pz1_z2, [2, 3, 4])

        x_mean = self.l_x_z1.construct(z1=z1).reshape(
            [-1, n_samples, self.n_x]).tensor
        x_scale = tf.exp(self.x_logsd)
        x = tf.expand_dims(x, 1)
        x = tf.clip_by_value(x, -0.5, 0.5)
        log_px_z1 = tf.log(logistic.cdf(x + 1. / 256, x_mean, x_scale) -
                           logistic.cdf(x, x_mean, x_scale) + 1e-8)
        log_px_z1 = tf.reduce_sum(log_px_z1, 2)

        return log_px_z1 + log_pz1_z2 + log_pz2


def q_net(n_x, n_xl, n_samples):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param n_x: Int. The dimension of observed variables (x).
    :param n_samples: A Int or a Tensor of type int. Number of samples of
        latent variables.

    :return: All :class:`Layer` instances needed.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        lx = InputLayer((None, n_x))
        lz1_x = PrettyTensor({'x': lx},
                             pt.template('x').
                             reshape([-1, n_xl, n_xl, 3]).
                             conv2d(3, 64))
        lz1_mean = PrettyTensor({'z1_hid': lz1_x},
                                pt.template('z1_hid').
                                conv2d(5, 64, stride=2, activation_fn=None).
                                reshape([-1, 1, 16, 16, 64]))
        lz1_logvar = PrettyTensor({'z1_hid': lz1_x},
                                  pt.template('z1_hid').
                                  conv2d(5, 64, stride=2, activation_fn=None).
                                  reshape([-1, 1, 16, 16, 64]))
        lz1 = Normal([lz1_mean, lz1_logvar], n_samples)
        lz2_z1 = PrettyTensor({'z1': lz1},
                              pt.template('z1').
                              reshape([-1, 16, 16, 64]).
                              conv2d(3, 64))
        lz2_mean = PrettyTensor({'z2_hid': lz2_z1},
                                pt.template('z2_hid').
                                conv2d(3, 64, activation_fn=None).
                                reshape([-1, n_samples, 16, 16, 64]))
        lz2_logvar = PrettyTensor({'z2_hid': lz2_z1},
                                  pt.template('z2_hid').
                                  conv2d(3, 64, activation_fn=None).
                                  reshape([-1, n_samples, 16, 16, 64]))
        lz2 = Normal([lz2_mean, lz2_logvar])
    return lx, lz1, lz2


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load CIFAR
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    print(x_train.max(), x_train.min())
    print(x_test.max(), x_test.min())
    _, n_xl, _, n_channels = x_train.shape
    n_x = n_xl * n_xl * n_channels
    x_train = x_train.reshape((-1, n_x))
    x_test = x_test.reshape((-1, n_x))
    n_y = t_train.shape[1]

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 3000
    batch_size = 16
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    def build_model(phase, reuse=False):
        with pt.defaults_scope(phase=phase):
            with tf.variable_scope("model", reuse=reuse) as scope:
                model = M1(n_x)
            with tf.variable_scope("variational", reuse=reuse) as scope:
                lx, lz1, lz2 = q_net(n_x, n_xl, n_samples)
        return model, lx, lz1, lz2

    # Build the training computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    model, lx, lz1, lz2 = build_model(pt.Phase.train)
    z1_outputs, z2_outputs = get_output([lz1, lz2], x)
    latent = {'z1': z1_outputs, 'z2': z2_outputs}
    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, latent, reduction_indices=1))
    bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
    # params = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(bits_per_dim, params), 5)
    # grads = list(zip(grads, params))
    grads = optimizer.compute_gradients(bits_per_dim)

    def l2_norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    update_ratio = learning_rate_ph * tf.reduce_mean(tf.pack(list(
        (l2_norm(k) / (l2_norm(v) + 1e-8)) for k, v in grads
        if k is not None)))

    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    eval_model, eval_lx, eval_lz1, eval_lz2 = build_model(
        pt.Phase.test, reuse=True)
    z1_outputs, z2_outputs = get_output([eval_lz1, eval_lz2], x)
    eval_latent = {'z1': z1_outputs, 'z2': z2_outputs}
    eval_lower_bound = tf.reduce_mean(advi(
        eval_model, {'x': x}, latent, reduction_indices=1))
    eval_bits_per_dim = -eval_lower_bound / n_x * 1. / np.log(2.)
    eval_log_likelihood = tf.reduce_mean(is_loglikelihood(
        eval_model, {'x': x}, latent, reduction_indices=1))
    eval_bits_per_dim_ll = -eval_log_likelihood / n_x * 1. / np.log(2.)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            bitss = []
            update_ratios = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]

                _, lb, bits, update_ratio_ = sess.run([infer, lower_bound,
                                               bits_per_dim, update_ratio],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_samples: lb_samples})
                update_ratios.append(update_ratio_)
                lbs.append(lb)
                bitss.append(bits)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {} bits = {}'.format(
                epoch, time_epoch, np.mean(lbs), np.mean(bitss)))
            print('update ratio = {}'.format(np.mean(update_ratios)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lb_bitss = []
                test_lls = []
                test_ll_bitss = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb, test_lb_bits = sess.run(
                        [eval_lower_bound, eval_bits_per_dim],
                        feed_dict={x: test_x_batch,
                                   n_samples: lb_samples})
                    test_ll, test_ll_bits = sess.run(
                        [eval_log_likelihood, eval_bits_per_dim_ll],
                        feed_dict={x: test_x_batch,
                                   n_samples: ll_samples})
                    test_lbs.append(test_lb)
                    test_lb_bitss.append(test_lb_bits)
                    test_lls.append(test_ll)
                    test_ll_bitss.append(test_ll_bits)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}, bits = {}'.format(
                    np.mean(test_lbs), np.mean(test_lb_bitss)))
                print('>> Test log likelihood = {}, bits = {}'.format(
                    np.mean(test_lls), np.mean(test_ll_bitss)))
