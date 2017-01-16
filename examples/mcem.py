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
try:
    from zhusuan.model import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
    from zhusuan.mcmc.hmc import HMC
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_z, n_x, n, n_particles, is_training):
        with StochasticGraph() as model:
            # (n_particles, n_z)
            z_mean = tf.zeros([n_particles, n_z])
            # (n_particles, n_z)
            z_logstd = tf.zeros([n_particles, n_z])
            # (n_particles, n, n_z)
            z = Normal(z_mean, z_logstd, sample_dim=1, n_samples=n)
            # (n_particles, n, 500)
            lx_z = layers.fully_connected(
                z.value, 500, normalizer_fn=layers.batch_norm,
                normalizer_params={'is_training': is_training,
                                   'updates_collections': None})
            # (n_particles, n, 500)
            lx_z = layers.fully_connected(
                lx_z, 500, normalizer_fn=layers.batch_norm,
                normalizer_params={'is_training': is_training,
                                   'updates_collections': None})
            # (n_particles, n, n_x)
            lx_z = layers.fully_connected(lx_z, n_x, activation_fn=None)
            # (n_particles, n, n_x)
            x = Bernoulli(lx_z)
        self.model = model
        self.x = x
        self.z = z
        self.n_particles = n_particles

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        # (n_particles, n, n_z)
        z = latent['z']
        # (n, n_x)
        x = observed['x']
        # (n_particles, n, n_x)
        x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
        # (n_particles, n, n_z), (n_particles, n, n_x)
        z_out, x_out = self.model.get_output([self.z, self.x],
                                             inputs={self.z: z, self.x: x})
        # (n_particles, n)
        log_px_z = tf.reduce_sum(x_out[1], -1)
        # (n_particles, n)
        log_pz = tf.reduce_sum(z_out[1], -1)
        # (n_particles, n)
        return log_px_z + log_pz


def q_net(x, n_z, n_particles, is_training):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: A Tensor.
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_particles: A Tensor or int. Number of samples of latent variables.
    """
    with StochasticGraph() as variational:
        # (n, 500)
        lz_x = layers.fully_connected(
            x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params={'is_training': is_training,
                               'updates_collections': None})
        # (n, 500)
        lz_x = layers.fully_connected(
            lz_x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params={'is_training': is_training,
                               'updates_collections': None})
        # (n, n_z)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        # (n, n_z)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        # (n_particles, n, n_z)
        z = Normal(lz_mean, lz_logstd, sample_dim=0, n_samples=n_particles)
    return variational, z


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    n_chains = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 1
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build model
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    model = M1(n_z, n_x, n, n_particles, is_training)
    var_list = tf.trainable_variables()

    # HMC
    sampler = HMC(step_size=0.1, n_leapfrogs=1, target_acceptance_rate=0.8)
    mass = [tf.ones([n_particles, n, n_z])]
    z = tf.zeros([n_particles, n, n_z])
    # (n_particles, n, n_z)
    z_samples = sampler.sample(
        model.log_prob, {'x': x}, {'z': z}, axis=2, mass=mass)[0][0]
    z_samples = tf.stop_gradient(z_samples)

    # Build updates
    # (n_particles, n)
    log_joint = model.log_prob({'z': z_samples}, {'x': x}, None)
    log_joint = tf.reduce_mean(log_joint)

    # Build the computation graph
    # is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    # learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    # n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    # x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    # n = tf.shape(x)[0]
    # optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    # model = M1(n_z, n_x, n, n_particles, is_training)
    # variational, lz = q_net(x, n_z, n_particles, is_training)
    # z, z_logpdf = variational.get_output(lz)
    # z_logpdf = tf.reduce_sum(z_logpdf, -1)
    # lower_bound = tf.reduce_mean(advi(
    #     model, {'x': x}, {'z': [z, z_logpdf]}, reduction_indices=0))
    # log_likelihood = tf.reduce_mean(is_loglikelihood(
    #     model, {'x': x}, {'z': [z, z_logpdf]}, reduction_indices=0))

    grads = optimizer.compute_gradients(-log_joint, var_list=var_list)
    infer = optimizer.apply_gradients(grads)

    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, log_joint],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_particles: n_chains,
                                            is_training: True})
                print('Iter', t, lb)
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Log joint = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(log_joint,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: n_chains,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test log likelihood = {}'.format(np.mean(test_lbs)))
