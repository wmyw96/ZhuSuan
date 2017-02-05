#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiple stochastic layers of z in MNIST. To compare different number of latent
variables.
"""
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
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    import utils
except:
    raise ImportError()


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_z, n_x, n, n_particles, depth, is_training):
        n_params= {'is_training': is_training,
                   'updates_collections': None}
        with StochasticGraph() as model:
            zs = {}
            z_mean = tf.zeros([n_particles, n_z])
            z_logstd = tf.zeros([n_particles, n_z])
            z = Normal(z_mean, z_logstd, sample_dim=1, n_samples=n)
            zs[0] = z
            for i in range(1, depth):
                lz = layers.fully_connected(
                    z.value, 500)
                lz = layers.fully_connected(
                    lz, 500)
                lz_mean = layers.fully_connected(
                    lz, n_z, activation_fn=None)
                lz_logstd = layers.fully_connected(
                    lz, n_z, activation_fn=None)
                z = Normal(lz_mean, lz_logstd)
                zs[i] = z
            lx_z = layers.fully_connected(
                z.value, 500)
            lx_z = layers.fully_connected(
                lx_z, 500)
            lx_z = layers.fully_connected(lx_z, n_x, activation_fn=None)
            x = Bernoulli(lx_z)
        self.model = model
        self.x = x
        self.zs = zs
        self.n_particles = n_particles

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        zs = list(six.itervalues(self.zs))
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
        inputs = dict(zip(zs, six.itervalues(latent)))
        inputs.update({self.x: x})
        outputs = self.model.get_output(
            zs + [self.x], inputs=inputs)
        log_pzs_and_px_z = sum(tf.reduce_sum(z[1], -1) for z in outputs)
        return log_pzs_and_px_z


def q_net(x, n_z, n_particles, depth, is_training):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: A Tensor.
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_particles: A Tensor or int. Number of samples of latent variables.
    """
    with StochasticGraph() as variational:
        n_params = {'is_training': is_training,
                    'updates_collections': None}
        zs = {}
        lz_x = layers.fully_connected(
            x, 500)
        lz_x = layers.fully_connected(
            lz_x, 500)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = Normal(lz_mean, lz_logstd, sample_dim=0, n_samples=n_particles)
        zs[0] = z
        for i in reversed(range(1, depth)):
            lz = layers.fully_connected(
                z.value, 500)
            lz = layers.fully_connected(
                lz, 500)
            lz_mean = layers.fully_connected(lz, n_z, activation_fn=None)
            lz_logstd = layers.fully_connected(lz, n_z,activation_fn=None)
            z = Normal(lz_mean, lz_logstd)
            zs[i] = z
    return variational, zs


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
    lb_samples = 10
    ll_samples = 500
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    depth = 2

    # Settings
    flags = tf.flags
    flags.DEFINE_string("model_file", "",
                        "restoring model file")
    flags.DEFINE_string("save_dir", os.environ['MODEL_RESULT_PATH_AND_PREFIX'],
                        'path and prefix to save params')
    flags.DEFINE_integer('save_freq', 10, 'save frequency')

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    # optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    optimizer = utils.AdamaxOptimizer(learning_rate_ph, beta1=0.9, beta2=0.999)
    model = M1(n_z, n_x, n, n_particles, depth, is_training)
    variational, lzs = q_net(x, n_z, n_particles, depth, is_training)
    zs_outputs = variational.get_output(list(six.itervalues(lzs)))

    for i, z_output in enumerate(zs_outputs):
        variable_summaries(z_output[0], 'z_{}'.format(i))

    zs_ = [[z_output[0], tf.reduce_sum(z_output[1], -1)]
           for z_output in zs_outputs]

    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, latent=dict(zip(list(six.iterkeys(lzs)), zs_)),
        reduction_indices=0))
    log_likelihood = tf.reduce_mean(is_loglikelihood(
        model, {'x': x}, latent=dict(zip(six.iterkeys(lzs), zs_)),
        reduction_indices=0))

    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    merged = tf.merge_all_summaries()
    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    params = tf.trainable_variables()
    total_size = 0
    for i in params:
        total_size += np.prod([int(s) for s in i.get_shape()])
        print(i.name, i.get_shape())
    print("Num trainable variables: %d" % total_size)

    # Run the inference
    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(flags.FLAGS.save_dir, sess.graph)
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
                try:
                    _, lb, summary = sess.run(
                        [infer, lower_bound, merged],
                        feed_dict={x: x_batch,
                                   learning_rate_ph: learning_rate,
                                   n_particles: lb_samples,
                                   is_training: True})
                except tf.errors.InvalidArgumentError as error:
                    if ("NaN" in error.message) or ("Inf" in error.message):
                        continue
                    raise error
                lbs.append(lb)
                train_writer.add_summary(summary, (epoch - 1) * iters + t)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))

