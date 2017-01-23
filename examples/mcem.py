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
    import zhusuan as zs
    from zhusuan.mcmc.hmc import HMC
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


def vae(observed, n, n_x, n_z, n_particles, is_training):
    with zs.StochasticGraph(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n_particles, n_z])
        z_logstd = tf.zeros([n_particles, n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=1, n_samples=n)
        lx_z = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_mean = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_mean)
    return model

def try_vae(observed, n, n_x, n_z, n_particles, is_training):
    try:
        with tf.variable_scope("vae", reuse=True):
            return vae(observed, n, n_x, n_z, n_particles, is_training)
    except:
        with tf.variable_scope("vae"):
            return vae(observed, n, n_x, n_z, n_particles, is_training)

# def q_net(x, n_z, n_particles, is_training):
#     with zs.StochasticGraph() as variational:
#         normalizer_params = {'is_training': is_training,
#                              'updates_collections': None}
#         lz_x = layers.fully_connected(
#             x, 500, normalizer_fn=layers.batch_norm,
#             normalizer_params=normalizer_params)
#         lz_x = layers.fully_connected(
#             lz_x, 500, normalizer_fn=layers.batch_norm,
#             normalizer_params=normalizer_params)
#         lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
#         lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
#         z = zs.Normal('z', lz_mean, lz_logstd, sample_dim=0,
#                       n_samples=n_particles)
#     return variational


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
    mcmc_iters = 5
    n_leapfrogs = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 1000
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.float32)
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    def log_joint(latent, observed, given):
        x = observed['x']
        z = latent['z']
        x = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
        model = try_vae({'x': x, 'z': z}, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return tf.reduce_sum(log_pz, -1) + tf.reduce_sum(log_px_z, -1)

    def log_joint2(latent, observed, given):
        x = tf.expand_dims(observed['x'], 0)
        z = tf.expand_dims(latent['z'], 0)
        model = try_vae({'x': x, 'z': z}, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return tf.squeeze(tf.reduce_sum(log_pz, -1)) + tf.squeeze(tf.reduce_sum(log_px_z, -1))

    # HMC
    z_train = tf.Variable(tf.zeros([batch_size, n_z]), trainable=False)
    reset_z_train = tf.assign(z_train, tf.zeros([batch_size, n_z]))
    z_train_input = tf.placeholder(tf.float32, shape=[mcmc_iters, batch_size, n_z])
    train_hmc = HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs)
    train_sampler = train_hmc.sample(log_joint2,
                                     {'x': x}, {'z': z_train}, chain_axis=0)

    obj = tf.reduce_mean(log_joint({'z': z_train_input}, {'x': x}, None))

    # variational = q_net(x, n_z, n_particles, is_training)
    # qz_samples, log_qz = variational.query('z', outputs=True,
    #                                        local_log_prob=True)
    # log_qz = tf.reduce_sum(log_qz, -1)
    # with tf.variable_scope("model"):
    #     lower_bound = tf.reduce_mean(zs.advi(
    #         log_joint, {'x': x}, {'z': [qz_samples, log_qz]},
    #         reduction_indices=0))
    # with tf.variable_scope("model", reuse=True):
    #     log_likelihood = tf.reduce_mean(zs.is_loglikelihood(
    #         log_joint, {'x': x}, {'z': [qz_samples, log_qz]},
    #         reduction_indices=0))

    grads = optimizer.compute_gradients(-obj)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Begin GD')

        x_batch = x_train[:batch_size]
        x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})

        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})

                # E-step: HMC
                sess.run(reset_z_train)
                samples = []
                for i in range(mcmc_iters):
                    sample, _, oh, nh, ll, acc = sess.run(train_sampler,
                                                          feed_dict={x: x_batch_bin,
                                                                     is_training: True,
                                                                     n_particles: 1})
                    samples.append(sample[0])
                    print('Step {}: ll = {}, acceptance = {}'.format(i, np.mean(ll), np.mean(acc)))

                samples = np.array(samples)

                # M-step
                _, lb = sess.run([infer, obj],
                                 feed_dict={x: x_batch_bin,
                                            z_train_input: samples,
                                            learning_rate_ph: learning_rate,
                                            n_particles: mcmc_iters,
                                            is_training: True})
                lbs.append(lb)

            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))


        # for epoch in range(1, epoches + 1):
        #     time_epoch = -time.time()
        #     if epoch % anneal_lr_freq == 0:
        #         learning_rate *= anneal_lr_rate
        #     np.random.shuffle(x_train)
        #     lbs = []
        #     for t in range(iters):
        #         x_batch = x_train[t * batch_size:(t + 1) * batch_size]
        #         x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
        #         _, lb = sess.run([infer, lower_bound],
        #                          feed_dict={x: x_batch_bin,
        #                                     learning_rate_ph: learning_rate,
        #                                     n_particles: lb_samples,
        #                                     is_training: True})
        #         lbs.append(lb)
        #     time_epoch += time.time()
        #     print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
        #         epoch, time_epoch, np.mean(lbs)))
        #     if epoch % test_freq == 0:
        #         time_test = -time.time()
        #         test_lbs = []
        #         test_lls = []
        #         for t in range(test_iters):
        #             test_x_batch = x_test[
        #                 t * test_batch_size:(t + 1) * test_batch_size]
        #             test_lb = sess.run(lower_bound,
        #                                feed_dict={x: test_x_batch,
        #                                           n_particles: lb_samples,
        #                                           is_training: False})
        #             test_ll = sess.run(log_likelihood,
        #                                feed_dict={x: test_x_batch,
        #                                           n_particles: ll_samples,
        #                                           is_training: False})
        #             test_lbs.append(test_lb)
        #             test_lls.append(test_ll)
        #         time_test += time.time()
        #         print('>>> TEST ({:.1f}s)'.format(time_test))
        #         print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
        #         print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
