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
from matplotlib import pyplot as plt

from examples import conf
from examples.utils import dataset, save_image_collections


@zs.reuse('model')
def mixture_gaussian_vae(observed, n, n_x, n_h, cls, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_logits = tf.get_variable('z_logits', shape=[1, cls],
                                   initializer=tf.random_normal_initializer(
                                       0, 0.1))
        para = z_logits
        z_logits = tf.tile(z_logits, [n, 1])
        z = zs.OnehotCategorical('z', logits=z_logits, n_samples=n_particles,
                                 group_event_ndims=0, dtype=tf.float32)
        # [n_particles, n, cls]
        z = tf.identity(z)
        h_mean = tf.get_variable('h_mean', shape=[cls, n_h],
                                 initializer=tf.random_normal_initializer(
                                     0, 0.1))
        m_para = h_mean
        h_logstd = tf.get_variable('h_logstd', shape=[cls, n_h],
                                   initializer=tf.random_normal_initializer(
                                        0, 0.1))
        s_para = h_logstd
        # [n_particles, cls, n_h]
        h_mean = tf.tile(tf.expand_dims(h_mean, 0), [n_particles, 1, 1])
        h_mean = tf.matmul(z, h_mean)
        h_logstd = tf.tile(tf.expand_dims(h_logstd, 0), [n_particles, 1, 1])
        h_logstd = tf.matmul(z, h_logstd)
        # [n_particles, n, n_x]
        h = zs.Normal('h', h_mean, h_logstd, group_event_ndims=1)
        lx_h = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_h = layers.fully_connected(
            lx_h, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_h, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)

    return model, [tf.squeeze(tf.nn.softmax(para, 1), 0), m_para, s_para], \
           x_logits


@zs.reuse('variational')
def q_net(observed, x, n_h, cls, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lh_x = layers.fully_connected(
             tf.to_float(x), 500, normalizer_fn=layers.batch_norm,
             normalizer_params=normalizer_params)
        lh_x = layers.fully_connected(
             lh_x, 500, normalizer_fn=layers.batch_norm,
             normalizer_params=normalizer_params)
        h_mean = layers.fully_connected(lh_x, n_h, activation_fn=None)
        h_logstd = layers.fully_connected(lh_x, n_h, activation_fn=None)
        h = zs.Normal('h', h_mean, h_logstd, n_samples=n_particles,
                      group_event_ndims=1, is_reparameterized=False)
        lz_h = layers.fully_connected(
            h, 50, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_logits = layers.fully_connected(lz_h, cls, activation_fn=None)
        z = zs.OnehotCategorical('z', z_logits, group_event_ndims=0,
                                 dtype=tf.float32)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    n_h = 40
    cls = 10
    lb_samples = 5
    ll_samples = 100
    epoches = 1000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    save_freq = 100
    image_freq = 2
    result_path = "results/mixture_gaussian_vae"

    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model, _, _ = mixture_gaussian_vae(observed, n, n_x, n_h, cls,
                                           n_particles, is_training)
        log_pz, log_ph_z, log_px_h = model.local_log_prob(['z', 'h', 'x'])
        print(log_pz.get_shape())
        print(log_ph_z.get_shape())
        print(log_px_h.get_shape())
        return log_pz + log_ph_z + log_px_h

    variational = q_net({}, x, n_h, cls, n_particles, is_training)
    qh_samples, log_qh = variational.query('h', outputs=True,
                                           local_log_prob=True)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)

    print('--')
    print(qz_samples.get_shape())
    print(log_qz.get_shape())
    print('--')

    cost, lower_bound = zs.rws(log_joint, {'x': x_obs},
                                 {'z': [qz_samples, log_qz],
                                  'h': [qh_samples, log_qh]}, axis=0)
    lower_bound = tf.reduce_mean(lower_bound)
    cost = tf.reduce_mean(cost)

    jl = log_joint({'z': qz_samples, 'h': qh_samples})
    jl = tf.reduce_mean(jl)
    sq = -tf.reduce_mean(log_qz + log_qh)

    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz],
                             'h': [qh_samples, log_qh]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    n_gen = 100
    _, _, x_logits = mixture_gaussian_vae({}, n_gen, n_x, n_h, cls,
                                          1, is_training=False)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb, pp, ss = sess.run([infer, lower_bound, jl, sq],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                #print('Lower bound = {}, Joint Prob = {}, '
                #      'Sum Q = {}'.format(lb, pp, ss))
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

            if epoch % image_freq == 0:
                images = sess.run(x_gen)
                name = "results/mixture_gaussian_vae/vae.epoch.{}.png".format(epoch)
                save_image_collections(images, name)