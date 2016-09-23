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
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi, iwae
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    from deconv import deconv2d
except:
    raise ImportError()


class M3(object):
    def __init__(self, n_x, n_y):
        self.n_y = n_y
        self.n_x = n_x
        # with pt.defaults_scope(activation_fn=None):
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            lz0 = InputLayer((None, None, 100))
            n_samples = InputLayer(())
            ly_p = PrettyTensor({'z0': lz0, 'n_samples': n_samples},
                                pt.template('z0').
                                reshape((-1, 100)).
                                fully_connected(10,
                                                activation_fn=tf.nn.softmax).
                                reshape((-1,
                                         pt.UnboundVariable('n_samples'),
                                         10)))
            ly = Discrete(ly_p, self.n_y)
            lz0_4d = PrettyTensor({'z0': lz0}, pt.template('z0').
                                  reshape((-1, 1, 1, 100)))
            lz1_mean = PrettyTensor({'z0_4d': lz0_4d, 'n_samples': n_samples},
                                    pt.template('z0_4d').
                                    deconv2d(3, 128, edges='VALID',
                                             activation_fn=None).
                                    reshape((-1,
                                             pt.UnboundVariable('n_samples'),
                                             3 * 3 * 128)))
            lz1_logvar = PrettyTensor({'z0_4d': lz0_4d,
                                       'n_samples': n_samples},
                                      pt.template('z0_4d').
                                      deconv2d(3, 128, edges='VALID',
                                               activation_fn=None).
                                      reshape((-1,
                                               pt.UnboundVariable('n_samples'),
                                               3 * 3 * 128)))
            lz1 = ReparameterizedNormal([lz1_mean, lz1_logvar])
            lz1_4d = PrettyTensor({'z1': lz1}, pt.template('z1').
                                  apply(tf.nn.relu).
                                  reshape((-1, 3, 3, 128)))
            lz2_mean = PrettyTensor({'z1_4d': lz1_4d, 'n_samples': n_samples},
                                    pt.template('z1_4d').
                                    deconv2d(5, 64, edges='VALID',
                                             activation_fn=None).
                                    reshape((-1,
                                             pt.UnboundVariable('n_samples'),
                                             7 * 7 * 64)))
            lz2_logvar = PrettyTensor({'z1_4d': lz1_4d,
                                       'n_samples': n_samples},
                                      pt.template('z1_4d').
                                      deconv2d(5, 64, edges='VALID',
                                               activation_fn=None).
                                      reshape((-1,
                                              pt.UnboundVariable('n_samples'),
                                               7 * 7 * 64)))
            lz2 = ReparameterizedNormal([lz2_mean, lz2_logvar])
            lz2_4d = PrettyTensor({'z2': lz2}, pt.template('z2').
                                  apply(tf.nn.relu).
                                  reshape((-1, 7, 7, 64)))
            lz3_mean = PrettyTensor({'z2_4d': lz2_4d, 'n_samples': n_samples},
                                    pt.template('z2_4d').
                                    deconv2d(5, 32, stride=2,
                                             activation_fn=None).
                                    reshape((-1,
                                             pt.UnboundVariable('n_samples'),
                                             14 * 14 * 32)))
            lz3_logvar = PrettyTensor({'z2_4d': lz2_4d,
                                       'n_samples': n_samples},
                                      pt.template('z2_4d').
                                      deconv2d(5, 32, stride=2,
                                               activation_fn=None).
                                      reshape((-1,
                                               pt.UnboundVariable('n_samples'),
                                               14 * 14 * 32)))
            lz3 = ReparameterizedNormal([lz3_mean, lz3_logvar])
            lx = PrettyTensor({'z3': lz3}, pt.template('z3').
                              apply(tf.nn.relu).
                              reshape((-1, 14, 14, 32)).
                              deconv2d(5, 1, stride=2,
                                       activation_fn=tf.nn.sigmoid))
        # lx = PrettyTensor({'z0': lz0}, pt.template('z0').
        #                   reshape((-1, 1, 1, 100)).
        #                   deconv2d(3, 128, edges='VALID').
        #                   batch_normalize(scale_after_normalization=True).
        #                   deconv2d(5, 64, edges='VALID').
        #                   batch_normalize(scale_after_normalization=True).
        #                   deconv2d(5, 32, stride=2).
        #                   batch_normalize(scale_after_normalization=True).
        #                   deconv2d(5, 1, stride=2,
        #                            activation_fn=tf.nn.sigmoid))
        self.n_samples = n_samples
        self.ly = ly
        self.lzs = [lz0, lz1, lz2, lz3]
        # self.lzs = [lz0]
        self.lx = lx

    def log_prob(self, latent, observed, given):
        raise NotImplementedError()


class M3Labeled(M3):
    def __init__(self, n_x, n_y):
        super(M3Labeled, self).__init__(n_x, n_y)

    def log_prob(self, latent, observed, given):
        # zs: (batch_size, n_samples, n_z)
        zs = [latent[k] for k in ['z0', 'z1', 'z2', 'z3']]
        # zs = [latent['z0']]
        # y: (batch_size, n_y), x: (batch_size, n_x)
        y, x = observed['y'], observed['x']

        n_samples = tf.shape(zs[0])[1]
        inputs = {self.ly: tf.expand_dims(y, 1),
                  self.n_samples: n_samples}
        inputs.update(dict(zip(self.lzs, zs)))
        outputs = get_output(self.lzs[1:] + [self.lx, self.ly], inputs)
        z_outputs = outputs[:-2]
        x_mean, _ = outputs[-2]
        _, y_logpdf = outputs[-1]
        log_px_z = tf.reduce_sum(bernoulli.logpdf(
            tf.expand_dims(x, 1),
            tf.reshape(x_mean, (-1, n_samples, self.n_x)), eps=1e-6), 2)
        log_pz0 = tf.reduce_sum(norm.logpdf(zs[0]), 2)
        log_pz = log_pz0 + sum(z_logpdf for _, z_logpdf in z_outputs)
        log_py_z = y_logpdf
        return log_px_z + log_pz + log_py_z


class M3Unlabeled(M3):
    def __init__(self, n_x, n_y):
        super(M3Unlabeled, self).__init__(n_x, n_y)

    def log_prob(self, latent, observed, given):
        # zs: (batch_size, n_samples, n_z)
        zs = [latent[k] for k in ['z0', 'z1', 'z2', 'z3']]
        # zs = [latent['z0']]
        # x: (batch_size, n_x)
        x = observed['x']

        n_samples = tf.shape(zs[0])[1]
        inputs = dict(zip(self.lzs, zs))
        inputs.update({self.n_samples: n_samples})
        outputs = get_output(self.lzs[1:] + [self.lx], inputs)
        z_outputs = outputs[:-1]
        x_mean, _ = outputs[-1]
        log_px_z = tf.reduce_sum(bernoulli.logpdf(
            tf.expand_dims(x, 1),
            tf.reshape(x_mean, (-1, n_samples, self.n_x)), eps=1e-6), 2)
        log_pz0 = tf.reduce_sum(norm.logpdf(zs[0]), 2)
        log_pz = log_pz0 + sum(z_logpdf for _, z_logpdf in z_outputs)
        return log_px_z + log_pz


def q_net(n_x, n_xl, n_y, n_z, n_samples):
    # with pt.defaults_scope(activation_fn=None):
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        lx = InputLayer((None, n_x))
        # lz_x = PrettyTensor({'x': lx}, pt.template('x').
        #                     reshape([-1, n_xl, n_xl, 1]).
        #                     conv2d(5, 32, stride=2).
        #                     batch_normalize(scale_after_normalization=True).
        #                     conv2d(5, 64, stride=2).
        #                     batch_normalize(scale_after_normalization=True).
        #                     conv2d(5, 128, edges='VALID').
        #                     batch_normalize(scale_after_normalization=True).
        #                     dropout(0.9).
        #                     flatten())
        # lz0_mean = PrettyTensor({'z': lz_x}, pt.template('z').
        #                         fully_connected(n_z, activation_fn=None).
        #                         reshape((-1, 1, n_z)))
        # lz0_logvar = PrettyTensor({'z': lz_x}, pt.template('z').
        #                           fully_connected(n_z, activation_fn=None).
        #                           reshape((-1, 1, n_z)))
        lx_4d = PrettyTensor({'x': lx}, pt.template('x').
                             reshape((-1, n_xl, n_xl, 1)))
        lz3_mean = PrettyTensor({'x_4d': lx_4d}, pt.template('x_4d').
                                conv2d(5, 32, stride=2,
                                       activation_fn=None).
                                reshape((-1, 1, 14 * 14 * 32)))
        lz3_logvar = PrettyTensor({'x_4d': lx_4d}, pt.template('x_4d').
                                  conv2d(5, 32, stride=2,
                                         activation_fn=None).
                                  reshape((-1, 1, 14 * 14 * 32)))
        lz3 = ReparameterizedNormal([lz3_mean, lz3_logvar], n_samples)
        lz3_4d = PrettyTensor({'z3': lz3}, pt.template('z3').
                              apply(tf.nn.relu).
                              reshape((-1, 14, 14, 32)))
        lz2_mean = PrettyTensor({'z3_4d': lz3_4d}, pt.template('z3_4d').
                                conv2d(5, 64, stride=2,
                                       activation_fn=None).
                                reshape((-1, n_samples, 7 * 7 * 64)))
        lz2_logvar = PrettyTensor({'z3_4d': lz3_4d}, pt.template('z3_4d').
                                  conv2d(5, 64, stride=2,
                                         activation_fn=None).
                                  reshape((-1, n_samples, 7 * 7 * 64)))
        lz2 = ReparameterizedNormal([lz2_mean, lz2_logvar])
        lz2_4d = PrettyTensor({'z2': lz2}, pt.template('z2').
                              apply(tf.nn.relu).
                              reshape((-1, 7, 7, 64)))
        lz1_mean = PrettyTensor({'z2_4d': lz2_4d}, pt.template('z2_4d').
                                conv2d(5, 128, edges='VALID',
                                       activation_fn=None).
                                reshape((-1, n_samples, 3 * 3 * 128)))
        lz1_logvar = PrettyTensor({'z2_4d': lz2_4d}, pt.template('z2_4d').
                                  conv2d(5, 128, edges='VALID',
                                         activation_fn=None).
                                  reshape((-1, n_samples, 3 * 3 * 128)))
        lz1 = ReparameterizedNormal([lz1_mean, lz1_logvar])
        lz1_4d = PrettyTensor({'z1': lz1}, pt.template('z1').
                              apply(tf.nn.relu).
                              reshape((-1, 3, 3, 128)))
        lz0_mean = PrettyTensor({'z1_4d': lz1_4d}, pt.template('z1_4d').
                                conv2d(3, 100, edges='VALID',
                                       activation_fn=None).
                                reshape((-1, n_samples, 100)))
        lz0_logvar = PrettyTensor({'z1_4d': lz1_4d}, pt.template('z1_4d').
                                  conv2d(3, 100, edges='VALID',
                                         activation_fn=None).
                                  reshape((-1, n_samples, 100)))
        lz0 = ReparameterizedNormal([lz0_mean, lz0_logvar])
    lzs = [lz0, lz1, lz2, lz3]
    # lzs = [lz0]
    return lx, lzs


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    np.random.seed(1234)
    x_labeled, t_labeled, x_unlabeled, x_test, t_test = \
        dataset.load_mnist_semi_supervised(data_path, one_hot=True)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_labeled, n_x = x_labeled.shape
    n_xl = np.sqrt(n_x)
    n_y = t_labeled.shape[1]

    # Define model parameters
    n_z = 100

    # Define training/evaluation parameters
    lb_samples = 1
    beta = 1200.
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_unlabeled.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.0003
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    n_samples = tf.placeholder(tf.int32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    x_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_x))
    y_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_y))
    x_unlabeled_ph = tf.placeholder(tf.float32, shape=(None, n_x))

    # def build_model(test=False):
    #     with pt.defaults_scope(phase=pt.Phase.train):
    test = False
    with tf.variable_scope("model", reuse=test) as scope:
        m3_labeled = M3Labeled(n_x, n_y)
    with tf.variable_scope("model", reuse=True) as scope:
        m3_unlabeled = M3Unlabeled(n_x, n_y)
    with tf.variable_scope("variational", reuse=test) as scope:
        lx, lzs = q_net(n_x, n_xl, n_y, n_z, n_samples)

    # Labeled
    inputs = {lx: x_labeled_ph}
    z_outputs = get_output(lzs, inputs)
    labeled_observed = {'x': x_labeled_ph, 'y': y_labeled_ph}
    labeled_latent = dict(('z' + str(i), z_outputs[i])
                          for i in range(len(z_outputs)))
    labeled_lower_bound = advi(m3_labeled, labeled_observed,
                               labeled_latent, reduction_indices=1)

    # Unlabeled
    inputs = {lx: x_unlabeled_ph}
    z_outputs = get_output(lzs, inputs)
    unlabeled_latent = dict(('z' + str(i), z_outputs[i])
                            for i in range(len(z_outputs)))
    unlabeled_observed = {'x': x_unlabeled_ph}
    unlabeled_lower_bound = advi(m3_unlabeled, unlabeled_observed,
                                 unlabeled_latent, reduction_indices=1)

    # Build classifier
    with tf.variable_scope("variational", reuse=True) as scope:
        lx, lzs = q_net(n_x, n_xl, n_y, n_z, 1)
    inputs = {lx: x_labeled_ph}
    z0 = get_output(lzs[0], inputs, deterministic=True)
    inputs = {m3_labeled.lzs[0]: z0, m3_labeled.n_samples: 1}
    y_p = tf.reshape(
        get_output(m3_labeled.ly, inputs, deterministic=True),
        (-1, n_y))
    pred_y = tf.argmax(y_p, 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y,
                         tf.argmax(y_labeled_ph, 1)), tf.float32) /
        tf.cast(tf.shape(y_labeled_ph)[0], tf.float32))
    # return labeled_lower_bound, unlabeled_lower_bound, acc

    # labeled_lower_bound, unlabeled_lower_bound, acc = build_model()
    # eval_labeled_lower_bound, eval_unlabeled_lower_bound, eval_acc = \
    #     build_model(test=True)

    # Gather gradients
    cost = -(unlabeled_lower_bound) / 2.
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # graph_writer = tf.train.SummaryWriter('/home/ishijiaxin/log',
    #                                       tf.get_default_graph())

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_unlabeled)
            lbs_labeled = []
            lbs_unlabeled = []
            train_accs = []
            for t in range(iters):
                labeled_indices = np.random.randint(0, n_labeled,
                                                    size=batch_size)
                x_labeled_batch = x_labeled[labeled_indices]
                y_labeled_batch = t_labeled[labeled_indices]
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                x_labeled_batch = np.random.binomial(
                    n=1, p=x_labeled_batch,
                    size=x_labeled_batch.shape).astype('float32')
                x_unlabeled_batch = np.random.binomial(
                    n=1, p=x_unlabeled_batch,
                    size=x_unlabeled_batch.shape).astype('float32')
                _, lb_labeled, lb_unlabeled, train_acc = sess.run(
                    [infer, labeled_lower_bound, unlabeled_lower_bound, acc],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch,
                               x_unlabeled_ph: x_unlabeled_batch,
                               learning_rate_ph: learning_rate,
                               n_samples: lb_samples})
                lbs_labeled.append(lb_labeled)
                lbs_unlabeled.append(lb_unlabeled)
                train_accs.append(train_acc)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s), Lower bound: labeled = {}, '
                  'unlabeled = {} Accuracy: {:.2f}%'.
                  format(epoch, time_epoch, np.mean(lbs_labeled),
                         np.mean(lbs_unlabeled), np.mean(train_accs) * 100.))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lls_labeled = []
                test_lls_unlabeled = []
                test_accs = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_y_batch = t_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_ll_labeled, test_ll_unlabeled, test_acc = sess.run(
                        [labeled_lower_bound, unlabeled_lower_bound,
                         acc],
                        feed_dict={x_labeled_ph: test_x_batch,
                                   y_labeled_ph: test_y_batch,
                                   x_unlabeled_ph: test_x_batch,
                                   n_samples: lb_samples})
                    test_lls_labeled.append(test_ll_labeled)
                    test_lls_unlabeled.append(test_ll_unlabeled)
                    test_accs.append(test_acc)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound: labeled = {}, unlabeled = {}'.
                      format(np.mean(test_lls_labeled),
                             np.mean(test_lls_unlabeled)))
                print('>> Test accuracy: {:.2f}%'.format(
                    100. * np.mean(test_accs)))
