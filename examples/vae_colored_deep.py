#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import prettytensor as pt
import dataset
from collections import namedtuple
from deconv import deconv2d
from adamax import AdamaxOptimizer
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.utils import log_mean_exp
except:
    raise ImportError()


def save_image_collections(x, filename, shape=(10, 10), scaleeach=False, transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: uint8 numpy array
        Input image collections.(number_of_images, rows, columns, channels)or(number_of_images, channels, rows, columns)
    :param scaleeach: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels), i.e., put channels behind.
    :return: uint8 numpy array
        The output image.
    """
    n = x.shape[0]

    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scaleeach is True:
        for i in xrange(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r*c < n:
        print 'Shape too small to contain all images'
    h, w = x.shape[1], x.shape[2]
    ret = np.zeros((h*r, w*c, n_channels), dtype='uint8')
    for i in xrange(r):
        for j in xrange(c):
            if i*c+j < n:
                ret[i*h:(i+1)*h, j*w:(j+1)*w, :] = x[i*c+j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret


def resize_nearest_neighbor(x, scale):
    input_shape = tf.shape(x)
    size = [tf.cast(tf.cast(input_shape[1], tf.float32) * scale, tf.int32),
            tf.cast(tf.cast(input_shape[2], tf.float32) * scale, tf.int32)]
    x = tf.image.resize_nearest_neighbor(x, size)
    return x


def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
    return tf.reduce_sum(logp, 2)


def gaussian_diag_logps(mean, logvar, sample=None, expand_dims=True):
    if expand_dims is True:
        mean = tf.expand_dims(mean, 1)
        logvar = tf.expand_dims(logvar, 1)
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) /
                   tf.exp(logvar))


class DiagonalGaussian(object):
    """
    :expand_dim : if True, insert n_samples dim next to batch_size to mean & var
    """

    def __init__(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar

    def samples(self, k=1, expand_dim=True):
        if expand_dim is True:
            mean = tf.expand_dims(self.mean, 1)
            logvar = tf.expand_dims(self.logvar, 1)
        else:
            mean = self.mean
            logvar = self.logvar
        if k == 1:
            noise = tf.random_normal(tf.shape(mean))
            return mean + tf.exp(0.5 * logvar) * noise
        else:
            noise = tf.random_normal(tf.concat(0,
                                               [tf.pack([tf.shape(mean)[0], k]),
                                                tf.shape(mean)[2:]]))
            samples_n = mean + tf.exp(0.5 * logvar) * noise
            samples_n.set_shape([None, k] +
                                [None] * len(mean.get_shape()[2:]))
            return samples_n

    def logps(self, sample, expand_dim=True):
        return gaussian_diag_logps(self.mean, self.logvar, sample, expand_dim)


class M1(object):
    def __init__(self, batch_size, test_batch_size, n_z, n_y, groups,
                 kl_min, ll_samples, lb_samples, image_size, beta):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_z = n_z
        self.n_y = n_y
        self.groups = groups
        self.kl_min = kl_min
        self.ll_samples = ll_samples
        self.lb_samples = lb_samples
        self.image_size = image_size
        self.n_x = 3 * image_size * image_size
        self.beta = beta
        self.x_l = tf.placeholder(tf.float32, shape=(None, self.n_x))
        self.x_u = tf.placeholder(tf.float32, shape=(None, self.n_x))
        self.y_l = tf.placeholder(tf.float32, shape=(None, self.n_y))
        self.x_logsd = tf.get_variable('x_logsd', (),
                                       initializer=tf.zeros_initializer)
        # self.h_top = tf.get_variable('h_top', [self.groups[-1].num_filters],
        #                              initializer=tf.zeros_initializer)
        self.x_enc, self.up_convs = self.q_net()
        self.l_x_z, self.down_convs = self.p_net()
        self.l_y_x = self.x2y()

    def x2y(self):
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            l_y_x = (pt.template('x').
                     reshape([-1, 32, 32, 1]).
                     conv2d(3, 32).
                     conv2d(3, 32).
                     max_pool(2, 2).
                     conv2d(3, 64).
                     conv2d(3, 64).
                     max_pool(2, 2).
                     conv2d(3, 128).
                     conv2d(3, 128).
                     max_pool(2, 2).
                     flatten().
                     fully_connected(500).
                     fully_connected(self.n_y, activation_fn=tf.nn.softmax)
                     )
        return l_y_x

    def q_net(self):
        up_convs = {}
        x_enc = (pt.template('x').
                 reshape([-1, self.image_size, self.image_size, 3]).
                 conv2d(5, self.groups[0].num_filters, stride=2,
                        activation_fn=None))

        for group_i, group in enumerate(self.groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)
                stride = 1
                if group_i > 0 and block_i == 0:
                    stride = 2
                up_convs[name+'_up_conv1'] = (
                    pt.template('h1').
                    conv2d(3, group.num_filters + 2*self.n_z, stride=stride,
                           name=name + '_up_conv1_h_z',
                           activation_fn=None))

                up_convs[name+'_up_conv2'] = (
                    pt.template('h2').
                    conv2d(3, group.num_filters, name=name + '_up_conv2',
                           activation_fn=None))
        return x_enc, up_convs

    def p_net(self):
        down_convs = {}
        for group_i, group in reversed(list(enumerate(self.groups))):
            for block_i in reversed(range(group.num_blocks)):
                name = 'group_%d/block_%d' % (group_i, block_i)
                stride = 1
                if group_i > 0 and block_i == 0:
                    stride = 2
                down_convs[name+'_down_conv1'] = (
                    pt.template('h1').
                    deconv2d(3, 1 * group.num_filters + 4 * self.n_z,
                             name=name+'_down_conv1',
                             activation_fn=None))

                down_convs[name+'_down_conv2'] = (
                    pt.template('h2_plus_z').
                    deconv2d(3, group.num_filters, stride=stride,
                             name=name+'_down_conv2',
                             activation_fn=None))

        l_x_z = (pt.template('h').
                 deconv2d(5, 3, stride=2, activation_fn=None))
        return l_x_z, down_convs

    def forward_labeled(self, batch_size, n_samples):
        h1 = self.x_enc.construct(x=self.x_l).tensor
        qz_mean = {}
        qz_logsd = {}
        # bottom up for labeled data
        for group_i, group in enumerate(self.groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)
                h2_z = tf.nn.elu(h1)
                h2_z = self.up_convs[name+'_up_conv1'].construct(
                    h1=h2_z).tensor
                qz_mean[name], qz_logsd[name], h2 = split(
                    h2_z, -1, [self.n_z] * 2 + [group.num_filters])
                h2 = tf.nn.elu(h2)
                h = self.up_convs[name+'_up_conv2'].construct(h2=h2).tensor
                if group_i > 0 and block_i == 0:
                    h1 = resize_nearest_neighbor(h1, 0.5)
                h1 += 0.1 * h
        y_l = tf.reshape(self.y_l, [-1, 1, 1, self.n_y])
        y_l = tf.tile(y_l, [n_samples, self.groups[-1].map_size,
                            self.groups[-1].map_size, 1])
        h = y_l
        kl_cost_dic = {}
        for group_i, group in reversed(list(enumerate(self.groups))):
            for block_i in reversed(range(group.num_blocks)):
                name = 'group_%d/block_%d' % (group_i, block_i)
                input_h = h
                h1 = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv1'].construct(h1=h1).tensor

                pz_mean, pz_logsd, rz_mean, rz_logsd, h = \
                    split(h, -1, [self.n_z] * 4 + [group.num_filters])
                pz_mean, pz_logsd, rz_mean, rz_logsd = map(
                    lambda k: tf.reshape(k, [batch_size, n_samples,
                                             group.map_size, group.map_size,
                                             self.n_z]),
                    [pz_mean, pz_logsd, rz_mean, rz_logsd])

                prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
                post_z_mean = rz_mean + tf.expand_dims(qz_mean[name], 1)
                post_z_logsd = rz_logsd + tf.expand_dims(qz_logsd[name], 1)

                # To be done: posterior with precision average and correct mu

                posterior = DiagonalGaussian(post_z_mean, 2 * post_z_logsd)
                # context = tf.expand_dims(up_context[name], 1) + down_context
                z = posterior.samples(expand_dim=False)  # (bs,k,map,map,c)
                logqs = posterior.logps(z, expand_dim=False)  # (bs,k,map,map,c)

                logps = prior.logps(z, expand_dim=False)
                kl_cost_dic[name] = logqs - logps  # (batch_size,k,map,map,c)
                z = tf.reshape(z, [-1, group.map_size,
                                   group.map_size, self.n_z])
                h = tf.concat(3, [h, z])
                if group_i > 0 and block_i == 0:
                    input_h = resize_nearest_neighbor(input_h, 2)
                h = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv2'].construct(
                    h2_plus_z=h).tensor
                h = input_h + 0.1 * h
        h = tf.nn.elu(h)
        x_mean = self.l_x_z.construct(h=h).reshape([-1, n_samples, self.n_x]
                                                   ).tensor
        x_mean = tf.clip_by_value(x_mean * 0.1, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        x = tf.expand_dims(self.x_l, 1)
        log_px_z1 = discretized_logistic(x_mean, self.x_logsd, sample=x)
        return log_px_z1, kl_cost_dic
        # E_q(z|x,y) log p(x|y,z) + log p(y) + log p(z) - log q(z|x, y)

    def forward_unlabeled(self, batch_size, n_samples):
        h1 = self.x_enc.construct(x=self.x_u).tensor
        qz_mean = {}
        qz_logsd = {}
        # bottom up for unlabeled data
        for group_i, group in enumerate(self.groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)
                h2_z = tf.nn.elu(h1)
                h2_z = self.up_convs[name+'_up_conv1'].construct(
                    h1=h2_z).tensor
                qz_mean[name], qz_logsd[name], h2 = split(
                    h2_z, -1, [self.n_z] * 2 + [group.num_filters])
                h2 = tf.nn.elu(h2)
                h = self.up_convs[name+'_up_conv2'].construct(h2=h2).tensor
                if group_i > 0 and block_i == 0:
                    h1 = resize_nearest_neighbor(h1, 0.5)
                h1 += 0.1 * h
        y_l = tf.reshape(self.y_l, [-1, 1, 1, self.n_y])
        y_l = tf.tile(y_l, [n_samples, self.groups[-1].map_size,
                            self.groups[-1].map_size, 1])

        y_u = self.l_y_x.construct(x=self.x_u).tensor

        h = y_l
        kl_cost_dic = {}
        for group_i, group in reversed(list(enumerate(self.groups))):
            for block_i in reversed(range(group.num_blocks)):
                name = 'group_%d/block_%d' % (group_i, block_i)
                input_h = h
                h1 = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv1'].construct(h1=h1).tensor

                pz_mean, pz_logsd, rz_mean, rz_logsd, h = \
                    split(h, -1, [self.n_z] * 4 + [group.num_filters])
                pz_mean, pz_logsd, rz_mean, rz_logsd = map(
                    lambda k: tf.reshape(k, [batch_size, n_samples,
                                             group.map_size, group.map_size,
                                             self.n_z]),
                    [pz_mean, pz_logsd, rz_mean, rz_logsd])

                prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
                post_z_mean = rz_mean + tf.expand_dims(qz_mean[name], 1)
                post_z_logsd = rz_logsd + tf.expand_dims(qz_logsd[name], 1)

                # To be done: posterior with precision average and correct mu

                posterior = DiagonalGaussian(post_z_mean, 2 * post_z_logsd)
                # context = tf.expand_dims(up_context[name], 1) + down_context
                z = posterior.samples(expand_dim=False)  # (bs,k,map,map,c)
                logqs = posterior.logps(z, expand_dim=False)  # (bs,k,map,map,c)

                logps = prior.logps(z, expand_dim=False)
                kl_cost_dic[name] = logqs - logps  # (batch_size,k,map,map,c)
                z = tf.reshape(z, [-1, group.map_size,
                                   group.map_size, self.n_z])
                h = tf.concat(3, [h, z])
                if group_i > 0 and block_i == 0:
                    input_h = resize_nearest_neighbor(input_h, 2)
                h = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv2'].construct(
                    h2_plus_z=h).tensor
                h = input_h + 0.1 * h
        h = tf.nn.elu(h)
        x_mean = self.l_x_z.construct(h=h).reshape([-1, n_samples, self.n_x]
                                                   ).tensor
        x_mean = tf.clip_by_value(x_mean * 0.1, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        x = tf.expand_dims(self.x_l, 1)
        log_px_z1 = discretized_logistic(x_mean, self.x_logsd, sample=x)
        return log_px_z1, kl_cost_dic
        # E_q(z|x,y) log p(x|y,z) + log p(y) + log p(z) - log q(z|x, y)

    def advi_klmin(self, batch_size, n_samples):
        total_kl_cost = 0
        total_kl_obj = 0
        log_px_z1, kl_cost_dic = self.forward(batch_size, n_samples)
        for z in kl_cost_dic:
            kl_cost = kl_cost_dic[z]
            if self.kl_min > 0:
                kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]), [0, 1],
                                        keep_dims=True)  # average over bs and k
                kl_ave = tf.maximum(kl_ave, self.kl_min)
                kl_ave = tf.tile(kl_ave, [batch_size, n_samples, 1])
                kl_obj = tf.reduce_sum(kl_ave, [2])
            else:
                kl_obj = tf.reduce_sum(kl_cost, [2, 3, 4])
            kl_cost = tf.reduce_sum(kl_cost, [2, 3, 4])
            total_kl_cost += kl_cost
            total_kl_obj += kl_obj
            tf.scalar_summary("train_sum/kl_cost_" + z, tf.reduce_mean(kl_cost))
            tf.scalar_summary("train_sum/kl_obj_" + z, tf.reduce_mean(kl_obj))

        lower_bound = tf.reduce_mean(log_px_z1 - total_kl_cost,
                                     reduction_indices=1)
        lower_bound_obj = tf.reduce_mean(log_px_z1 - total_kl_obj,
                                         reduction_indices=1)
        tf.scalar_summary("train_sum/log_pxz", -tf.reduce_mean(log_px_z1))
        tf.scalar_summary("train_sum/kl_obj", tf.reduce_mean(total_kl_obj))
        tf.scalar_summary("train_sum/kl_cost", tf.reduce_mean(total_kl_cost))

        return lower_bound, lower_bound_obj

    def lb_is_loglikelihood(self, batch_size, n_samples):
        log_px_z1, kl_cost_dic = self.forward(batch_size, n_samples)
        total_kl_cost = 0
        for z in kl_cost_dic:
            total_kl_cost += tf.reduce_sum(kl_cost_dic[z], [2, 3, 4])
        lower_bound = tf.reduce_mean(log_px_z1 - total_kl_cost,
                                     reduction_indices=1)
        log_w = log_px_z1 - total_kl_cost
        return lower_bound, log_mean_exp(log_w, 1)

    def generate_samples(self, batch_size):
        h_top = tf.reshape(self.h_top, [1, 1, 1, -1])
        h = tf.tile(h_top, [batch_size,
                            self.groups[-1].map_size,
                            self.groups[-1].map_size, 1])
        for group_i, group in reversed(list(enumerate(self.groups))):
            for block_i in reversed(range(group.num_blocks)):
                name = 'group_%d/block_%d' % (group_i, block_i)
                input_h = h
                h1 = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv1'].construct(h1=h1).tensor
                pz_mean, pz_logsd, _, _, h = split(h, -1, [self.n_z] * 4 + [group.num_filters])
                prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
                z = prior.samples(k=1, expand_dim=False)  # (bs,k,map,map,c)
                h = tf.concat(3, [h, z])
                if group_i > 0 and block_i == 0:
                    input_h = resize_nearest_neighbor(input_h, 2)
                # h = tf.nn.elu(h)
                h = self.down_convs[name+'_down_conv2'].construct(
                    h2_plus_z=h).tensor
                h = input_h + 0.1 * h
        # h = tf.nn.elu(h)
        x_mean = self.l_x_z.construct(h=h).tensor
        x_mean = tf.clip_by_value(x_mean * 0.1, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        return x_mean

    def test(self, x_test, sess, test_fetch):
        test_iters = x_test.shape[0] // self.test_batch_size
        time_test = -time.time()
        test_lbs = []
        test_lb_bitss = []
        test_lls = []
        test_ll_bitss = []
        for t in range(test_iters):
            test_x_batch = x_test[
                           t * self.test_batch_size:
                           (t + 1) * self.test_batch_size]
            test_lb, test_lb_bits, test_ll, test_ll_bits = sess.run(
                test_fetch, feed_dict={self.x: test_x_batch})
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

    def train(self, config):
        tf.set_random_seed(1234)
        np.random.seed(1234)
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'cifar10', 'cifar-10-python.tar.gz')
        x_train, t_train, x_test, t_test = \
            dataset.load_cifar10(data_path, normalize=True, one_hot=True)
        x_train = x_train.reshape((-1, self.n_x))
        x_test = x_test.reshape((-1, self.n_x))
        learning_rate_ph = tf.placeholder(tf.float32, shape=[])
        # optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        optimizer = AdamaxOptimizer(learning_rate_ph)
        with pt.defaults_scope(phase=pt.Phase.train):
            lower_bound, lower_bound_obj = self.advi_klmin(self.batch_size,
                                                           self.lb_samples)
        lower_bound = tf.reduce_mean(lower_bound)
        lower_bound_obj = tf.reduce_mean(lower_bound_obj)
        bits_per_dim = -lower_bound / self.n_x * 1. / np.log(2.)
        tf.scalar_summary("train_sum/bits_per_dim", bits_per_dim)
        tf.scalar_summary("train_sum/dec_log_stdv", self.x_logsd)
        grads_and_vars = optimizer.compute_gradients(-lower_bound_obj)
        merged = tf.merge_all_summaries()
        infer = optimizer.apply_gradients(grads_and_vars)

        with pt.defaults_scope(phase=pt.Phase.test):
            eval_lower_bound, eval_log_likelihood = self.lb_is_loglikelihood(
                self.test_batch_size, self.ll_samples)
            samples = self.generate_samples(self.test_batch_size)
        eval_lower_bound = tf.reduce_mean(eval_lower_bound)
        eval_bits_per_dim = -eval_lower_bound / self.n_x * 1. / np.log(2.)
        eval_log_likelihood = tf.reduce_mean(eval_log_likelihood)
        eval_bits_per_dim_ll = -eval_log_likelihood / self.n_x * 1. / np.log(2.)
        test_fetch = [eval_lower_bound, eval_bits_per_dim,
                      eval_log_likelihood, eval_bits_per_dim_ll]

        total_size = 0
        params = tf.trainable_variables()
        for i in params:
            total_size += np.prod([int(s) for s in i.get_shape()])
            print i.name, i.get_shape()
        print("Num trainable variables: %d" % total_size)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        iters = x_train.shape[0] // self.batch_size
        start_time = time.time()
        with tf.Session() as sess:
            train_writer = tf.train.SummaryWriter(config.save_dir + '/train',
                                                  sess.graph)
            sess.run(init)
            if config.model_file is not "":
                saver.restore(sess, config.model_file)
            for epoch in range(1, config.epochs + 1):
                time_epoch = -time.time()
                if epoch % config.anneal_lr_freq == 0:
                    config.learning_rate *= config.anneal_lr_rate
                np.random.shuffle(x_train)
                lbs = []
                bitss = []
                for t in range(iters):
                    x_batch = x_train[t * self.batch_size:(t + 1) *
                                      self.batch_size]

                    _, lb, bits, x_logsd, summary = sess.run(
                        [infer, lower_bound, bits_per_dim,
                         self.x_logsd, merged], feed_dict={
                            self.x: x_batch,
                            learning_rate_ph: config.learning_rate})
                    lbs.append(lb)
                    bitss.append(bits)
                    train_writer.add_summary(summary, (epoch - 1) * iters + t)

                    # if (epoch - 1) * iters + t < 10 or (
                    #         (epoch - 1) * iters + t) % 20 == 0:
                    #     print('Iteration {} ({:.1f}s): bits_per_dim = {} '
                    #           'log_std = {}'.format((epoch - 1) * iters + t,
                    #                                 time.time() - start_time,
                    #                                 bits,
                    #                                 x_logsd))
                    #     start_time = time.time()

                time_epoch += time.time()
                print('Epoch {} ({:.1f}s): Lower bound = {} bits = {}'.format(
                    epoch, time_epoch, np.mean(lbs), np.mean(bitss)))
                if config.save_freq is not 0 and epoch % config.save_freq is 0:
                    saver.save(sess,
                               config.save_dir + "/model_{0}.ckpt".format(epoch))
                if epoch % config.test_freq == 0:
                    self.test(x_test, sess, test_fetch)
                    imgs = sess.run(samples)
                    save_image_collections(imgs, scaleeach=True,
                                           filename=os.path.join(
                                               config.save_dir,
                                               'samples_{:03d}.png'.format(
                                                   epoch)))

if __name__ == "__main__":
    tf.flags.DEFINE_integer("epochs", 5000, "max epoch num")
    tf.flags.DEFINE_integer("learning_rate", 1e-3, "")
    tf.flags.DEFINE_integer("anneal_lr_freq", 200,
                            "learning rate decay every epochs")
    tf.flags.DEFINE_integer("anneal_lr_rate", 0.75, "learning rate decay rate")
    tf.flags.DEFINE_string("save_dir",
                           os.environ['MODEL_RESULT_PATH_AND_PREFIX'],
                           'path and prefix to save params')
    tf.flags.DEFINE_integer("save_freq", 10, 'save frequency of param file')
    tf.flags.DEFINE_integer("test_freq", 1, 'test frequency of param file')
    tf.flags.DEFINE_string("model_file", "",
                           "restoring model file")
    FLAGS = tf.flags.FLAGS
    bottle_neck_group = namedtuple(
        'bottle_neck_group',
        ['num_blocks', 'num_filters', 'map_size'])
    groups = [
        bottle_neck_group(2, 64, 16),
        bottle_neck_group(2, 64, 8),
        bottle_neck_group(2, 64, 4)
    ]
    model = M1(batch_size=16, test_batch_size=100, n_z=32, groups=groups,
               kl_min=0.1, ll_samples=10, lb_samples=1, image_size=32)
    model.train(FLAGS)
