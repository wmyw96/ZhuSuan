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
from tensorflow.python.ops import init_ops
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dataset
import utils
import multi_gpu
from multi_gpu import FLAGS
import pdb


def lrelu(input_tensor, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_tensor + f2 * abs(input_tensor)


def max_out(input_tensor, num_pieces=2):
    shape = input_tensor.get_shape()
    output = tf.reshape(input_tensor, [-1, int(shape[1]), int(shape[2]),
                                       int(shape[3]) // num_pieces, num_pieces])
    output = tf.reduce_max(output, axis=-1)
    return output


def multi_gan(classifier, g_model, l_infer, u_infer):
    """
    Implementation of multi-class gan with (x, y, z) input.
    :param classifier: A function that accepts three tensors and return
        unnormalized logits for the multi-class
    :param g_model:
    :param l_infer: 
    :param u_infer: 
    :return: 
    """
    g_model_class_logits = classifier(g_model['x'], g_model['y'], g_model['z'])
    l_infer_class_logits = classifier(l_infer['x'], l_infer['y'], l_infer['z'])
    u_infer_class_logits = classifier(u_infer['x'], u_infer['y'], u_infer['z'])
    qy_u = u_infer['qy']

    true_g = tf.zeros([tf.shape(g_model_class_logits)[0]], tf.int32)
    true_l = tf.ones([tf.shape(l_infer_class_logits)[0]], tf.int32)
    true_u = 2*tf.ones([tf.shape(u_infer_class_logits)[0]], tf.int32)

    # o_0' = - log (exp(o_1 -o_0) + exp(o_2 - o_0))
    o_0, o_1, o_2 = tf.split(g_model_class_logits, 3, axis=1)
    fake_g_logits = -tf.reduce_logsumexp(tf.stack([o_1 - o_0, o_2 - o_0],
                                                  axis=1), axis=1)
    o_0, o_1, o_2 = tf.split(l_infer_class_logits, 3, axis=1)
    fake_l_logits = -tf.reduce_logsumexp(tf.stack([o_0 - o_1, o_2 - o_1],
                                                  axis=1), axis=1)
    o_0, o_1, o_2 = tf.split(u_infer_class_logits, 3, axis=1)
    fake_u_logits = -tf.reduce_logsumexp(tf.stack([o_0 - o_2, o_1 - o_2],
                                                  axis=1), axis=1)

    disc_losses_list = [
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_g,
            logits=g_model_class_logits)),
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_l,
            logits=l_infer_class_logits)),
        tf.reduce_mean(tf.reshape(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=true_u,
                logits=u_infer_class_logits),
            (-1, tf.shape(qy_u)[1])) * qy_u)
    ]
    disc_loss = sum(disc_losses_list)

    gen_losses_list = [
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_g_logits),
            logits=fake_g_logits)),
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_l_logits),
            logits=fake_l_logits)),
        tf.reduce_mean(tf.reshape(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_u_logits),
                logits=fake_u_logits),
            (-1, tf.shape(qy_u)[1])) * qy_u)
    ]
    gen_loss = sum(gen_losses_list)
    return disc_loss, gen_loss, disc_losses_list, gen_losses_list


@zs.reuse('px_yz')
def px_yz(observed, n, n_x, n_y, n_z, is_training):
    with zs.StochasticGraph(observed=observed) as px_yz:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n_z])
        z_logstd = tf.zeros([n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=0, n_samples=n)
        y_logits = tf.zeros([n_y])
        y = zs.Discrete('y', y_logits, sample_dim=0, n_samples=n)
        lx_z = tf.reshape(z, [-1, 1, 1, n_z])
        lx_y = tf.reshape(y, [-1, 1, 1, n_y])
        lx_z = tf.concat([lx_z, lx_y], -1)
        lx_z = layers.conv2d_transpose(lx_z, 256, 4, 1, activation_fn=lrelu,
                                       padding='VALID', weights_initializer=
                                       init_ops.RandomNormal(stddev=0.01),
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)

        lx_z = layers.conv2d_transpose(lx_z, 128, 4, stride=2,
                                       padding='VALID', weights_initializer=
                                       init_ops.RandomNormal(stddev=0.01),
                                       activation_fn=lrelu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 64, 4, stride=1,
                                       padding='VALID', weights_initializer=
                                       init_ops.RandomNormal(stddev=0.01),
                                       activation_fn=lrelu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 32, 4, stride=2,
                                       padding='VALID', weights_initializer=
                                       init_ops.RandomNormal(stddev=0.01),
                                       activation_fn=lrelu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 32, 5, stride=1,
                                       padding='VALID', weights_initializer=
                                       init_ops.RandomNormal(stddev=0.01),
                                       activation_fn=lrelu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d(lx_z, 32, 1, stride=1, activation_fn=lrelu,
                             weights_initializer=
                             init_ops.RandomNormal(stddev=0.01),
                             padding='VALID', normalizer_fn=layers.batch_norm,
                             normalizer_params=normalizer_params)
        x = layers.conv2d(lx_z, 3, 1, stride=1, activation_fn=tf.nn.sigmoid,
                          weights_initializer=
                          init_ops.RandomNormal(stddev=0.01))
    return x, y, z


@zs.reuse('qz_xy')
def qz_xy(x, y, n_z, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    x = tf.reshape(x, [-1, 32, 32, 3])
    y = tf.reshape(y, [-1, 1, 1, 10])
    y = tf.tile(y, [1, 32, 32, 1])
    lz_x = layers.conv2d(tf.concat([x, y], -1), 32, 5, stride=1, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 64, 4, stride=2, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 128, 4, stride=1, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 256, 4, stride=2, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 512, 4, stride=1, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 512, 1, stride=1, activation_fn=lrelu,
                         padding='VALID', normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    lz_x = layers.conv2d(lz_x, 128, 1, stride=1, activation_fn=None,
                         weights_initializer=
                         init_ops.RandomNormal(stddev=0.01))
    z_mean, z_logstd = lz_x[:, :, :, :n_z], lz_x[:, :, :, n_z:]
    return z_mean, z_logstd


@zs.reuse('qy_x')
def qy_x(x, n_y, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    ly_x = tf.reshape(x, [-1, 32, 32, 3])
    ly_x = layers.dropout(ly_x, keep_prob=0.8, is_training=is_training)
    ly_x = layers.conv2d(ly_x, 128, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 128, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 128, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.max_pool2d(ly_x, kernel_size=2)
    ly_x = layers.dropout(ly_x, keep_prob=0.5, is_training=is_training)
    ly_x = layers.conv2d(ly_x, 256, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 256, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 256, 3, normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.max_pool2d(ly_x, kernel_size=2)
    ly_x = layers.dropout(ly_x, keep_prob=0.5, is_training=is_training)
    ly_x = layers.conv2d(ly_x, 512, 3, padding='VALID',
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 256, 1, padding='VALID',
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = layers.conv2d(ly_x, 128, 1, padding='VALID',
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    ly_x = tf.reduce_mean(ly_x, axis=[1, 2])
    y_logits = layers.fully_connected(ly_x, n_y, activation_fn=None)
    return y_logits


def labeled(x, y, n_z, is_training):
    with zs.StochasticGraph() as labeled:
        z_mean, z_logstd = qz_xy(x, y, n_z, is_training)
        z = zs.Normal('z', z_mean, z_logstd, reparameterized=True)
    return z


def unlabeled(x, n_x, n_y, n_z, is_training):
    with zs.StochasticGraph() as unlabeled:
        y_logits = qy_x(x, n_y, is_training)
        qy = tf.nn.softmax(y_logits)
        y_diag = tf.diag(tf.ones(n_y))
        y = tf.reshape(
            tf.tile(tf.expand_dims(y_diag, 0), [tf.shape(x)[0], 1, 1]),
            [-1, n_y])
        x = tf.reshape(tf.tile(tf.expand_dims(x, 1), [1, n_y, 1]), [-1, n_x])
        z_mean, z_logstd = qz_xy(x, y, n_z, is_training)
        z = zs.Normal('z', z_mean, z_logstd, reparameterized=True)
    return x, y, z, qy, y_logits


def unlabeled_test(x, n_y, n_z, is_training):
    with zs.StochasticGraph() as unlabeled_test:
        y_logits = qy_x(x, n_y, is_training)
        y = tf.arg_max(y_logits, dimension=1)
        y_onehot = tf.one_hot(y, n_y)
        z_mean, z_logstd = qz_xy(x, y_onehot, n_z, is_training)
        z = zs.Normal('z', z_mean, z_logstd, reparameterized=True)
    return y, y_onehot, z


@zs.reuse('discriminator')
def discriminator(x, y, z, n_y, n_z, is_training):
    num_pieces = 2
    x = tf.reshape(x, [-1, 32, 32, 3])
    lc_x = max_out(layers.conv2d(
        layers.dropout(x, 0.8, is_training=is_training),
        32, 5, padding='VALID', activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_x = max_out(layers.conv2d(
        layers.dropout(lc_x, 0.5, is_training=is_training),
        64, 4, stride=2, padding='VALID', activation_fn=None,
        weights_initializer=init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_x = max_out(layers.conv2d(
        layers.dropout(lc_x, 0.5, is_training=is_training),
        128, 4, padding='VALID', activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_x = max_out(layers.conv2d(
        layers.dropout(lc_x, 0.5, is_training=is_training),
        256, 4, stride=2, padding='VALID', activation_fn=None,
        weights_initializer=init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_x = max_out(layers.conv2d(
        layers.dropout(lc_x, 0.5, is_training=is_training),
        512, 4, padding='VALID', activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)

    z = tf.reshape(z, [-1, 1, 1, n_z])
    lc_z = max_out(layers.conv2d(
        layers.dropout(z, 0.8, is_training=is_training),
        512, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_z = max_out(layers.conv2d(
        layers.dropout(lc_z, 0.5, is_training=is_training),
        512, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)

    y = tf.reshape(y, [-1, 1, 1, n_y])
    lc_y = max_out(layers.conv2d(
        layers.dropout(y, 0.8, is_training=is_training),
        512, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_y = max_out(layers.conv2d(
        layers.dropout(lc_y, 0.5, is_training=is_training),
        512, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)

    lc_xyz = tf.concat([lc_x, lc_z, lc_y], -1)
    lc_xyz = max_out(layers.conv2d(
        layers.dropout(lc_xyz, 0.5, is_training=is_training),
        1024, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_xyz = max_out(layers.conv2d(
        layers.dropout(lc_xyz, 0.5, is_training=is_training),
        1024, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)

    lc_xyz = layers.flatten(lc_xyz)
    lc_xyz = layers.dropout(lc_xyz, 0.5, is_training=is_training)
    class_logits = layers.fully_connected(lc_xyz, 3, activation_fn=None,
                                          weights_initializer=
                                          init_ops.RandomNormal(stddev=0.01))
    return class_logits


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR10
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_labeled, t_labeled, x_unlabeled, t_unlabeled, x_test, t_test = \
        dataset.load_cifar10_semi_supervised(data_path, normalize=True,
                                             one_hot=True)

    _, n_xl, _, n_channels = x_labeled.shape
    n_x = n_xl * n_xl * n_channels
    x_labeled, x_unlabeled, x_test = [
        x.reshape((-1, n_x)) for x in [x_labeled, x_unlabeled, x_test]]
    n_labeled = x_labeled.shape[0]
    n_y = t_labeled.shape[1]

    # Define model parameters
    n_z = 64

    # Define training/evaluation parameters
    epoches = 3000
    batch_size = 16
    test_batch_size = 100
    iters = x_unlabeled.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = iters
    save_freq = iters
    print_freq = 200
    learning_rate = 0.0001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    recon_size = 50
    beta = 1.

    result_path = os.environ['MODEL_RESULT_PATH_AND_PREFIX']

    y_gen = np.diag(np.ones(10))
    y_gen = np.tile(y_gen, [10, 1])

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    # Labeled
    x_labeled_ph = tf.placeholder(tf.float32, shape=[None, n_x], name='x_l')
    y_labeled_ph = tf.placeholder(tf.float32, shape=[None, n_y], name='y_l')
    x_unlabeled_ph = tf.placeholder(tf.float32, shape=(None, n_x), name='x_u')
    n = tf.shape(x_labeled_ph)[0]

    # Get the three groups of data
    x_gen, y_prior, z_prior = px_yz(None, n, n_x, n_y, n_z, is_training)
    z_l = labeled(x_labeled_ph, y_labeled_ph, n_z, is_training)
    x_u, y_u, z_u, qy_u, qy_logits_u = unlabeled(x_unlabeled_ph, n_x, n_y, n_z,
                                                 is_training)

    def classifier(tmp_x, tmp_y, tmp_z):
        return discriminator(tmp_x, tmp_y, tmp_z, n_y, n_z, is_training)

    disc_loss, gen_loss, d_loss_list, g_loss_list = multi_gan(
        classifier,
        g_model={'x': x_gen, 'y': y_prior, 'z': z_prior},
        l_infer={'x': x_labeled_ph, 'y': y_labeled_ph, 'z': z_l},
        u_infer={'x': x_u, 'y': y_u, 'z': z_u, 'qy': qy_u})

    # q(y|x) through labeled data, classifier loss and accuracy
    qy_logits_l = qy_x(x_labeled_ph, n_y, is_training)
    qy_l = tf.nn.softmax(qy_logits_l)
    pred_y = tf.argmax(qy_l, 1)
    acc = tf.reduce_mean(
        tf.cast(tf.equal(pred_y, tf.argmax(y_labeled_ph, 1)), tf.float32))
    log_qy_x = zs.discrete.logpmf(y_labeled_ph, qy_logits_l)
    log_qy_x = tf.reduce_mean(log_qy_x)
    cls_loss = -beta * log_qy_x

    px_yz_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='px_yz')
    qz_xy_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='qz_xy')
    qy_x_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='qy_x')
    disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='discriminator')
    gen_var_list = px_yz_var_list + qz_xy_var_list + qy_x_var_list

    disc_grads = optimizer.compute_gradients(disc_loss,
                                             var_list=disc_var_list)
    gen_grads = optimizer.compute_gradients(gen_loss,
                                            var_list=gen_var_list)
    cls_grads = optimizer.compute_gradients(cls_loss, var_list=qy_x_var_list)

    # Get losses and gradients
    grads = disc_grads + gen_grads + cls_grads
    infer = optimizer.apply_gradients(grads)
    each_loss = d_loss_list + g_loss_list

    def check_grads(loss, var_list):
        grads_to_be_check = optimizer.compute_gradients(loss, var_list=var_list)
        grads_to_be_check_ = [g for g, v in grads_to_be_check if g is not None]
        grads_to_be_check_mean = tf.reduce_mean(
            tf.stack([tf.reduce_mean(tf.square(g)) for g in grads_to_be_check_]))
        grads_to_be_check_var = tf.reduce_mean(
            tf.stack([tf.nn.moments(g, list(range(0, g.get_shape().ndims)))[1]
                      for g in grads_to_be_check_]))
        return grads_to_be_check_mean, grads_to_be_check_var


    check_qyx_grads = check_grads(gen_loss, var_list=qy_x_var_list)
    check_qyx_grads_u = check_grads(g_loss_list[2], var_list=qy_x_var_list)
    check_cls_grads = check_grads(cls_loss, var_list=qy_x_var_list)
    check_px_grads = check_grads(gen_loss, var_list=px_yz_var_list)
    check_qzxy_grads_l = check_grads(g_loss_list[1], var_list=qz_xy_var_list)
    check_qzxy_grads_u = check_grads(g_loss_list[2], var_list=qz_xy_var_list)
    check_qzxy_grads_g = check_grads(gen_loss, var_list=qz_xy_var_list)

    check_list = []
    for i in [check_qyx_grads, check_qyx_grads_u, check_cls_grads,
              check_px_grads, check_qzxy_grads_l, check_qzxy_grads_u,
              check_qzxy_grads_g]:
        check_list += i

    # eval generation
    eval_x_gen, _, _ = px_yz({'y': y_labeled_ph},
                             2 * recon_size, n_x, n_y, n_z, is_training)
    # eval reconstruction, unlabeled data
    eval_y, eval_y_gen_onehot, eval_z_gen = unlabeled_test(
        x_unlabeled_ph, n_y, n_z, is_training)
    eval_x_recon, _, _ = px_yz({'y': eval_y_gen_onehot, 'z': eval_z_gen},
                               tf.shape(x_unlabeled_ph)[0], n_x, n_y, n_z,
                               is_training)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate

            indices = np.random.permutation(x_unlabeled.shape[0])
            x_unlabeled = x_unlabeled[indices]
            t_unlabeled = t_unlabeled[indices]
            fetches = []
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

                ft = sess.run([infer, gen_loss, disc_loss, cls_loss, acc]
                              + check_list + each_loss,
                              feed_dict={x_labeled_ph: x_labeled_batch,
                                         y_labeled_ph: y_labeled_batch,
                                         x_unlabeled_ph: x_unlabeled_batch,
                                         learning_rate_ph: learning_rate,
                                         is_training: True})
                fetches.append(ft[1:])

                if iter % print_freq == 0:
                    pt_fetches = np.mean(fetches, axis=0)
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Gen loss = {:.3f} Disc loss = {:.3f} '
                          'Cls loss = {:.3f} Acc = {:.2f}\n'
                          'Check qy_x Grads, Mean = {}, Var = {}\n'
                          'Check qy_x_u Grads, Mean = {}, Var = {}\n'
                          'Check cls Grads, Mean = {}, Var = {}\n'
                          'Check p_x Grads, Mean = {}, Var = {}\n'
                          'Check qzxy_grads_l Grads, Mean = {}, Var = {}\n'
                          'Check qzxy_grads_u Grads, Mean = {}, Var = {}\n'
                          'Check qzxy_grads_g Grads, Mean = {}, Var = {}\n'
                          'd_g = {:.3f} d_l = {:.3f} d_u = {:.3f}\n'
                          'g_g = {:.3f} g_l = {:.3f} g_u = {:.3f}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 *pt_fetches))
                    fetches = []

                if iter % test_freq == 0:
                    time_test = -time.time()
                    gen_images = sess.run(eval_x_gen,
                                          feed_dict={
                                              y_labeled_ph:y_gen,
                                              is_training: False})
                    gen_images = np.reshape(gen_images, [-1, n_xl, n_xl,
                                                         n_channels])
                    name = "gen/epoch.{}.iter.{}.png".format(epoch, iter)
                    name = os.path.join(result_path, name)
                    utils.save_image_collections(gen_images, name,
                                                 scale_each=True)
                    test_accs = []
                    for tt in range(test_iters):
                        test_x_batch = x_test[
                                   tt*test_batch_size: (tt + 1)*test_batch_size]
                        test_y_batch = t_test[
                                   tt*test_batch_size: (tt + 1)*test_batch_size]
                        test_acc = sess.run(acc,
                                            feed_dict={
                                                x_labeled_ph: test_x_batch,
                                                y_labeled_ph: test_y_batch,
                                                x_unlabeled_ph: test_x_batch,
                                                is_training: False})
                        test_accs.append(test_acc)
                    x_batch = x_test[:recon_size]
                    eval_zs, recon_images, eval_ys = \
                        sess.run([eval_z_gen.tensor, eval_x_recon,
                                  eval_y],
                                 feed_dict={x_labeled_ph: x_batch,
                                            x_unlabeled_ph: x_batch,
                                            is_training: False})
                    name = "recon/epoch.{}.iter.{}.png".format(
                        epoch, iter)
                    name = os.path.join(result_path, name)
                    x_batch = np.reshape(x_batch, [-1, n_xl, n_xl, n_channels])
                    recon_images = np.reshape(recon_images, [-1, n_xl, n_xl,
                                                             n_channels])
                    utils.save_contrast_image_collections(x_batch,
                                                          recon_images, name,
                                                          scale_each=True)
                    time_test += time.time()
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test accuracy: {:.2f}%'.format(
                        100. * np.mean(test_accs)))
                    print('Reconst labels {}'.format(eval_ys))

                if iter % print_freq == 0:
                    time_train = -time.time()
