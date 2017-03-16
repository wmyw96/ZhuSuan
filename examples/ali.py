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


@zs.reuse('decoder')
def decoder(observed, n, n_z=64, is_training=True):
    with zs.StochasticGraph(observed=observed) as decoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}

        z_mean = tf.zeros([n_z])
        z_logstd = tf.zeros([n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=0, n_samples=n)
        lx_z = tf.reshape(z, [-1, 1, 1, n_z])
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
        lx_z = layers.conv2d(lx_z, 3, 1, stride=1, activation_fn=tf.nn.sigmoid,
                             weights_initializer=
                             init_ops.RandomNormal(stddev=0.01))

    return decoder, lx_z, z


@zs.reuse('encoder')
def encoder(observed, x, n_z, is_training):
    with zs.StochasticGraph(observed=observed) as encoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = layers.conv2d(x, 32, 5, stride=1, activation_fn=lrelu,
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
        mu, logstd = lz_x[:, :, :, :n_z], lz_x[:, :, :, n_z:]
        lz_x = zs.Normal('z', mu, logstd)
    return encoder, lz_x


@zs.reuse('discriminator')
def discriminator(x, z, is_training):
    num_pieces = 2
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

    lc_xz = tf.concat([lc_x, lc_z], -1)
    lc_xz = max_out(layers.conv2d(
        layers.dropout(lc_xz, 0.5, is_training=is_training),
        1024, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_xz = max_out(layers.conv2d(
        layers.dropout(lc_xz, 0.5, is_training=is_training),
        1024, 1, activation_fn=None, weights_initializer=
        init_ops.RandomNormal(stddev=0.01)), num_pieces)
    lc_xz = layers.flatten(lc_xz)
    lc_xz = layers.dropout(lc_xz, 0.5, is_training=is_training)
    class_logits = layers.fully_connected(lc_xz, 1, activation_fn=None,
                                          weights_initializer=
                                          init_ops.RandomNormal(stddev=0.01))
    return class_logits


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join("/home/yucen/mfs/code/ZhuSuan/examples",
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    _, n_xl, _, n_channels = x_train.shape
    n_y = t_train.shape[1]

    # Define model parameters
    n_z = 64

    # Define training/evaluation parameters
    lb_samples = 1
    epoches = 5000
    batch_size = 100 * FLAGS.num_gpus
    gen_size = 100
    recon_size = 50
    iters = x_train.shape[0] // batch_size
    print_freq = 100
    test_freq = iters
    save_freq = iters
    learning_rate = 0.0001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    result_path = os.environ['MODEL_RESULT_PATH_AND_PREFIX']

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels),
                       name='x')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)


    def build_tower_graph(x, id_):
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
                    (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]
        _, x_gen, z_prior = decoder(None, n, n_z, is_training)
        _, z_gen = encoder(None, tower_x, n_z, is_training)

        classifier = lambda tmp_x, tmp_z: discriminator(tmp_x, tmp_z,
                                                        is_training)
        disc_loss, gen_loss = zs.ali(classifier,
                                     encoder={'x': tower_x, 'z': z_gen},
                                     decoder={'x': x_gen, 'z': z_prior})

        gen_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='decoder') + \
                       tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='encoder')
        disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='discriminator')

        disc_grads = optimizer.compute_gradients(disc_loss,
                                                 var_list=disc_var_list)
        gen_grads = optimizer.compute_gradients(gen_loss,
                                                var_list=gen_var_list)
        grads = disc_grads + gen_grads
        return grads, gen_loss, disc_loss


    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                grads, gen_loss, disc_loss = build_tower_graph(x, i)
                tower_losses.append([gen_loss, disc_loss])
                tower_grads.append(grads)
    gen_loss, disc_loss = multi_gpu.average_losses(tower_losses)
    grads = multi_gpu.average_gradients(tower_grads)
    infer = optimizer.apply_gradients(grads)
    # eval generation
    _, eval_x_gen, _ = decoder(None, gen_size, n_z, is_training)
    # eval reconstruction
    _, eval_z_gen = encoder(None, x, n_z, is_training)
    _, eval_x_recon, _ = decoder({'z': eval_z_gen},
                                 tf.shape(x)[0], n_z, is_training)
    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='decoder') + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='encoder') + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='discriminator')

    saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
    # Run the inference
    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, 'results/ali/ali.epoch.10.iter.1562.ckpt')
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            gen_losses, disc_losses = [], []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, g_loss, d_loss = sess.run(
                    [infer, gen_loss, disc_loss],
                    feed_dict={x: x_batch,
                               learning_rate_ph: learning_rate,
                               is_training: True})
                gen_losses.append(g_loss)
                disc_losses.append(d_loss)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Gen loss = {} Disc loss = {}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(gen_losses), np.mean(disc_losses)))
                    gen_losses = []
                    disc_losses = []

                if iter % test_freq == 0:
                    time_test = -time.time()
                    gen_images = sess.run(eval_x_gen,
                                          feed_dict={is_training: False})
                    name = "gen/ali.epoch.{}.iter.{}.png".format(epoch, iter)
                    name = os.path.join(result_path, name)
                    utils.save_image_collections(gen_images, name,
                                                 scale_each=True)

                    x_batch = x_test[:recon_size]
                    eval_zs, recon_images = \
                        sess.run([eval_z_gen.tensor, eval_x_recon],
                                 feed_dict={x: x_batch, is_training: False})
                    name = "recon/ali.epoch.{}.iter.{}.png".format(
                        epoch, iter)
                    name = os.path.join(
                        result_path, name)
                    utils.save_contrast_image_collections(x_batch, recon_images,
                                                          name, scale_each=True)
                    time_test += time.time()

                if iter % save_freq == 0:
                    save_path = "ali.epoch.{}.iter.{}.ckpt".format(epoch, iter)
                    save_path = os.path.join(result_path, save_path)
                    saver.save(sess, save_path)

                if iter % print_freq == 0:
                    time_train = -time.time()

