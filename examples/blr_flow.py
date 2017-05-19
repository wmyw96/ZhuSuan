#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm, colors


def house_hold_flow(z):
    #return z
    static_z_shape = z.get_shape()
    dynamic_z_shape = tf.shape(z)
    ndim = static_z_shape.ndims
    D = int(static_z_shape[ndim - 1])
    vt = tf.Variable(tf.random_normal(shape=(D, 1), mean=0, stddev=0.05))
    H = 2 * tf.matmul(vt, tf.transpose(vt)) / tf.matmul(tf.transpose(vt), vt)
    H = tf.eye(D) - H
    #H = tf.eye(D)
    # print(H.get_shape())
    z = tf.reshape(z, [-1, D])
    z = tf.matmul(z, H)
    z = tf.reshape(z, dynamic_z_shape)
    return z


def convert_to_int(x):
    """
    Try to convert input to type int in python.

    :param x: The input instance.
    :return: A int if succeed, else None.
    """
    if isinstance(x, int):
        return x
    return None


def planar_normalizing_flow(samples, log_probs, n_iters):
    """
    Perform Planar Normalizing Flow along the last axis of inputs.

    .. math ::

        f(z_t) = z_{t-1} + h(z_{t-1} * w_t + b_t) * u_t

    with activation function `tanh` as well as the invertibility trick
    from (Danilo 2016).

    :param samples: A N-D (N>=2) `float32` Tensor of shape `[..., d]`, and
        planar normalizing flow will be performed along the last axis.
    :param log_probs: A (N-1)-D `float32` Tensor, should be of the same shape
        as the first N-1 axes of `samples`.
    :param n_iters: A int, which represents the number of successive flows.

    :return: A N-D Tensor, the transformed samples.
    :return: A (N-1)-D Tensor, the log probabilities of the transformed
        samples.
    """
    if not isinstance(n_iters, int):
        raise ValueError('n_iters should be type \'int\'')

    # check shapes of samples and log_probs
    static_sample_shape = samples.get_shape()
    static_logprob_shape = log_probs.get_shape()
    static_sample_ndim = convert_to_int(static_sample_shape.ndims)
    static_logprob_ndim = convert_to_int(static_logprob_shape.ndims)
    if static_sample_ndim and static_sample_ndim <= 1:
        raise ValueError('samples should have rank >= 2')
    if static_sample_ndim and static_logprob_ndim \
            and static_sample_ndim != static_logprob_ndim + 1:
        raise ValueError('log_probs should have rank (N-1), while N is the '
                         'rank of samples')
    try:
        tf.broadcast_static_shape(static_sample_shape,
                                  static_logprob_shape)
    except ValueError:
        raise ValueError(
             "samples and log_probs don't have same shape of (N-1) dims,"
             "while N is the rank of samples")
    dynamic_sample_shape = tf.shape(samples)
    dynamic_logprob_shape = tf.shape(log_probs)
    dynamic_sample_ndim = tf.rank(samples)
    dynamic_logprob_ndim = tf.rank(log_probs)
    _assert_sample_ndim = \
        tf.assert_greater_equal(dynamic_sample_ndim, 2,
                                message='samples should have rank >= 2')
    with tf.control_dependencies([_assert_sample_ndim]):
        samples = tf.identity(samples)
    _assert_logprob_ndim = \
        tf.assert_equal(dynamic_logprob_ndim, dynamic_sample_ndim - 1,
                        message='log_probs should have rank (N-1), while N is'
                                ' the rank of samples')
    with tf.control_dependencies([_assert_logprob_ndim]):
        log_probs = tf.identity(log_probs)
    _assert_same_shape = \
        tf.assert_equal(dynamic_sample_shape[:-1], dynamic_logprob_shape,
                        message="samples and log_probs don't have same shape "
                                "of (N-1) dims,while N is the rank of samples")
    with tf.control_dependencies([_assert_same_shape]):
        samples = tf.identity(samples)
        log_probs = tf.identity(log_probs)

    input_x = tf.convert_to_tensor(samples, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    static_x_shape = input_x.get_shape()
    if not static_x_shape[-1:].is_fully_defined():
        raise ValueError(
            'Inputs {} has undefined last dimension.'.format(input_x.name))
    d = int(static_x_shape[-1])

    # define parameters
    with tf.name_scope('planar_flow_parameters'):
        param_bs, param_us, param_ws = [], [], []
        for iter in range(n_iters):
            param_b = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32),
                                  name='param_b_%d' % iter)
            aux_u = tf.Variable(
                tf.random_normal(shape=[d, 1], mean=0, stddev=0.005,
                                 dtype=tf.float32),
                name='aux_u_%d' % iter)
            param_w = tf.Variable(
                tf.random_normal(shape=[d, 1], mean=0, stddev=0.005,
                                 dtype=tf.float32),
                name='para_w_%d' % iter)
            dot_prod = tf.matmul(param_w, aux_u, transpose_a=True)
            param_u = aux_u + param_w / tf.matmul(param_w, param_w,
                                                  transpose_a=True) \
                * (tf.log(tf.exp(dot_prod) + 1) - 1 - dot_prod)
            param_u = tf.transpose(param_u, name='param_u_%d' % iter)
            param_bs.append(param_b)
            param_ws.append(param_w)
            param_us.append(param_u)

    # forward and log_det_jacobian
    z = tf.reshape(input_x, [-1, d])
    for iter in range(n_iters):
        scalar = tf.matmul(param_us[iter], param_ws[iter], name='scalar')
        scalar = tf.reshape(scalar, [])

        # check invertible
        invertible_check = tf.assert_greater_equal(
            scalar, tf.constant(-1.0 - 1e-3, dtype=tf.float32),
            message="w'u must be greater or equal to -1")
        with tf.control_dependencies([invertible_check]):
            scalar = tf.identity(scalar)

        param_w = param_ws[iter]
        activation = tf.tanh(
            tf.matmul(z, param_w, name='score') + param_bs[iter],
            name='activation')
        param_u = param_us[iter]

        reduce_act = tf.reduce_sum(activation, axis=-1)
        det_ja = scalar * (
            tf.constant(1.0, dtype=tf.float32) - reduce_act * reduce_act) \
            + tf.constant(1.0, dtype=tf.float32)
        log_probs -= tf.log(det_ja)
        z = z + tf.matmul(activation, param_u, name='update')
    z = tf.reshape(z, tf.shape(input_x))

    return z, log_probs


def random_weights(n_in, n_out):
    return tf.random_normal(shape=(n_in, n_out), mean=0,
                            stddev=np.sqrt(2 / n_in), dtype=tf.float32)


def random_bias(n_out):
    #return tf.constant([0] * n_out, dtype=tf.float32)
    return tf.random_normal(shape=(n_out,), mean=0, stddev=0.05, dtype=tf.float32)


def get_linear_mask(input_pri, output_pri, hidden):
    layers = [len(input_pri)] + hidden + [len(output_pri)]
    max_pri = max(input_pri)
    print((hidden[0] / len(input_pri)))
    priority = [input_pri]
    for units in hidden:
        priority += [[j for j in range(max_pri)
                      for k in range(int(units / len(input_pri)))]]
        print([j for j in range(max_pri)
                      for k in range(int(units / len(input_pri)))])
    priority += [output_pri]
    mask = []
    for l in range(len(layers) - 1):
        # z_{j} = z_{i} * W_{ij}
        maskl = np.zeros((layers[l], layers[l + 1]))
        for i in range(layers[l]):
            for j in range(layers[l + 1]):
                maskl[i][j] = (priority[l][i] <= priority[l + 1][j]) * 1.0
        mask.append(maskl)
    print(mask)
    return mask


def get_diagonal_mask(D):
    mask = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            mask[i][j] = (i < j) * 1.0
    return mask


def made(name, id, z, hidden, multiple=5, hidden_layers=2):
    static_z_shape = z.get_shape()
    if not static_z_shape[-1:].is_fully_defined():
        raise ValueError('Inputs {} has undefined last dimension.'.format(z))
    d = int(static_z_shape[-1])

    units = multiple * d
    layer_unit = [d] + [units] * hidden_layers + [2 * d]
    mask = get_linear_mask([i + 1 for i in range(d)],
                           [i for i in range(d)] * 2, [units] * hidden_layers)

    masked_weights = []
    biases = []
    with tf.name_scope(name + '%d' % id):
        layer = tf.identity(z)
        layer = tf.reshape(layer, [-1, d])
        inputx = tf.identity(layer)
        for i in range(hidden_layers):
            w = tf.Variable(random_weights(layer_unit[i], layer_unit[i + 1]))
            u = tf.Variable(random_weights(layer_unit[i], layer_unit[i + 1]))
            print(mask[i])
            masked_weights.append(w)
            w = w * tf.constant(mask[i], dtype=tf.float32)
            u = u * tf.constant(mask[i], dtype=tf.float32)
            b = tf.Variable(random_bias(layer_unit[i + 1]))
            biases.append(b)
            constant = tf.ones(shape=tf.shape(layer))
            linear = tf.matmul(layer, w) + tf.matmul(constant, u)+ b
            layer = tf.nn.elu(linear, name='layer_%d' % (i + 1))

        constant = tf.ones(shape=tf.shape(layer))

        m_w = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        m_u = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        m_i = tf.Variable(random_weights(d, d))
        print(mask[hidden_layers][:, :d])
        m_w = m_w * tf.constant(mask[hidden_layers][:, :d], dtype=tf.float32)
        m_u = m_u * tf.constant(mask[hidden_layers][:, :d], dtype=tf.float32)
        masked_weights.append(m_w)
        m_b = tf.Variable(random_bias(d))
        biases.append(m_b)
        m = tf.matmul(layer, m_w) + tf.matmul(constant, m_u) + \
            tf.matmul(inputx, m_i * tf.constant(get_diagonal_mask(d),
                                                dtype=tf.float32)) + m_b

        s_w = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        s_u = tf.Variable(random_weights(layer_unit[hidden_layers], d))
        s_i = tf.Variable(random_weights(d, d))
        s_w = s_w * tf.constant(mask[hidden_layers][:, d:], dtype=tf.float32)
        s_u = s_u * tf.constant(mask[hidden_layers][:, d:], dtype=tf.float32)
        print(mask[hidden_layers][:, d:])
        masked_weights.append(s_w)
        s_b = tf.Variable(random_bias(d))
        biases.append(s_b)
        s = tf.matmul(layer, s_w) + tf.matmul(constant, s_u) + \
            tf.matmul(inputx, s_i * tf.constant(get_diagonal_mask(d),
                                                dtype=tf.float32)) + s_b

    m = tf.reshape(m, tf.shape(z))
    s = tf.reshape(s, tf.shape(z))

    return m, s, masked_weights, biases


def inv_autoregressive_flow(samples, hidden, log_probs, autoregressive_nn,
                            n_iters, update='normal'):
    joint_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    z = tf.convert_to_tensor(samples, dtype=tf.float32)

    for iter in range(n_iters):
        m, s, masked_weights, biases = autoregressive_nn('iaf', iter, z, hidden)

        if update == 'gru':
            sigma = tf.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            joint_probs = joint_probs - tf.reduce_sum(tf.log(sigma), axis=-1)

        if update == 'normal':
            z = z * s + m
            #z = z + m
            joint_probs = joint_probs - tf.reduce_sum(tf.log(abs(s)), axis=-1)

        z = tf.reverse(z, [-1])

    return z, joint_probs, s, masked_weights, biases


@zs.reuse('model')
def blr(observed, D, x, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        # w: [n_particles, D]
        w = zs.Normal('w', tf.zeros([D]), tf.ones([D]), n_samples=n_particles,
                      group_event_ndims=1)
        # x: [N, D]
        pred = tf.matmul(w, tf.transpose(x))
        # y: [n_particles, N]
        y = zs.Bernoulli('y', pred, dtype=tf.float32)
    return model


@zs.reuse('variational')
def q_net(observed):
    with zs.BayesianNet(observed=observed) as variational:
        w_mean = tf.get_variable('w_mean', shape=[D], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.))
        w_logstd = tf.get_variable('w_logstd', shape=[D], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))
        # w: [n_particles, D]
        w = zs.Normal('w', w_mean, w_logstd, n_samples=n_particles,
                      group_event_ndims=1)
    return variational


def kde(samples, points, kernel_stdev):
    # samples: [n D]
    # points: [m D]
    # return [m]
    samples = tf.expand_dims(samples, 1)
    points = tf.expand_dims(points, 0)
    # [n m D]
    Z = np.sqrt(2 * np.pi) * kernel_stdev
    log_probs = -np.log(Z) + (-0.5 / kernel_stdev ** 2) * (
        samples - points) ** 2
    log_probs = tf.reduce_sum(log_probs, -1)
    log_probs = zs.log_mean_exp(log_probs, 0)

    return log_probs


def compute_tickers(probs, n_bins=20):
    # Sort
    flat_probs = list(probs.flatten())
    flat_probs.sort()
    flat_probs.reverse()

    num_intervals = n_bins * (n_bins - 1) // 2
    interval_size = len(flat_probs) // num_intervals

    tickers = []
    cnt = 0
    for i in range(n_bins - 1):
        tickers.append(flat_probs[cnt])
        cnt += interval_size * (i + 1)
    tickers.append(flat_probs[-1])
    tickers.reverse()
    return tickers


def contourf(x, y, z):
    tickers = compute_tickers(z)
    palette = cm.PuBu
    plt.contourf(x, y, z, tickers,
                 cm=palette,
                 norm=colors.BoundaryNorm(tickers, ncolors=palette.N))
    plt.colorbar()


def plot_samples(w_samples, n, m, id, iter):
    # Plot the variational posterior with flow
    samples = sess.run(w_samples,
                       feed_dict={n_particles: n_qw_samples})
    ax = plt.subplot(n, m, id)
    ax.scatter(samples[:, 0], samples[:, 1], s=0.1)
    ax.set_xlim(lower_box, upper_box)
    ax.set_ylim(lower_box, upper_box)
    plt.title('Posterior samples in iter %d' % iter)


if __name__ == "__main__":
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Define model parameters
    N = 200
    D = 2
    learning_rate = 1
    learning_rate_g = 0.01
    learning_rate_d = 0.003
    t0 = 100
    t0_d = 100
    t0_g = 100
    epoches = 100
    epoches_d = 10
    epoches_d0 = 1000
    epoches_g = 500
    disc_n_samples = 1000
    gen_n_samples = 1000
    lower_box = -5
    upper_box = 5
    kde_batch_size = 2000
    n_qw_samples = 10000
    kde_stdev = 0.05
    plot_interval = 100

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[N, D], name='x')
    y = tf.placeholder(tf.float32, shape=[N], name='y')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    # y_rep = tf.tile(tf.expand_dims(y, axis=1), [1, n_particles])
    # [n_particles, N]
    y_obs = tf.tile(tf.expand_dims(y, axis=0), [n_particles, 1])

    # Generate synthetic data
    model = blr({}, D, x, n_particles)
    pw_outputs, py_outputs = model.query(['w', 'y'], outputs=True,
                                         local_log_prob=True)
    pw_samples, log_pw = pw_outputs
    py_samples = py_outputs[0]

    # Variational inference
    def log_joint(observed):
        model = blr(observed, D, x, n_particles)
        # log_pw: [n_particles]; log_py_w: [n_particles, N]
        log_pw, log_py_w = model.local_log_prob(['w', 'y'])
        # [n_particles]
        return log_pw + tf.reduce_sum(log_py_w, 1), log_pw, \
            tf.reduce_sum(log_py_w, 1)

    # MFVI
    variational = q_net({})
    # [n_particles, D], [n_particles]
    vw_samples, log_qw = variational.query('w', outputs=True,
                                           local_log_prob=True)
    #lower_bound = tf.reduce_mean(
    #    log_joint({'w': vw_samples, 'y': y_obs}) - log_qw)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    #v_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                 scope="variational")
    #infer = optimizer.minimize(-lower_bound, var_list=v_parameters)

    # VI with Flow
    fw_samples, log_fw, flow_s, masked_weights, biases = inv_autoregressive_flow(vw_samples,
                                                   None, log_qw,
                                                   made, 1, update='normal')
    #fw_samples = vw_samples
    #log_fw = log_qw
    #fw_samples, log_fw = planar_normalizing_flow(vw_samples, log_qw, 80)
    #fw_samples = house_hold_flow(vw_samples)
    #for iter in range(20):
    #    fw_samples = house_hold_flow(fw_samples)
    #fw_samples = vw_samples
    #log_fw = log_qw
    dif = tf.reduce_mean((vw_samples - fw_samples) * (vw_samples - fw_samples))

    flow_lower_bound = tf.reduce_mean(
        log_joint({'w': fw_samples, 'y': y_obs})[0] - log_fw)

    flow_optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    flow_infer = optimizer.minimize(-flow_lower_bound)

    # Plotting
    w_ph = tf.placeholder(tf.float32, shape=[None, D], name='w_ph')
    log_joint_value, log_prior, _ = log_joint({'w': w_ph, 'y': y_obs})
    log_mean_field = q_net({'w': w_ph}).local_log_prob('w')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Generate data
        train_x = np.random.rand(N, D) * (upper_box - lower_box) + lower_box
        train_w_sample, train_y = sess.run(
            [pw_samples, py_samples], feed_dict={x: train_x, n_particles: 1})
        print(train_w_sample.shape)
        # train_w_sample: [D]
        train_w_sample = np.squeeze(train_w_sample)
        print(train_w_sample.shape)
        print("decision boundary: {}x1 + {}x2 = 0".format(train_w_sample[0],
                                                          train_w_sample[1]))
        # train_y: [N]
        train_y = np.squeeze(train_y)
        '''
        # Run the mean-field variational inference
        for epoch in range(1, epoches + 1):
            lr = learning_rate * t0 / (t0 + epoch - 1)
            time_epoch = -time.time()
            _, lb = sess.run([infer, lower_bound],
                             feed_dict={x: train_x,
                                        y: train_y,
                                        learning_rate_ph: lr,
                                        n_particles: 100})
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, lb))
        '''
        # Draw the decision boundary
        def draw_decision_boundary(x, w, y):
            positive_x = x[y == 1, :]
            negative_x = x[y == 0, :]

            x0_points = np.linspace(lower_box, upper_box, num=100)
            x1_points = np.linspace(lower_box, upper_box, num=100)
            grid_x, grid_y = np.meshgrid(x0_points, x1_points)
            points = np.vstack((np.ravel(grid_x), np.ravel(grid_y))).T
            y_pred = 1.0 / (1 + np.exp(-np.sum(points * w, axis=1)))
            grid_pred = np.reshape(y_pred, grid_x.shape)

            plt.pcolormesh(grid_x, grid_y, grid_pred)
            plt.colorbar()
            CS = plt.contour(grid_x, grid_y, grid_pred, colors='k',
                                 levels=np.array([0.25, 0.5, 0.75]))
            plt.clabel(CS)
            plt.plot(positive_x[:, 0], positive_x[:, 1], 'x')
            plt.plot(negative_x[:, 0], negative_x[:, 1], 'o')

        plt.subplot(3, 4, 1)
        draw_decision_boundary(train_x, train_w_sample, train_y)
        #plt.scatter(train_w_sample[:, 0], train_w_sample[:, 1], s=0.1)
        #plt.set_xlim(lower_box, upper_box)
        #plt.set_ylim(lower_box, upper_box)
        #plt.title('w_samples for train')
        # draw_decision_boundary(train_x, train_w_sample, train_y)
        # plt.title('Decision boundary')

        # Plot unnormalized true posterior
        # Generate a w grid
        w0 = np.linspace(lower_box, upper_box, 100)
        w1 = np.linspace(lower_box, upper_box, 100)
        w0_grid, w1_grid = np.meshgrid(w0, w1)
        w_points = np.vstack((np.ravel(w0_grid), np.ravel(w1_grid))).T
        # [n_particles]
        log_joint_points = sess.run(
            log_joint_value,
            feed_dict={x: train_x,
                       y: train_y,
                        n_particles: w_points.shape[0],
                       w_ph: w_points})
        log_joint_grid = np.reshape(log_joint_points, w0_grid.shape)

        plt.subplot(3, 4, 2)
        contourf(w0_grid, w1_grid, log_joint_grid)
        plt.plot(train_w_sample[0], train_w_sample[1], 'x')
        plt.title('Unnormalized true posterior')
        '''
        # Plot the variational posterior
        log_v_points = sess.run(
            log_mean_field,
            feed_dict={x: train_x,
                       y: train_y,
                       n_particles: w_points.shape[0],
                       w_ph: w_points})
        log_v_grid = np.reshape(log_v_points, w0_grid.shape)

        plt.subplot(2, 2, 3)
        contourf(w0_grid, w1_grid, log_v_grid)
        plt.title('Mean field posterior')
        '''
        # Run the mean-field variational inference with flow
        for epoch in range(1, 1001):
            lr = learning_rate * t0 * 0.1 / (t0 + epoch - 1)
            time_epoch = -time.time()
            _, lb, d = sess.run([flow_infer, flow_lower_bound, dif],
                                feed_dict={x: train_x,
                                        y: train_y,
                                        learning_rate_ph: lr,
                                        n_particles: 100})
            time_epoch += time.time()
            #print(s)
            print('Epoch {} ({:.1f}s): Lower bound = {}, variation = {}'.format(
                epoch, time_epoch, lb, d))
            #break
            if epoch % 100 == 0:
                plot_samples(fw_samples, 3, 4, epoch / 100 + 2, epoch)
            '''if epoch % 100 == 0:
                mws = sess.run(masked_weights, feed_dict={x:train_x, y:train_y,
                                                          n_particles: 100})
                bs = sess.run(biases, feed_dict={x:train_x, y:train_y,
                                                          n_particles: 100})
                print(mws)
                print(bs)'''



        plt.show()
