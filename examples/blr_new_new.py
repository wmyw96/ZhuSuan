#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import math
import os
import sys

import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    # from zhusuan.distributions_old import norm, bernoulli
    # from zhusuan.distributions import norm, bernoulli
    from zhusuan.mcmc.hmc2 import HMC
    from zhusuan.diagnostics import ESS
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


tf.set_random_seed(0)

# Parameters
chain_length = 50
burnin = 25

# Load MNIST dataset
n = 600
n_dims = 784
mu = 0
sigma = 1. / math.sqrt(n)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'mnist.pkl.gz')
X_train, y_train, _, _, X_test, y_test = \
    dataset.load_binary_mnist_realval(data_path)
X_train = X_train[:n] * 256
y_train = y_train[:n]
X_test = X_test * 256


def blrf(X_train, y_train, beta, sigma):
    D = len(beta)
    l = D * (-0.5 * np.log(2*np.pi) - np.log(sigma))
    l -= 0.5/sigma**2 * np.sum(np.square(beta))
    a = np.sum(X_train * beta, 1)
    a[a < -30] = -30
    a[a > 30] = 30
    h = 1 / (1 + np.exp(-a))
    # print('Prior = {}'.format(l))
    # print('Likelihood = {}'.format(np.sum(y_train * np.log(h)) + np.sum((1-y_train) * np.log(1-h))))
    l += np.sum(y_train * np.log(h)) + np.sum((1-y_train) * np.log(1-h))
    grad = -0.5*beta/sigma**2 + np.sum(X_train.transpose()*(y_train*(1-h)), 1) \
           - np.sum(X_train.transpose()*((1-y_train)*h), 1)

    accuracy = np.mean((a>0.5)==y_train)
    return l, accuracy, grad

def numerical_gradient(func, x):
    grad = np.zeros(x.shape)
    epsilon = 1e-5
    for i in range(len(x)):
        x[i] += epsilon
        p = func(x)
        x[i] -= 2 * epsilon
        n = func(x)
        x[i] += epsilon
        grad[i] = (p - n) / (2 * epsilon)
    return grad

beta = np.zeros((n_dims))
l, accuracy_train, grad = blrf(X_train, y_train, beta, sigma)
_, accuracy_test, _ = blrf(X_test, y_test, beta, sigma)
def get_grad(x):
    l, _, _ = blrf(X_train, y_train, x, sigma)
    return l
# num_grad = numerical_gradient(get_grad, beta)
# print(np.sum(np.square(num_grad - grad)))


# print(l, accuracy_train, accuracy_test)

# Do a MAP solution
map_epsilon = 2e-7
for i in range(50):
    l, accuracy_train, grad = blrf(X_train, y_train, beta, sigma)
    _, accuracy_test, _ = blrf(X_test, y_test, beta, sigma)
    beta = beta + map_epsilon * grad
    print('Log Posterior = {}, Train accuracy = {}, Test accuracy = {}'
          .format(l, accuracy_train, accuracy_test))


# Load German credits dataset
# n = 900
# n_dims = 24
# mu = 0
# sigma = 1./math.sqrt(n)
#
# data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                          'data', 'german.data-numeric')
# X_train, y_train, X_test, y_test = load_uci_german_credits(data_path, n)

# Define graph
# Data
x_input = tf.placeholder(tf.float32, [None, n_dims], name='x_input')
x = tf.Variable(tf.zeros((n, n_dims)), trainable=False, name='x')
y = tf.placeholder(tf.float32, [None], name='y')
update_data = tf.assign(x, x_input, validate_shape=False, name='update_data')


class BLR:
    def __init__(self, x):
        with StochasticGraph() as model:
            beta = Normal(tf.zeros((n_dims)), tf.ones((n_dims)) * tf.log(sigma))
            h = tf.reduce_sum(x * beta.value, 1)
            y_mean = tf.sigmoid(h)
            y = Bernoulli(h)

        self.model = model
        self.y = y
        self.beta = beta
        self.y_mean = y_mean
        self.h = h

    def log_prob(self, latent, observed, given):
        # p(y, beta | X)
        y = observed['y']
        beta = latent['beta']
        y_out, beta_out = self.model.get_output([self.y, self.beta],
                                                inputs={self.y: y, self.beta: tf.identity(beta)})

        return tf.reduce_sum(y_out[1]) + tf.reduce_sum(beta_out[1])

    def evaluate(self, beta):
        prediction = self.model.get_output(self.y_mean, {self.beta: tf.identity(beta)})
        return prediction[0]

#
beta = tf.Variable(np.zeros((n_dims)), dtype=tf.float32)
blr = BLR(x)
hmc = HMC(step_size=1e-4, n_leapfrogs=10)
sampler = hmc.sample(blr.log_prob, {'y': y}, {'beta': beta})

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(update_data, feed_dict={x_input: X_train})

# print('Likelihood = {}'.format(sess.run(blr.log_prob({'beta': beta}, {'y': y}, None),
#                                         feed_dict={y: y_train})))
#
print('Sampling...')
samples = []
for i in range(chain_length):
    sample, pred = sess.run([sampler, blr.evaluate(beta)], feed_dict={y: y_train})
    beta_sample, _, oh, nh, lp, acc = sample
    if i >= burnin:
        samples.append(beta_sample[0])
    train_accuracy = np.mean((pred > 0.5) == y_train)
    print('Log posterior = {}, Training Accuracy = {}, Acceptance Rate = {}'.format(lp, train_accuracy, acc))

e_beta = np.mean(samples, 0)
sess.run(tf.assign(beta, e_beta))
ll, pred = sess.run([blr.log_prob({'beta': beta}, {'y': y}, None), blr.evaluate(beta)],
                    feed_dict={y: y_train})
train_accuracy = np.mean((pred > 0.5) == y_train)
sess.run(update_data, feed_dict={x_input: X_test})
pred = sess.run(blr.evaluate(beta), feed_dict={y: y_test})
test_accuracy = np.mean((pred > 0.5) == y_test)
print('Expected classifier: log posterior = {}, training accuracy = {}, testing accuracy = {}'
      .format(ll, train_accuracy, test_accuracy))

# Evaluate
# scores = tf.reduce_sum(x * beta, reduction_indices=(1,))
# logits = tf.nn.sigmoid(scores)
# predictions = tf.cast(logits > 0.5, tf.float32)
# n_correct = tf.reduce_sum(predictions * y + (1 - predictions) * (1 - y))
# get_log_joint = tf.reduce_sum(norm.logpdf(beta, 0, sigma)) + \
#                 tf.reduce_sum(bernoulli.logpdf(y, logits))

# Sampler
# chain_length = 100
# burnin = 50
#
# sampler = HMC(step_size=1e-5,
#               n_leapfrogs=10,
#               target_acceptance_rate=0.8,
#               m_adapt=burnin)
# sample_steps = sampler.sample(log_joint, vars, mass)
#
# # Session
# sess = tf.Session()
#
# # Find a MAP solution
# sess.run(tf.initialize_all_variables())
# sess.run(update_data, feed_dict={x_input: X_train})
#
# sample_sum = []
# num_samples = chain_length - burnin
# train_scores = np.zeros((X_train.shape[0]))
# test_scores = np.zeros((X_test.shape[0]))
#
# all_samples = []
#
# for i in range(chain_length):
#     # Feed data in
#     sess.run(update_data, feed_dict={x_input: X_train})
#     model, p, oh, nh, acc, t, ss = sess.run(sample_steps,
#                                             feed_dict={y: y_train})
#
#     # Compute model sum
#     if i == burnin:
#         sample_sum = model
#     elif i > burnin:
#         for j in range(len(model)):
#             sample_sum[j] += model[j]
#     if i >= burnin:
#         all_samples.append(model)
#
#     # evaluate
#     n_train_c, train_pred_c, lj = sess.run(
#         [n_correct, logits, get_log_joint], feed_dict={y: y_train})
#     sess.run(update_data, feed_dict={x_input: X_test})
#     n_test_c, test_pred_c = sess.run([n_correct, logits],
#                                      feed_dict={y: y_test})
#     print('Iteration %d, Log likelihood = %f, Acceptance rate = %f, '
#           'Step size = %f, Train set accuracy = %f, test set accuracy = %f' % (
#               i, lj, acc, ss, float(n_train_c) / X_train.shape[0],
#               float(n_test_c) / X_test.shape[0]))
#
#     # Accumulate scores
#     if i >= burnin:
#         train_scores += train_pred_c
#         test_scores += test_pred_c
#
# all_samples = np.squeeze(np.array(all_samples))
#
# # Gibbs classifier
# train_scores /= num_samples
# test_scores /= num_samples
#
# train_pred = (train_scores > 0.5).astype(np.float32)
# test_pred = (test_scores > 0.5).astype(np.float32)
#
# train_accuracy = float(np.sum(train_pred == y_train)) / X_train.shape[0]
# test_accuracy = float(np.sum(test_pred == y_test)) / X_test.shape[0]
#
# # Expected classifier
# # Compute mean
# set_mean = []
# for j in range(len(vars)):
#     set_mean.append(vars[j].assign(sample_sum[j] / num_samples))
# sess.run(set_mean)
#
# # Test expected classifier
# sess.run(update_data, feed_dict={x_input: X_train})
# r_log_likelihood = sess.run(get_log_joint, feed_dict={y: y_train})
# n_train_c = sess.run(n_correct, feed_dict={y: y_train})
# sess.run(update_data, feed_dict={x_input: X_test})
# n_test_c = sess.run(n_correct, feed_dict={y: y_test})
#
# print('Log likelihood of expected parameters: %f, train set accuracy = %f, '
#       'test set accuracy = %f' %
#       (r_log_likelihood, (float(n_train_c) / X_train.shape[0]),
#        (float(n_test_c) / X_test.shape[0])))
# print('Gibbs classifier: train set accuracy = %f, test set accuracy = %f'
#       % (train_accuracy, test_accuracy))
#
# print('ESS = {}'.format(ESS(all_samples, burnin=0)))
