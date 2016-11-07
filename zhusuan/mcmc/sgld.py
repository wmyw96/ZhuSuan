#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from copy import copy
from zhusuan.optimization.adagrad import AdaGrad, RMSProp

class SGLD:
    def __init__(self, sample_threshold=10^-2, a=1.26e-3, b=0.023, gamma=0.55, global_learning_rate=1e-3):
        self.sample_threshold = sample_threshold
        self.a = a
        self.b = b
        self.gamma = gamma
        self.global_learning_rate = global_learning_rate

    def sample(self, log_likelihood, log_prior, var_list=None, ratio=None, minibatch_size=None,
               squared_learning_rate=None, RMS_decay_rate=None, Ada=False, RMS=False):
        self.q = copy(var_list)
        self.q_shapes = map(lambda x: x.initialized_value().get_shape(), self.q)
        self.ratio=ratio
        self.Ada=Ada
        self.minibatch_size=minibatch_size
        self.RMS=RMS
        self.RMS_decay_rate=RMS_decay_rate

        self.t = tf.Variable(0.0)
        new_t = tf.assign(self.t, tf.add(self.t, 1.0))

        #Get step size
        def get_step_size(i):
            step_size=self.a*tf.pow(self.b+i,-self.gamma)
            return step_size

        #Generate injected Gaussian noise
        def get_injected_noise(i):
            noise = map(lambda s: tf.sqrt(get_step_size(i))*tf.random_normal(shape=s), self.q_shapes)
            return noise

        current_q = copy(self.q)

        def get_prior_gradient(var_list):
            log_pr = log_prior(var_list)
            grads = tf.gradients(log_pr, var_list)
            return log_pr, grads

        def get_likelihood_gradient(var_list):
            log_post = log_likelihood(var_list)
            grads = tf.gradients(log_post, var_list)
            return log_post, grads

        # Full steps
        _, priorgrads = get_prior_gradient(current_q)
        _, minibatch_likeligrads = get_likelihood_gradient(current_q)
        total_likeligrads = map(lambda x: self.ratio * x, minibatch_likeligrads)
        # AdaGrad step size
        if self.Ada==True:
            self.r = copy(squared_learning_rate)
            AdaG = AdaGrad(underflow_constant=1e-7)
            sgd_likeligrads = map(lambda (x, y): x / self.minibatch_size + y, zip(minibatch_likeligrads, priorgrads))
            [new_update, current_r] = AdaG.update_step_size(sgd_likeligrads, self.r, self.global_learning_rate)
            current_q = map(lambda (x, y, noise): x + 0.5*y + noise, zip(current_q, new_update, get_injected_noise(new_t)))
            current_step_size = map(lambda (x, y): x.assign(y), zip(self.r, current_r))

        # RMSProp step size
        elif self.RMS==True:
            self.r = copy(squared_learning_rate)
            RMSP = RMSProp(underflow_constant=1e-7)
            sgd_likeligrads = map(lambda (x, y): x / self.minibatch_size + y, zip(minibatch_likeligrads, priorgrads))
            [new_update, current_r] = RMSP.update_step_size(sgd_likeligrads, self.r,
                                                            self.global_learning_rate, self.RMS_decay_rate)
            current_q = map(lambda (x, y, noise): x + 0.5*y + noise, zip(current_q, new_update, get_injected_noise(new_t)))
            current_step_size = map(lambda (x, y): x.assign(y), zip(self.r, current_r))

        # Default power decrease step size
        else:
            current_q = map(lambda (x, prior_g, likelihood_g, noise):
                            x + 0.5 * get_step_size(new_t) * (prior_g + likelihood_g) + noise,
                            zip(current_q, priorgrads, total_likeligrads, get_injected_noise(new_t)))
            current_step_size = get_step_size(new_t)

        new_q = map(lambda (x, y): x.assign(y),
                                    zip(self.q, current_q))
        return new_q, current_step_size

