#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import math
from copy import copy

class SGLD:
    def __init__(self, sample_threshold=10^-2, a=1.26e-3, b=0.023, gamma=0.55):
        self.sample_threshold = sample_threshold
        self.a = a
        self.b = b
        self.gamma = gamma

    def sample(self, log_likelihood, log_prior, var_list=None, ratio=None):
        self.q = copy(var_list)
        self.q_shapes = map(lambda x: x.initialized_value().get_shape(), self.q)
        self.ratio=ratio

        self.t = tf.Variable(0.0)
        new_t = tf.assign(self.t, self.t+1)

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
        # def loop_cond(self,i, current_q):
        #     return self.get_step_size(i) > self.sample_threshold

        # def loop_body(self,i, current_q):
        _, priorgrads = get_prior_gradient(current_q)
        _, minibatch_likeligrads = get_likelihood_gradient(current_q)
        total_likeligrads = self.ratio*minibatch_likeligrads
        current_q = map(lambda x: x + 0.5*get_step_size(new_t)*(priorgrads+total_likeligrads)+get_injected_noise(new_t),
                            current_q)
            # return [i+1, current_q]

        # i = tf.constant(0)
        # I, current_q = tf.while_loop(loop_cond,
        #                      loop_body,
        #                      [i, current_q],
        #                      back_prop=False)
        new_q = map(lambda (x, y): x.assign(y),
                                    zip(self.q, current_q))
        current_step_size = get_step_size(new_t)
        return new_q, current_step_size

