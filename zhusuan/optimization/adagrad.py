#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from copy import copy

class AdaGrad:
    def __init__(self, underflow_constant=1e-7):
        self.underflow_constant = underflow_constant

    def update_step_size(self, gradient, squared_learning_rate, global_learning_rate=1e-3):
        self.global_learning_rate = global_learning_rate
        new_r = map(lambda (x, y): x*x+y, zip(gradient, squared_learning_rate))
        new_update = map(lambda (x, y): global_learning_rate*y
                           / (self.underflow_constant + tf.sqrt(x)), zip(new_r, gradient))
        return new_update, new_r

class RMSProp:
    def __init__(self, underflow_constant=1e-7):
        self.underflow_constant = underflow_constant

    def update_step_size(self, gradient, squared_learning_rate, global_learning_rate=1e-3, decay_rate=0.5):
        self.global_learning_rate = global_learning_rate
        self.decay_rate = decay_rate
        new_r = map(lambda (x, y): (1-self.decay_rate)*x*x+self.decay_rate*y, zip(gradient, squared_learning_rate))
        new_update = map(lambda (x, y): global_learning_rate*y
                           / tf.sqrt(self.underflow_constant+x), zip(new_r, gradient))
        return new_update, new_r
