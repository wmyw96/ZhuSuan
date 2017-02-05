#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSL with graph based on a pretrained VAE's latent z. q(y) not conditioned on x.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
from sklearn.neighbors import NearestNeighbors
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


def const_graph(x, k):
    """
    To compute l2 distance of each data point pair.
    :param x: (batch_size, feature_size)
    :param k: k nearest neighbour
    :return: sparse matrix w.
    """
    # x_col = np.expand_dims(x, 0)
    # x_row = np.expand_dims(x, 1)
    # w = np.square(x_col - x_row)
    # w = np.sum(w, 2)
    # index = np.argsort(w, axis=-1)
    # ind2 = index[:, :k+1].flatten()  # include itself
    # knn_w = np.zeros_like(w, dtype='float32')
    # ind1 = np.tile(np.expand_dims(np.arange(w.shape[0]), 1), [1, 2]).flatten()
    # knn_w[ind1, ind2] = np.exp(-w[ind1, ind2])  # symmetric
    # knn_w = np.maximum(knn_w, knn_w.transpose())
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
    distances, indices = nbrs.kneighbors(x)
    return indices, distances












