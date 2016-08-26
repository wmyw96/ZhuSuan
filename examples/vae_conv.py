#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    from deconv import deconv2d
except:
    raise ImportError()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
            'task_index', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')
tf.app.flags.DEFINE_integer("num_workers", None,
                            "Total number of workers (must be >= 1)")


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_z, n_x):
        self.n_z = n_z
        self.n_x = n_x
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            self.l_x_z = (pt.template('z').
                          reshape([-1, 1, 1, self.n_z]).
                          deconv2d(3, 128, edges='VALID').
                          batch_normalize(scale_after_normalization=True).
                          deconv2d(5, 64, edges='VALID').
                          batch_normalize(scale_after_normalization=True).
                          deconv2d(5, 32, stride=2).
                          batch_normalize(scale_after_normalization=True).
                          deconv2d(5, 1, stride=2,
                                   activation_fn=tf.nn.sigmoid))

    def log_prob(self, latent, observed):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        z = latent['z']
        x = observed['x']

        l_x_z = self.l_x_z.construct(
            z=tf.reshape(z, (-1, self.n_z))).reshape(
            (-1, tf.shape(z)[1], self.n_x)).tensor
        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        return log_px_z + log_pz


def q_net(n_x, n_xl, n_z, n_samples):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param n_x: Int. The dimension of observed variables (x).
    :param n_xl: Int. The height/width dimension of observed variables (x).
    :param n_z: Int. The dimension of latent variables (z).
    :param n_samples: A Int or a Tensor of type int. Number of samples of
        latent variables.

    :return: All :class:`Layer` instances needed.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        lx = InputLayer((None, n_x))
        lz_x = PrettyTensor({'x': lx}, pt.template('x').
                            reshape([-1, n_xl, n_xl, 1]).
                            conv2d(5, 32, stride=2).
                            batch_normalize(scale_after_normalization=True).
                            conv2d(5, 64, stride=2).
                            batch_normalize(scale_after_normalization=True).
                            conv2d(5, 128, edges='VALID').
                            batch_normalize(scale_after_normalization=True).
                            dropout(0.9).
                            flatten())
        lz_mean = PrettyTensor({'z': lz_x}, pt.template('z').
                               fully_connected(n_z, activation_fn=None).
                               reshape((-1, 1, n_z)))
        lz_logstd = PrettyTensor({'z': lz_x}, pt.template('z').
                                 fully_connected(n_z, activation_fn=None).
                                 reshape((-1, 1, n_z)))
        lz = ReparameterizedNormal([lz_mean, lz_logstd], n_samples)
    return lx, lz


if __name__ == "__main__":
    ############################################################################
    #Parse the parameter server and worker address
    ps_hosts = FLAGS.ps_hosts.split(",")
    print (ps_hosts)
    worker_hosts = FLAGS.worker_hosts.split(",")
    print (worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name = FLAGS.job_name,
                             task_index = FLAGS.task_index)
    ############################################################################
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]
    n_xl = np.sqrt(n_x)

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 30
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10

    ############################################################################
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        #Assign following operators to workers by default
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            global_step = tf.Variable(0)
            is_chief = (FLAGS.task_index == 0)
    ############################################################################
            # Build the training computation graph
            x = tf.placeholder(tf.float32, shape=(batch_size, x_train.shape[1]))
            n_samples = tf.placeholder(tf.int32, shape=())
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
            optimizer = tf.train.SyncReplicasOptimizer(#synchronous optimizer
                            optimizer,
                            replicas_to_aggregate=FLAGS.num_workers,
                            total_num_replicas=FLAGS.num_workers,
                            replica_id=FLAGS.task_index)
            with pt.defaults_scope(phase=pt.Phase.train):
                with tf.variable_scope("model") as scope:
                    train_model = M1(n_z, x_train.shape[1])
                with tf.variable_scope("variational") as scope:
                    lx, lz = q_net(n_x, n_xl, n_z, n_samples)
            grads, lower_bound = advi(
                train_model, {'x': x}, {'x': lx}, {'z': lz}, optimizer)
            infer = optimizer.apply_gradients(grads, global_step=global_step)
            
            # Build the evaluation computation graph
            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model", reuse=True) as scope:
                    eval_model = M1(n_z, x_train.shape[1])
                with tf.variable_scope("variational", reuse=True) as scope:
                    lx, lz = q_net(n_x, n_xl, n_z, n_samples)
            eval_log_likelihood = is_loglikelihood(
                eval_model, {'x': x}, {'x': lx}, {'z': lz})
            
            params = tf.trainable_variables()
            for i in params:
                print(i.name, i.get_shape())

            if is_chief:
            # Initial token and chief queue runners required by the sync_replicas mode
                chief_queue_runner = optimizer.get_chief_queue_runner()
                init_tokens_op = optimizer.get_init_tokens_op()
            
        saver = tf.train.Saver()
        summary = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        ############################################################################
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief = is_chief,
                                 init_op = init,
                                 summary_op = summary,
                                 saver = saver,
                                 global_step = global_step,
                                 save_model_secs = FLAGS.save_interval_secs)
        # Create a local session to run the training.
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)
        with sv.prepare_or_wait_for_session(server.target, config=config) as sess:
            if is_chief:
                # Chief worker will start the chief queue runner and call the init op
                print("Starting chief queue runner and running init_tokens_op")
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_tokens_op)
        ############################################################################
        # Run the inference
        #with tf.Session() as sess:
            #sess.run(init)
            for epoch in range(1, epoches + 1):
                np.random.shuffle(x_train)
                lbs = []
                for t in range(iters):
                    x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                    x_batch = np.random.binomial(
                        n=1, p=x_batch, size=x_batch.shape).astype('float32')
                    _, lb = sess.run([infer, lower_bound],
                                     feed_dict={x: x_batch, n_samples: lb_samples})
                    lbs.append(lb)
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
                if epoch % test_freq == 0:
                    test_lbs = []
                    test_lls = []
                    for t in range(test_iters):
                        test_x_batch = x_test[
                            t * test_batch_size: (t + 1) * test_batch_size]
                        test_lb = sess.run(eval_log_likelihood,
                                           feed_dict={x: test_x_batch,
                                                      n_samples: lb_samples})
                        test_ll = sess.run(eval_log_likelihood,
                                           feed_dict={x: test_x_batch,
                                                      n_samples: ll_samples})
                        test_lbs.append(test_lb)
                        test_lls.append(test_ll)
                    print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                    print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
        ############################################################################
        # Ask for all the services to stop.
        sv.stop()
        ############################################################################
