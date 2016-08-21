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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp
    from zhusuan.variational import ReparameterizedNormal, advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
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
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               scale_after_normalization=True):
            self.l_x_z = (pt.template('z').
                          fully_connected(500).
                          batch_normalize().
                          fully_connected(500).
                          batch_normalize().
                          fully_connected(n_x, activation_fn=tf.nn.sigmoid))

    def log_prob(self, z, x):
        """
        The joint likelihood of M1 deep generative model.

        :param z: Tensor of shape (batch_size, samples, n_z). n_z is the
            dimension of latent variables.
        :param x: Tensor of shape (batch_size, n_x). n_x is the dimension of
            observed variables (data).

        :return: A Tensor of shape (batch_size, samples). The joint log
            likelihoods.
        """
        l_x_z = self.l_x_z.construct(
            z=tf.reshape(z, (-1, self.n_z))).reshape(
            (-1, int(z.get_shape()[1]), self.n_x)).tensor
        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        return log_px_z + log_pz


def q_net(x, n_z):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: Tensor of shape (batch_size, n_x).
    :param n_z: Int. The dimension of latent variables (z).

    :return: A Tensor of shape (batch_size, n_z). Variational mean of latent
        variables.
    :return: A Tensor of shape (batch_size, n_z). Variational log standard
        deviation of latent variables.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           scale_after_normalization=True):
        l_z_x = (pt.wrap(x).
                 fully_connected(500).
                 batch_normalize().
                 fully_connected(500).
                 batch_normalize())
        l_z_x_mean = l_z_x.fully_connected(n_z, activation_fn=None)
        l_z_x_logstd = l_z_x.fully_connected(n_z, activation_fn=None)
    return l_z_x_mean, l_z_x_logstd


def is_loglikelihood(model, x, z_proposal, n_samples=1000):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param model: A model object that has a method logprob(z, x) to compute the
        log joint likelihood of the model.
    :param x: A Tensor of shape (batch_size, n_x). The observed variables (
        data).
    :param z_proposal: A :class:`Variational` object used as the proposal
        in importance sampling.
    :param n_samples: Int. Number of samples used in this estimate.

    :return: A Tensor of shape (batch_size,). The log likelihood of data (x).
    """
    samples = z_proposal.sample(n_samples)
    log_w = model.log_prob(samples, x) - z_proposal.logpdf(samples)
    return log_mean_exp(log_w, 1)


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

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    epoches = 3000
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
            x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
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
                    train_vz_mean, train_vz_logstd = q_net(x, n_z)
                    train_variational = ReparameterizedNormal(
                        train_vz_mean, train_vz_logstd)
            grads, lower_bound = advi(
                train_model, x, train_variational, lb_samples, optimizer)
            infer = optimizer.apply_gradients(grads, global_step=global_step)

            # Build the evaluation computation graph
            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model", reuse=True) as scope:
                    eval_model = M1(n_z, x_train.shape[1])
                with tf.variable_scope("variational", reuse=True) as scope:
                    eval_vz_mean, eval_vz_logstd = q_net(x, n_z)
                    eval_variational = ReparameterizedNormal(
                        eval_vz_mean, eval_vz_logstd)
            eval_lower_bound = is_loglikelihood(
                eval_model, x, eval_variational, lb_samples)
            eval_log_likelihood = is_loglikelihood(
                eval_model, x, eval_variational, ll_samples)

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
                sys.stdout.flush()
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, lower_bound], feed_dict={x: x_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}, timestamp = {}'.format(epoch, np.mean(lbs), datetime.now()))
            if epoch % test_freq == 0:
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb, test_ll = sess.run(
                        [eval_lower_bound, eval_log_likelihood],
                        feed_dict={x: test_x_batch}
                    )
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
    ############################################################################
    # Ask for all the services to stop.
    sv.stop()
    ############################################################################
