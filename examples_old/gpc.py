import tensorflow as tf
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zhusuan.optimization.gradient_descent_optimizer import \
    GradientDescentOptimizer
from zhusuan.mcmc.nuts import NUTS
from dataset import load_uci_german_credits, load_binary_mnist_realval

# Synthetic dataset
n = 400
n_dims = 8
n_test = 50

beta = np.random.randn(n_dims)
X_train = np.random.randn(n, n_dims)
y_train = (np.matmul(X_train, beta) > 1) * 2.0 - 1.0

X_test = np.random.randn(n_test, n_dims)
y_test = (np.matmul(X_test, beta) > 1) * 2.0 - 1.0

# UCI German Credits dataset
#n = 900
#n_dims = 24
#n_test = 100
#data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
#'data', 'german.data-numeric')
#X_train, y_train, X_test, y_test = load_uci_german_credits(data_path, n)

#y_train = y_train * 2.0 - 1.0
#y_test = y_test * 2.0 - 1.0

# Covariance function
# shape(X)==[N,P], shape(rho)==[P]
def kern(X, rho, eta, X2 = None, J = 0.1):
  Xr = X * rho
  Xrs = tf.reduce_sum(tf.square(Xr), 1)
  Xdist = None
  if X2 is None:
    Xdist = -2 * tf.matmul(Xr, tf.transpose(Xr)) + \
      tf.reshape(Xrs, (-1, 1)) + tf.reshape(Xrs, (1, -1))
    return eta * eta * tf.exp(-Xdist) + tf.eye(tf.shape(X)[0]) * J * J
  else:
    X2r = X2 * rho
    X2rs = tf.reduce_sum(tf.square(X2r), 1)
    Xdist = -2 * tf.matmul(Xr, tf.transpose(X2r)) + \
      tf.reshape(Xrs, (-1, 1)) + tf.reshape(X2rs, (1, -1))
    return eta * eta * tf.exp(-Xdist)

# tf inputs and variables
X_input = tf.placeholder(tf.float32, [n, n_dims], name = "X_input")
y_input = tf.placeholder(tf.float32, [n], name = "y_input")
X_test_input = tf.placeholder(tf.float32, [n_test, n_dims], name = "X_test_input")
f = tf.Variable(tf.random_normal([n]), dtype = tf.float32, name = "f")
rho = tf.Variable(tf.ones(n_dims), dtype = tf.float32, name = "rho")
eta = tf.Variable(1.0, dtype = tf.float32, name = "eta")

# Calculate posterior
K = kern(X_input, rho, eta)
L = tf.cholesky(K)
lnk = 2 * tf.reduce_sum(tf.log(tf.diag_part(L)))
u = tf.matrix_triangular_solve(L, tf.reshape(f,[-1,1]), lower=True)
fKf = tf.reduce_sum(tf.square(u))

lnpost = - tf.reduce_sum(tf.log(1 + tf.exp(- tf.multiply(y_input, f))))
lnpost -= 0.5 * lnk
lnpost -= 0.5 * fKf
# Gaussian prior
lnpost -= eta ** 4
lnpost -= tf.reduce_sum(tf.pow(rho, 4))

# Make predictions
K12 = kern(X_input, rho, eta, X_test_input)
K22 = kern(X_test_input, rho, eta)
LK = tf.matrix_triangular_solve(L, K12, lower=True)
f2mean = tf.matmul(tf.transpose(LK), u)
# Posterior mean is used as the only sample
# More samples can be drawn using post-mean (f2mean) and post-cov-matrix (f2cov)
#f2cov = K22 - tf.matmul(tf.transpose(LK), LK)
f_ = tf.reduce_sum(f2mean, 1)
y_train_prob = 1.0 / (1.0 + tf.exp(-f))
y_test_prob = 1.0 / (1.0 + tf.exp(-f_))
y_train_pred = tf.cast((y_train_prob > 0.5), tf.float32) * 2.0 - 1.0
y_test_pred = tf.cast((y_test_prob > 0.5), tf.float32) * 2.0 - 1.0
n_train_cor = tf.reduce_sum(tf.cast(tf.abs(y_train_pred - y_train) < 1e-2, tf.float32))
n_test_cor = tf.reduce_sum(tf.cast(tf.abs(y_test_pred - y_test) < 1e-2, tf.float32))

# Find MAP
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dict = {X_input: X_train, y_input: y_train, X_test_input: X_test}
optimizer = GradientDescentOptimizer(sess, dict, -lnpost, \
  tf.trainable_variables(), sess.run(tf.trainable_variables(), dict), \
  max_n_iterations=500, stepsize_tol=1e-9, tol=1e-5)
optimizer.optimize()

# Sample and predict
chain_length = 100
burnin = 50
sampler = NUTS(sess, dict, tf.trainable_variables(), lnpost, m_adapt=burnin)

sample_sum = []
num_samples = chain_length - burnin
train_scores = np.zeros(n)
test_scores = np.zeros(n_test)

for i in range(chain_length):
  model = sampler.sample()
  if i == burnin:
    sample_sum = model
  elif i > burnin:
    for j in range(len(model)):
      sample_sum[j] += model[j]

  # Evaluate
  n_train_c, n_test_c, y_train_p, y_test_p, lp = sess.run((n_train_cor, \
    n_test_cor, y_train_prob, y_test_prob, lnpost), dict)
  print('Log likelihood = %f, Train set accuracy = %f, test set accuracy = %f' % \
    (lp, (float(n_train_c) / n), (float(n_test_c) / n_test)))

  if i >= burnin:
    train_scores += y_train_p
    test_scores += y_test_p

train_scores /= num_samples
test_scores /= num_samples

train_pred = (train_scores > 0.5) * 2.0 - 1.0
test_pred = (test_scores > 0.5) * 2.0 - 1.0
train_accuracy = np.sum(np.abs(train_pred - y_train) < 1e-2)
test_accuracy = np.sum(np.abs(test_pred - y_test) < 1e-2)

print('\nFinish sampling: Train set accuracy = %f, Test set accuracy = %f' \
  % (float(train_accuracy) / n, float(test_accuracy) / n_test))
