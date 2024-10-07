import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import random
from tqdm import tqdm

from src.imputation.utils import (normalization, renormalization, xavier_init,
                                  binary_sampler, sample_batch_index, uniform_sampler)

random.seed(42)
tf.set_random_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

class GainClass:
  """Class that provides GAIN imputation"""
  # Sources:
  # 1. Title: GAIN Imputation for static data
  #    Authors: J.Yoon, J.Jordon, M.van der Schaar
  #    Available at: https://github.com/jsyoon0823/GAIN/tree/master
  #    Accessed on: 19-03-2024
  #
  # 2. Title: Handling missing data - imputation and classification
  #    Author: Julian Gilbey
  #    Available at: https://gitlab.developers.cam.ac.uk/maths/cia/covid-19-projects/handling_missing_data/-/tree/main/models/gain?ref_type=heads
  #    Accessed on: 19-03-2024


  def __init__(self, gain_parameters):

      # System parameters
      self.batch_size = gain_parameters['batch_size']
      self.hint_rate = gain_parameters['hint_rate']
      self.alpha = gain_parameters['alpha']
      self.iterations = gain_parameters['iter']

      self.norm_params = None
      self.model = None
      self.reached_iterations = gain_parameters['iter']

  def fit (self, data_x):
    """Fit GAIN model.

    Args:
      - x: incomplete dataset
    """
    tf.reset_default_graph()

    ## System Parameters
    iterations = self.iterations
    alpha = self.alpha

    no, dim = data_x.shape
    h_dim = int(dim)

    # mask matrix
    data_m = 1 - np.isnan(data_x)

    # normalize data
    norm_data, self.norm_params = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN Architecture
    # Input Placeholders
    # Mask Vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])
    # Data with missing values
    X = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables

    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN Function
    # Generator
    def generator(x,m):
      inputs = tf.concat(axis=1, values=[x, m])
      G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
      G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
      G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
      return G_prob

    # Discriminator
    def discriminator(x, h):
      inputs = tf.concat(axis=1, values=[x, h])
      D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
      D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
      D_logit = tf.matmul(D_h2, D_W3) + D_b3
      D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
      return D_prob

    ## GAIN Structure
    # Generator
    G_sample = generator(X, M)

    # Combine with original data
    X_hat = X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(X_hat, H)

    ## Loss
    D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
    G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
    MSE_train_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    D_loss = D_loss1
    G_loss = G_loss1 + alpha * MSE_train_loss

    ## Solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ## Start Iterations
    for it in tqdm(range(iterations)):

      # Inputs
      batch_idx = sample_batch_index(no, self.batch_size)
      x_mb = norm_data_x[batch_idx, :]
      m_mb = data_m[batch_idx, :]

      z_mb = uniform_sampler(0, 0.01, self.batch_size, dim)
      h_mb = binary_sampler(self.hint_rate, self.batch_size, dim)
      h_mb = m_mb * h_mb

      # Combine random vectors with observed vectors
      x_mb = m_mb * x_mb + (1-m_mb) * z_mb
      theta_D_val = sess.run(theta_D)
      theta_G_val = sess.run(theta_G)
      _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: m_mb, X: x_mb, H: h_mb})
      _, G_loss_curr = sess.run([G_solver, G_loss1], feed_dict={M: m_mb, X: x_mb, H: h_mb})

      if np.isnan(D_loss_curr) or np.isnan(G_loss_curr) or np.isinf(D_loss_curr) or np.isinf(G_loss_curr):
          # use the weights from previous iteration
          self.reached_iterations = it
          for i, var in enumerate(theta_D):
              sess.run(var.assign(theta_D_val[i]))
          for i, var in enumerate(theta_G):
              sess.run(var.assign(theta_G_val[i]))
          break

    self.model = {'inputs': {'X': X, 'M': M}, 'outputs': {'imputation': G_sample}, 'sess': sess}

  def transform(self, data_x):
      """Return imputed data by trained GAIN model.
        Args:
          - data: 2d numpy array with missing data

        Returns:
          - imputed data: 2d numpy array without missing data
        """

      no, dim = data_x.shape
      data_m = 1 - np.isnan(data_x)

      norm_x, _ = normalization(data_x, self.norm_params)
      # Set missing
      x = np.nan_to_num(norm_x, 0)

      ## Imputed data
      z = np.random.uniform(0., 0.01, size=[no, dim])
      x = data_m * x + (1 - data_m) * z

      # restore the original GAIN
      X = self.model['inputs']['X']
      M = self.model['inputs']['M']
      G_sample = self.model['outputs']['imputation']
      sess = self.model['sess']
      imputed_data = sess.run(G_sample, feed_dict={X: x, M: data_m})

      imputed_data = data_m * x + (1 - data_m) * imputed_data

      # Renormalize
      imputed_data = renormalization(imputed_data, self.norm_params)

      return imputed_data
