from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import os
import networkx as nx
import pickle
import random
from random import shuffle
import math

_BIG_NEGATIVE = -1000000.0

def lstm_cell_hidden(mprev, cprev, node_dim, attention_m=False, name=""):
  """Create an LSTM cell.

  The way this LSTM cell is
  used, there is no input x, instead the m and c are updated according to the
  LSTM equations treating the input x as the zero vector. However the m at each
  time step is concatenated with an external input as described in
  https://arxiv.org/pdf/1511.06391.pdf.

  Implements the equations in pg.2 from
  "Long Short-Term Memory Based Recurrent Neural Network Architectures
  For Large Vocabulary Speech Recognition",
  Hasim Sak, Andrew Senior, Francoise Beaufays.

  Args:
    mprev: m_{t-1}, the recurrent activations (same as the output)
      from the previous cell.
    cprev: c_{t-1}, the cell activations from the previous cell.
    node_dim: Number of hidden state of the LSTM.
    attention_m: If true then the hidden dim is twice the size of the cell dim
    name: prefix for the variable names

  Returns:
    m: Outputs of this cell.
    c: Cell Activations.
  """

  # Input Gate
  m_nodes = node_dim
  if attention_m:
    m_nodes = 2 * node_dim
  im = tf.get_variable(name + "im", [m_nodes, node_dim])
  ib = tf.get_variable(
      name + "ib", [1, node_dim], initializer=tf.zeros_initializer)
  i_g = tf.sigmoid(tf.matmul(mprev, im) + ib, name="i_g")

  # Forget Gate
  fm = tf.get_variable(name + "fm", [m_nodes, node_dim])
  fb = tf.get_variable(
      name + "fb", [1, node_dim], initializer=tf.zeros_initializer)
  f_g = tf.sigmoid(tf.matmul(mprev, fm) + fb, name="f_g")

  # Cell
  cm = tf.get_variable(name + "cm", [m_nodes, node_dim])
  cb = tf.get_variable(
      name + "cb", [1, node_dim], initializer=tf.zeros_initializer)
  cprime = tf.sigmoid(tf.matmul(mprev, cm) + cb)
  c = f_g * cprev + i_g * tf.tanh(cprime)

  # Output Gate
  om = tf.get_variable(name + "om", [m_nodes, node_dim])
  ob = tf.get_variable(
      name + "ob", [1, node_dim], initializer=tf.zeros_initializer)
  o_g = tf.sigmoid(tf.matmul(mprev, om) + ob, name="o_g")

  m = o_g * tf.tanh(c)
  return m, c


def set2vec(input_set,
            num_timesteps,
            mprev=None,
            cprev=None,
            mask=None,
            inner_prod="default",
            name="lstm"):
  """Part of the set2set model described in Vinyals et. al.

  Specifically this implements the "process" block described in
  https://arxiv.org/pdf/1511.06391.pdf. This maps a set to a single embedding
  m which is invariant to the order of the elements in that set. Thus it should
  be thought of as a "set2vec" model. It is part of the full set2set model from
  the paper. 

  There is an LSTM which from t = 1,...,num_timesteps emits a query vector at
  each time step, which is used to perform content based attention over the
  embedded input set (see https://arxiv.org/pdf/1506.03134.pdf sec 2.2), and
  the result of that content based attention is then fed back into the LSTM
  by concatenation with m, the output of the LSTM at that time step. After
  num_timesteps of computation we return the final cell c, and output m.
  m can be considered the order invariant embedding of the input_set.

  Args:
    input_set: tensor of shape [batch_size, num_nodes, 1, node_dim]
    num_timesteps: number of computation steps to run the LSTM for
    mprev: Used to initialize the hidden state of the LSTM, pass None if
      the hidden state should be initialized to zero.
    cprev: Used to initialize the cell of the LSTM, pass None if the cell
      state should be initialized to zero.
    mask: tensor of type bool, shape = [batch_size,num_nodes]. This is
      used when batches may contain sets of different sizes. The values should
      be binary. If set to None then the model will assume all sets have the
      same size.
    inner_prod: either 'default' or 'dot'. Default uses the attention mechanism
      as described in the pointer networks paper. Dot is standard dot product.
      The experiments for the MPNN paper (https://arxiv.org/pdf/1704.01212.pdf)
      did not show a significant difference between the two inner_product types,
      and the final experiments were run with default.
    name: (string)

  Returns:
    logit_att: A list of the attention masks over the set.
    c: The final cell state of the internal LSTM.
    m: The final output of the internal LSTM (note this is what we use as the
      order invariant representation of the set).

  Raises:
    ValueError: If an invalid inner product type is given.
  """

  batch_size = tf.shape(input_set)[0]
  node_dim = int(input_set.shape[3])

  # For our use case the "input" to the LSTM at each time step is the
  # zero vector, instead the hidden state of the LSTM at each time step
  # will be concatenated with the output of the content based attention
  # (see eq's 3-7 in the paper).
  if mprev is None:
    mprev = tf.zeros([batch_size, node_dim])
  mprev = tf.concat([mprev, tf.zeros([batch_size, node_dim])], axis=1)
  if cprev is None:
    cprev = tf.zeros([batch_size, node_dim])

  logit_att = []
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    attention_w2 = tf.get_variable(name + "att_W_2", [node_dim, node_dim])
    attention_v = tf.get_variable(name + "att_V", [node_dim, 1])

  # Batches may contain sets of different sizes, in which case the smaller
  # sets will be padded with null elements as specified by the mask.
  # In order to make the set2vec model invariant to this padding, we add
  # large negative numbers to the logits of the attention softmax (which when
  # exponentiated will become 0).
  if mask is not None:
    mask = tf.cast(mask, tf.float32)
    mask = (1 - mask) * _BIG_NEGATIVE
  for i in range(num_timesteps):
    with tf.variable_scope(
        tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      with tf.name_scope("%s_%d" % (name, i)):
        m, c = lstm_cell_hidden(
            mprev, cprev, node_dim, attention_m=True, name=name)
        query = tf.matmul(m, attention_w2)
        query = tf.reshape(query, [-1, 1, 1, node_dim])
        if inner_prod == "default":
          energies = tf.reshape(
              tf.matmul(
                  tf.reshape(tf.tanh(query + input_set), [-1, node_dim]),
                  attention_v), [batch_size, -1])
        elif inner_prod == "dot":
          att_mem_reshape = tf.reshape(input_set, [batch_size, -1, node_dim])
          query = tf.reshape(query, [-1, node_dim, 1])
          energies = tf.reshape(
              tf.matmul(att_mem_reshape, query), [batch_size, -1])
        else:
          raise ValueError("Invalid inner_prod type: {}".format(inner_prod))

        # Zero out the non nodes.
        if mask is not None:
          energies += mask
        att = tf.nn.softmax(energies)

        # Multiply attention mask over the elements of the sets.
        att = tf.reshape(att, [batch_size, -1, 1, 1])

        # Take the weighted average the elements in the set
        # This is the 'r' of the paper.
        read = tf.reduce_sum(att * input_set, [1, 2])
        m = tf.concat([m, read], axis=1)

        logit_att.append(m)
        mprev = m
        cprev = c

  return logit_att, c, m, read


class Input(object):
    def __init__(self, data, length, batch_size, max_len, shuffle_row=False, shuffle_col=False):
        self.data = data
        self.length = length

        self.batch_size = batch_size
        self.max_len = max_len # max length of a sample

        self.num_epochs = 0

        self.n_samples = len(self.data)
        self.epoch_size = self.n_samples // self.batch_size

        self.shuffle_row = shuffle_row
        self.shuffle_col = shuffle_col

        self.cur = 0

        self.lower_ones = np.tril(np.ones((max_len, max_len)))
        self.diag_ones = np.eye(self.max_len, dtype=int)

    def shuffle(self):
        self.cur = 0
        self.num_epochs += 1

        if self.shuffle_row:
            idx = np.arange(self.n_samples, dtype=int)
            random.shuffle(idx)
            self.data = self.data[idx]
            self.length = self.length[idx]

        if self.shuffle_col:
            for i in range(self.n_samples):
                l = self.length[i]
                idx = np.arange(l, dtype=int)
                random.shuffle(idx)
                self.data[i][:l] = self.data[i][idx]


    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')

        x = self.data[self.cur:self.cur+self.batch_size]
        x_len = self.length[self.cur:self.cur+self.batch_size]

        mask_y = np.take(self.diag_ones, x_len-1, axis=0)
        y = np.sum(x * mask_y, axis=1)

        # mask for valid entries in x
        mask_x = np.take(self.lower_ones, x_len-2, axis=0)

        self.cur += self.batch_size

        return x, mask_x, y

# x : [-1, n_input]
def dense(x, n_input, n_output, seed, scope='dense'):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable(name='w1', shape=[n_input, n_output],
							initializer=tf.glorot_uniform_initializer(seed=seed, dtype=tf.float32))

		b1 = tf.get_variable(name='b1', shape=[n_output],
							initializer=tf.zeros_initializer(dtype=tf.float32))

	x1 = tf.matmul(x, w1) + b1

	return x1


def mlp(x, x_dim, h_dim, seed=123, scope='mlp'):
    x1 = dense(x,
            x_dim,
            h_dim[0],
            seed=seed,
            scope=scope+'/dense0')

    x1 = tf.nn.relu(x1)

    for i in range(len(h_dim)-2):
        x1 = dense(x1,
                h_dim[i],
                h_dim[i+1],
                seed=seed,
                scope=scope+'/dense%d'%(i+1))

        x1 = tf.nn.relu(x1)

    logits = dense(x1,
                h_dim[-2],
                h_dim[-1],
                seed=seed,
                scope=scope+'/dense%d'%(len(h_dim)))

    #prob = tf.nn.sigmoid(logits)
    prob = tf.nn.softmax(logits)

    return prob, logits


# return top results
def hit_at(x_pred, target, top):
    x_ind = np.argsort(x_pred)[::-1]
    top_x = x_ind[:top]
    if target in top_x:
        return 1.0
    else:
        return 0.0

