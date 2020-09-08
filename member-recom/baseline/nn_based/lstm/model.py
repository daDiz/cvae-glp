import tensorflow as tf
import numpy as np
import os
import networkx as nx
import pickle
import random

def shuffle_authors(authors, seqlen, batch_size, time_step, do_shuffle_authors=True):
    label = []
    for i in range(batch_size):
        tmp = []
        for j in range(time_step):
            l = seqlen[i][j]
            if do_shuffle_authors:
                idx = np.arange(l, dtype=int)
                random.shuffle(idx)
                authors[i][j][:l] = authors[i][j][idx]
            tmp.append(authors[i][j][l-1])

        label.append(tmp)

    return np.array(label), authors

class Input(object):
    def __init__(self, event, authors, seqlen, batch_size, do_shuffle_authors):
        self.event = event # [n_sample, time_step]
	self.authors = authors # [n_sample, max_num_auth]
        self.seqlen = seqlen # [n_sample]
        self.batch_size = batch_size
        self.num_epochs = 0

        self.n_samples = self.event.shape[0]
        self.epoch_size = self.n_samples // self.batch_size
        self.time_step = self.event.shape[1]
        self.max_num_auth = self.authors.shape[1]

        self.do_shuffle_authors = do_shuffle_authors # whether shuffling the authors for each doc

        self.cur = 0


    def shuffle(self):
        self.cur = 0
        idx = np.arange(self.n_samples, dtype=int)
        random.shuffle(idx)
        self.event = self.event[idx]
        self.num_epochs += 1

    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')

        # [batch_size, time_step, num_authors]
        event_ = self.event[self.cur:self.cur+self.batch_size]

	authors_ = self.authors[event_]
	seqlen_ = self.seqlen[event_] 

        label, example = shuffle_authors(authors_, seqlen_, self.batch_size, self.time_step, self.do_shuffle_authors)

        self.cur += self.batch_size


        return example, label, seqlen_


class Model(object):
    def __init__(self, embedd, lower_triangular_ones, is_training, batch_size, time_step, N,
                hidden_size, num_layers, dropout=0.5, init_scale=0.05):
        if is_training: # embedd should be tf.random_uniform([N, hidden_size],
                        #                                   -init_scale, init_scale])
            self.embedding = tf.Variable(initial_value=embedd, dtype=tf.float32, trainable=True,
            name='embedding')
        else: # load learned embedding
            self.embedding = tf.Variable(initial_value=embedd, dtype=tf.float32, trainable=False,
            name='embedding')


        self.is_training = is_training
        self.batch_size = batch_size
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.N = N
        self.num_layers = num_layers


        self.dropout = dropout
        self.init_scale = init_scale


        # author indices as input, multiple authors per time_step, including all authors of each
        # document
        # [batch_size, time_step, MAX_NUM_AUTH]
        self.x = tf.placeholder(tf.int32, [batch_size, time_step, None])

        # author indices as output, one author per time step
        # [batch_size, time_step]
        self.y = tf.placeholder(tf.int32, [batch_size, time_step])

        # number of authors per time_step
        self.z = tf.placeholder(tf.int32, [batch_size, time_step])


        # [batch_size, time_step, MAX_NUM_AUTH, hidden_size]
        embedd = tf.nn.embedding_lookup(self.embedding, self.x)

        # [batch_size*time_step]
        seqlen = tf.reshape(self.z, [-1])

        # [batch_size*time_step, MAX_NUM_AUTH]
        # minus 2 as the last elem is label
        mask = tf.gather(lower_triangular_ones, seqlen - 2)

        # [batch_size, time_step, MAX_NUM_AUTH]
        mask = tf.reshape(mask, [batch_size, time_step, -1])

        # [batch_size, time_step, MAX_NUM_AUTH, 1]
        # expand one dim at the end
        mask = tf.expand_dims(mask, axis=-1)

        # [batch_size, time_step, MAX_NUM_AUTH, 1]
        # apply mask to the embedding
        # the embeddings out of seqlen become zero
        embedd = tf.multiply(mask, embedd)

        # [batch_size, time_step, hidden_size]
        inputs = tf.reduce_sum(embedd, axis=2)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)


        self.init_state = tf.placeholder(tf.float32,
                                        [num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)

        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],
            state_per_layer_list[idx][1])
            for idx in range(num_layers)]
        )

        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)

        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                            initial_state=rnn_tuple_state)


        # reshape to (batch_size*time_step,hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        # squeeze results from two lstms to ensure in the same scale
        output = tf.nn.tanh(output)

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, N], -self.init_scale,self.init_scale))
        softmax_b = tf.Variable(tf.random_uniform([N], -self.init_scale, self.init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, -1, N])

        # use contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(logits, self.y,
                                    tf.ones([self.batch_size, self.time_step], dtype=tf.float32),
                                    average_across_timesteps=False,
                                    average_across_batch=True)


        # update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, N]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.y,[-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return

        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})



if __name__ == '__main__':
    with open('./data/seq_1951_1970_train.pickle') as file:
        event = pickle.load(file)

    with open('./data/seq_1951_1970_author_train.pickle') as file:
	authors = pickle.load(file)

    with open('./data/seq_1951_1970_len_train.pickle') as file:
	seqlen = pickle.load(file)


    num_epochs = 10
    batch_size = 32
    do_shuffle_authors = True
    input_obj = Input(event, authors, seqlen, batch_size, do_shuffle_authors)

    for epoch in range(num_epochs):
	for step in range(input_obj.epoch_size):
            if step % 10 == 0:
                print("Epoch: %d, step: %d" % (epoch, step))
                x,y,z = input_obj.next_batch()
                print(x)
                print(y)
                print(z)

	input_obj.shuffle()
