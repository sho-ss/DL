import tensorflow as tf
import numpy as np
import logging
import os
import shutil
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class MaxOut():
	def __init__(self, n_in, n_out, n_hiddens, dropout, batch_size=100, every_eval=10, logger=None):
		"""
		@param n_hiddens list
			listの要素, scalar or tuple
			if n_hiddens[i] is tuple
				then arr[0] is channel num and arr[1] is output_dim
			if n_hiddens[i] is scalar
				then n_hiddens[i] is output dim
		"""
		self.n_in = n_in
		self.n_out = n_out
		self.n_hiddens = n_hiddens
		self.unit_list = self.__create_unit_list(n_in, n_out, n_hiddens)
		self.dropout = dropout
		self.batch_size = batch_size
		self.every_eval = every_eval
		self.logger = logger
		if self.logger:
			self.logger.info("unit_list: {}".format(self.unit_list))

	def __create_unit_list(self, n_in, n_out, n_hiddens):
		input_dim = n_in
		unit_list = []
		for n_hidden in n_hiddens:
			if type(n_hidden) is list:
				if len(n_hidden) == 2:
					unit_list.append([input_dim, n_hidden[0], n_hidden[1]])
					input_dim = n_hidden[1]
				elif len(n_hidden) == 1:
					unit_list.append([input_dim, n_hidden[0]])
					input_dim = n_hidden[0]
				else:
					raise TypeError("n_hiddens must be shape like following \n e.g. [[1,2], [3, 4], [2], 1]")
			elif type(n_hidden) is int or type(n_hidden) is float:
				unit_list.append([input_dim, n_hidden])
				input_dim = n_hidden
			else:
				raise TypeError("n_hiddens must be shape like following \n e.g. [[1,2], [3, 4], [2], 1]")
		unit_list.append([input_dim, n_out])
		return unit_list

	def _weight_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _bias_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _maxout_layer(self, input_tensor, input_dim, channel_dim, output_dim, layer_name):
		with tf.name_scope(layer_name):
			with tf.name_scope("W"):
				W = self._weight_variable([input_dim, channel_dim])
			with tf.name_scope("b"):
				b = self._bias_variable([channel_dim])
			with tf.name_scope("Wx_plus_b"):
				z = tf.matmul(input_tensor, W) + b
			with tf.name_scope("activation"):
				activation = tf.contrib.layers.maxout(z, num_units=output_dim)
			return activation

	def _fullconect_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softmax):
		with tf.name_scope(layer_name):
			with tf.name_scope("W"):
				W = self._weight_variable([input_dim, output_dim])
			with tf.name_scope("b"):
				b = self._bias_variable([output_dim])
			with tf.name_scope("Wx_plus_b"):
				z = tf.matmul(input_tensor, W) + b
			with tf.name_scope("activation"):
				activation = act(z)
			return activation

	def inference(self):
		self.x = tf.placeholder(tf.float32, shape=(None, self.n_in), name="x")
		self.keep_prob = tf.placeholder(tf.float32, name="drop_out")
		self.t = tf.placeholder(tf.float32, shape=(None, self.n_out), name="target")
		self.activations = {}

		self.activations[0] = self.x
		for i, units in enumerate(self.unit_list):
			name = "layer" + str(i)
			if len(units) == 3:
				activation = self._maxout_layer(self.activations[i], units[0], units[1], units[2],
					layer_name=name)
			else:
				activation = self._fullconect_layer(self.activations[i], units[0], units[1],
					layer_name=name)
			dropped = tf.nn.dropout(activation, self.keep_prob)
			self.activations[i + 1] = dropped
		return self.activations[len(self.unit_list)]

	def loss(self, logits):
		with tf.name_scope("loss"):
			loss = tf.reduce_mean(-tf.reduce_sum(self.t * tf.log(logits), reduction_indices=[1]))
		return loss

	def accuracy(self, pred):
		with tf.name_scope("acc"):
			with tf.name_scope("correct_pred"):
				correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.t, 1))
			with tf.name_scope("acc"):
				acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		return acc

	def _training(self, loss):
		with tf.name_scope("train"):
			train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
		return train_step

	def feed_dict(self, train):
		if train:
			xs, ys = mnist.train.next_batch(self.batch_size)
			k = self.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {self.x: xs, self.t: ys, self.keep_prob: k}

	def train_mnist(self, max_iter=10):
		with tf.Graph().as_default():
			# network initialize
			logits = self.inference()
			# loss
			loss = self.loss(logits)
			tf.summary.scalar("loss", loss)
			# accuracy
			accuracy = self.accuracy(logits)
			tf.summary.scalar("acc", accuracy)

			train_step = self._training(loss)

			merged = tf.summary.merge_all()

			init = tf.global_variables_initializer()

			with tf.Session() as sess:
				if os.path.exists("./summary"):
					shutil.rmtree("./summary")

				train_writer = tf.summary.FileWriter("./summary" + '/train',
					sess.graph)
				test_writer = tf.summary.FileWriter("./summary" + '/test')
				sess.run(init)

				for i in range(max_iter):
					total_batch = int(mnist.train.num_examples / self.batch_size)
					total_loss = 0
					for _ in range(total_batch):
						summary, _, loss_tmp = sess.run([merged, train_step, loss],
							feed_dict=self.feed_dict(True))
						train_writer.add_summary(summary, i)
						total_loss += loss_tmp
					if i % self.every_eval == 0:
						summary, acc_tmp = sess.run([merged, accuracy],
							feed_dict=self.feed_dict(False))
						test_writer.add_summary(summary, i)
						if self.logger:
							self.logger.info("step: {}, acc: {}".format(i, acc_tmp))
							self.logger.info("loss: {}".format(total_loss / total_batch))

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger("test")
	logger.info("start")

	tf.reset_default_graph()
	n_in = mnist.test.images.shape[1]
	n_out = mnist.test.labels.shape[1]
	maxout_net = MaxOut(n_in, n_out, [[100, 50]], 1.0, every_eval=10, logger=logger)
	maxout_net.train_mnist(max_iter=100)
