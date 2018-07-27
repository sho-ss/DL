import tensorflow as tf
import numpy as np
import logging
import os
import shutil


class ReluNet():
	def __init__(self, n_in, n_out, n_hiddens, dropout,
		initializer=tf.contrib.layers.xavier_initializer_conv2d()):
		self.n_in = n_in
		self.n_out = n_out
		self.n_hiddens = n_hiddens
		self.dropout = dropout
		self.keep_prob = tf.placeholder(tf.float32, name="drop_out")
		self.initializer = initializer
		self.W, self.b = self.__init_variable()

	def __init_variable(self):
		units = [self.n_in]
		units.extend(self.n_hiddens)
		units.append(self.n_out)
		self.n_layer = len(units) - 1
		W = []
		b = []
		with tf.name_scope("Discriminator"):
			for i in range(self.n_layer):
				layer_name = "layer" + str(i)
				with tf.name_scope(layer_name):
					with tf.name_scope("W"):
						W_tmp = self._weight_variable(shape=[units[i], units[i + 1]])
					with tf.name_scope("bias"):
						b_tmp = self._bias_variable(shape=[units[i + 1]])
				W.append(W_tmp)
				b.append(b_tmp)
		return W, b

	def _weight_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def _bias_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def inference(self, input_tensor, name):
		# create NN
		with tf.name_scope(name):
			for i in range(self.n_layer):
				# only last layer's act func is sigmoid
				# other relu
				with tf.name_scope("layer" + str(i)):
					with tf.name_scope("Wx_plus_b"):
						z = tf.matmul(input_tensor, self.W[i]) + self.b[i]
					# last layer without dropout, others with dropout
					if i == (self.n_layer - 1):
						with tf.name_scope("activation"):
							activation = tf.nn.sigmoid(z)
					else:
						with tf.name_scope("dropout"):
							z_dropped = tf.nn.dropout(z, keep_prob=self.keep_prob)
						with tf.name_scope("activation"):
							activation = tf.nn.leaky_relu(z_dropped)
					input_tensor = activation
		return activation
