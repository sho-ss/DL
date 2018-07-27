import tensorflow as tf
import numpy as np
import logging
import os
import shutil


class MaxOutNet():
	def __init__(self, n_in, n_out, n_channels, n_hiddens, dropout,
		initializer=tf.contrib.layers.xavier_initializer_conv2d()):
		self.n_in = n_in
		self.n_out = n_out
		self.n_channels = n_channels
		self.n_hiddens = n_hiddens
		self.n_channels = n_channels
		self.n_layer = len(self.n_channels) + 1
		self.dropout = dropout
		self.initializer = initializer

		self.W, self.b = self.__init_variable()

	def __init_variable(self):
		W = []
		b = []
		with tf.name_scope("Discriminator"):
			for i in range(self.n_layer):
				layer_name = "layer" + str(i)
				if i == 0:
					n_in = self.n_in
					n_out = self.n_channels[i]
				elif i == self.n_layer - 1:
					n_in = self.n_hiddens[i - 1]
					n_out = self.n_out
				else:
					n_in = self.n_hiddens[i - 1]
					n_out = self.n_channels[i]
				# get variable
				with tf.name_scope(layer_name):
					with tf.name_scope("W_maxout"):
						W_tmp = self.__weight_variable([n_in, n_out])
					with tf.name_scope("b_maxout"):
						b_tmp = self.__bias_variable([n_out])

				W.append(W_tmp)
				b.append(b_tmp)
		return W, b

	def __weight_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def __bias_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def inference(self, input_tensor, name):
		with tf.name_scope(name):
			self.keep_prob = tf.placeholder(tf.float32, name="drop_out")
			for i in range(self.n_layer):
				layer_name = "layer" + str(i)
				with tf.name_scope(layer_name):
					if i != self.n_layer - 1:
						with tf.name_scope("Wx_plus_b"):
							z = tf.matmul(input_tensor, self.W[i]) + self.b[i]
						with tf.name_scope("activation"):
							activation = tf.contrib.layers.maxout(z, num_units=self.n_hiddens[i])
						input_tensor = activation
					else:
						with tf.name_scope("Wx_plus_b"):
							z = tf.matmul(input_tensor, self.W[i]) + self.b[i]
						with tf.name_scope("logits"):
							logits = tf.nn.sigmoid(z)
		return logits
