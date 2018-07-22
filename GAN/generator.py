########################################################
# used for Generative adversarial framework's Generator
########################################################

import tensorflow as tf
import numpy as np
import os
import shutil
import logging
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Generator():
	def __init__(self, n_in, n_out, n_hidden, batch_size=100, every_eval=10, logger=None):
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.every_eval = every_eval
		self.logger = logger

	def _weight_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _bias_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _generator_layer(self, input_tensor, layer_name, n_in, n_out, act=tf.nn.relu):
		with tf.name_scope(layer_name):
			with tf.name_scope("W"):
				W = self._weight_variable(shape=[n_in, n_out])
			with tf.name_scope("bias"):
				b = self._bias_variable(shape=[n_out])
			with tf.name_scope("Wx_plus_b"):
				z = tf.matmul(input_tensor, W) + b
			with tf.name_scope("activation"):
				a = act(z)
			return a

	def inference(self):
		# create 3layer NN
		n_layer = 3
		units = [self.n_in, self.n_hidden, self.n_out]
		with tf.name_scope("Generator"):
			# define placeholder
			self.x = tf.placeholder(tf.float32, shape=[None, self.n_in])
			input_tensor = self.x
			activation = None
			for i in range(n_layer - 1):
				# only last layer's act func is sigmoid
				# other relu
				if i == (n_layer - 2):
					act = tf.nn.sigmoid
				else:
					act = tf.nn.relu
				layer_name = "layer" + str(i)
				activation = self._generator_layer(input_tensor, layer_name=layer_name,
					n_in=units[i], n_out=units[i + 1], act=act)
				input_tensor = activation
		return activation
