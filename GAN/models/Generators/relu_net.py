########################################################
# used for Generative adversarial framework's Generator
########################################################

import tensorflow as tf
import numpy as np
import os
import shutil
import logging


class ReluNet():
	def __init__(self, n_in, n_out, n_hiddens):
		self.n_in = n_in
		self.n_out = n_out
		self.n_hiddens = n_hiddens

	def _weight_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _bias_variable(self, shape):
		initializer = tf.contrib.layers.xavier_initializer_conv2d()
		return tf.Variable(initializer(shape=shape))

	def _generator_layer(self, input_tensor, layer_name, n_in, n_out, act=tf.nn.relu, apply_bn=True):
		with tf.name_scope(layer_name):
			with tf.name_scope("W"):
				W = self._weight_variable(shape=[n_in, n_out])
			with tf.name_scope("bias"):
				b = self._bias_variable(shape=[n_out])
			with tf.name_scope("Wx_plus_b"):
				z = tf.matmul(input_tensor, W) + b
			if apply_bn:
				with tf.name_scope("batch_norm"):
					z_bn = tf.layers.batch_normalization(z, training=self.is_training)
			with tf.name_scope("activation"):
				if apply_bn:
					a = act(z_bn)
				else:
					a = act(z)
			return a

	def inference(self):
		# create 3layer NN
		units = [self.n_in]
		units.extend(self.n_hiddens)
		units.append(self.n_out)
		n_layer = len(units) - 1
		with tf.name_scope("Generator"):
			# define placeholder
			self.x = tf.placeholder(tf.float32, shape=[None, self.n_in], name="input")
			# used for batch normalization
			self.is_training = tf.placeholder(tf.bool, name="bn_fool")
			input_tensor = self.x
			activation = None
			for i in range(n_layer):
				# only last layer's act func is sigmoid
				# other relu
				if i == (n_layer - 1):
					act = tf.nn.sigmoid
					apply_bn = False
				else:
					act = tf.nn.leaky_relu
					apply_bn = True
				layer_name = "layer" + str(i)
				activation = self._generator_layer(input_tensor, layer_name=layer_name,
					n_in=units[i], n_out=units[i + 1], act=act, apply_bn=apply_bn)
				input_tensor = activation
		return activation
