import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, h_in, w_in, dim_in):
		# inputs dim
		self.h_in = h_in
		self.w_in = w_in
		self.dim_in = dim_in
		self.dim_layer1 = 16
		self.dim_layer2 = 32
		self.dim_layer3 = 64
		self.filter_size = 3

		# define param initializer
		self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

		# define keep prob
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

		# initialize filter
		with tf.name_scope("filter_of_conv0"):
			self.W1 = self._weight_variable(shape=[self.filter_size, self.filter_size,
				self.dim_in, self.dim_layer1])
		with tf.name_scope("filter_of_conv1"):
			self.W2 = self._weight_variable(shape=[self.filter_size, self.filter_size,
				self.dim_layer1, self.dim_layer2])
		with tf.name_scope("filter_of_conv2"):
			self.W3 = self._weight_variable(shape=[self.filter_size, self.filter_size,
				self.dim_layer2, self.dim_layer3])
		with tf.name_scope("param_of_last_fullconect_layer"):
			self.W4 = self._weight_variable(shape=[self.filter_size, self.filter_size,
				self.dim_layer3, 1])

	def _weight_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def _bias_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def _global_average_pool(self, x):
		for _ in range(2):
			x = tf.reduce_mean(x, axis=[1])
		return x

	def inference(self, input_tensor, name):
		strides = [1, 2, 2, 1]
		with tf.name_scope(name):
			# conpute batch size
			with tf.name_scope("conv_layer0"):
				z = tf.nn.conv2d(input_tensor, self.W1, strides=strides, padding="SAME")
				#z_bn = tf.layers.batch_normalization(z, training=self.is_training)
				z_bn = tf.nn.dropout(z, self.keep_prob)
				activation = tf.nn.leaky_relu(z)
			#batch_size = tf.shape(activation)[0]
			with tf.name_scope("conv_layer1"):
				z1 = tf.nn.conv2d(activation, self.W2, strides=strides, padding="SAME")
				#z1_bn = tf.layers.batch_normalization(z1, training=self.is_training)
				z1_bn = tf.nn.dropout(z1, self.keep_prob)
				activation = tf.nn.leaky_relu(z1)
			with tf.name_scope("conv_layer2"):
				z2 = tf.nn.conv2d(activation, self.W3, strides=strides, padding="SAME")
				#z2_bn = tf.layers.batch_normalization(z2, training=self.is_training)
				z2_bn = tf.nn.dropout(z2, self.keep_prob)
				activation = tf.nn.leaky_relu(z2)
			with tf.name_scope("layer3"):
				z3 = tf.nn.conv2d(activation, self.W4, strides=strides, padding="SAME")
				z3_ = self._global_average_pool(z3)
				activation = tf.nn.sigmoid(z3_)
		return activation


def main():
	# summary dir
	__summary_dir = "./summary/cnn"
	xs = np.zeros(shape=50 * 28 * 28)
	xs = np.reshape(xs, newshape=(50, 28, 28, 1))

	with tf.Graph().as_default():
		with tf.name_scope("Discriminator"):
			conv_net = CNNonlyConv(h_in=28, w_in=28, dim_in=1)
			with tf.name_scope("input"):
				shape = np.append([None], xs.shape[1:])
				x = tf.placeholder(tf.float32, shape=shape)
			logit = conv_net.inference(x, "inference")

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			# remove existing output dir of summary
			if os.path.exists(__summary_dir):
				shutil.rmtree(__summary_dir)
			else:
				os.makedirs(__summary_dir)
			writer = tf.summary.FileWriter(__summary_dir,
				sess.graph)
			sess.run(init)
			ans = sess.run(logit, feed_dict={x: xs, conv_net.is_training: False, conv_net.keep_prob: 1.0})
			print(ans.shape)

if __name__ == '__main__':
	main()
