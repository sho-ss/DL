import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, fullconect_units=[256, 1], channels=[1, 64, 128], widths=[28, 14, 7],
		kernels=[5, 5], strides=[2, 2],
		w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		b_initializer=tf.zeros_initializer()):

		# convolution's channels
		self.in_channels = channels[:-1]
		self.out_channels = channels[1:]
		self.n_layer = len(channels) - 1

		# full conect layer's units
		fullconect_units.insert(0, widths[-1] * widths[-1] * self.out_channels[-1])
		self.in_dense_units = fullconect_units[:-1]
		self.out_dense_units = fullconect_units[1:]
		self.n_dense_layer = len(fullconect_units) - 1

		# image widths each layer
		self.widths = widths
		# conv's kernel size
		self.kernels = kernels
		# conv's stride size
		self.strides = strides

		# define param initializer
		#self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.w_initializer = w_initializer
		self.b_initializer = b_initializer

	def _global_average_pool(self, x):
		for _ in range(2):
			x = tf.reduce_mean(x, axis=[1])
		return x

	def inference(self, input_tensor, reuse=False):
		activation = input_tensor
		for i in range(self.n_layer):
			with tf.variable_scope("conv{}".format(i), reuse=reuse):
				w_shape = [self.kernels[i], self.kernels[i], self.in_channels[i], self.out_channels[i]]
				# if variable that have same scope exits, get that variable(reuse)
				W = tf.get_variable("weights", shape=w_shape,
					dtype=tf.float32, initializer=self.w_initializer)
				b = tf.get_variable("bias", shape=[self.out_channels[i]],
					dtype=tf.float32, initializer=self.b_initializer)

				z = tf.nn.conv2d(activation, W, strides=[1, self.strides[i], self.strides[i], 1],
					padding="SAME")

				z = tf.nn.bias_add(z, b)
				# used batch norm except first layer
				if i != 0:
					with tf.name_scope("batch_norm"):
						mean, variance = tf.nn.moments(z, axes=[0, 1, 2])
						z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-05)
				with tf.name_scope("leaky_relu"):
					activation = tf.nn.leaky_relu(z)
		with tf.name_scope("to_vec"):
			activation = tf.layers.flatten(activation)
		for i in range(self.n_dense_layer):
			with tf.variable_scope("dense{}".format(i + self.n_layer), reuse=reuse):
				w_shape = [self.in_dense_units[i], self.out_dense_units[i]]
				W = tf.get_variable("weights", shape=w_shape,
					dtype=tf.float32, initializer=self.w_initializer)
				b = tf.get_variable("bias", shape=[self.out_dense_units[i]],
					dtype=tf.float32, initializer=self.b_initializer)
				z = tf.matmul(activation, W) + b
				with tf.name_scope("batch_norm"):
					mean, variance = tf.nn.moments(z, axes=[0])
					z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-05)

				if i == self.n_dense_layer - 1:
					with tf.name_scope("sigmoid"):
						activation = tf.nn.sigmoid(z)
				else:
					with tf.name_scope("leaky_relu"):
						activation = tf.nn.leaky_relu(z)

		return activation
		"""
				w_shape = [1, 1, self.out_channels[-1], self.out_dense_units[-1]]
				# if variable that have same scope exits, get that variable(reuse)
				W = tf.get_variable("weights", shape=w_shape,
					dtype=tf.float32, initializer=self.w_initializer)
				#b = tf.get_variable("bias", shape=[self.out_dim],
				#	dtype=tf.float32, initializer=self.b_initializer)
				with tf.name_scope("global_avg_pool"):
					z = tf.nn.conv2d(activation, W, strides=[1, 1, 1, 1], padding="SAME")
					z = tf.reduce_mean(z, [1, 2])
				with tf.name_scope("sigmoid"):
					activation = tf.nn.sigmoid(z)

			activation = tf.layers.flatten(activation)
			z = tf.matmul(activation, W) + b
			with tf.name_scope("batch_norm"):
				mean, variance = tf.nn.moments(z, axes=[0])
				z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-08)
			with tf.name_scope("sigmoid"):
				activation = tf.nn.sigmoid(z)

		return activation
		"""


def main():
	# summary dir
	__summary_dir = "./summary/cnn"
	xs = np.zeros(shape=50 * 28 * 28)
	xs = np.reshape(xs, newshape=(50, 28, 28, 1))

	with tf.Graph().as_default():
		with tf.name_scope("Discriminator"):
			conv_net = CNNonlyConv()
			with tf.name_scope("input"):
				shape = np.append([None], xs.shape[1:])
				x = tf.placeholder(tf.float32, shape=shape)
			logit = conv_net.inference(x)

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
			ans = sess.run(logit, feed_dict={x: xs})
			print(ans.shape)


if __name__ == '__main__':
	main()
