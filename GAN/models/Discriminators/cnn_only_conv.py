import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, out_dim=1, channels=[1, 64, 128, 256], widths=[28, 10, 5, 3], kernels=[3, 2, 2]):
		# convolution's channels
		self.in_channels = channels[:-1]
		self.out_channels = channels[1:]
		self.n_layer = len(channels) - 1

		# image widths each layer
		self.widths = widths
		# conv's kernel size also stride size too
		self.kernels = kernels

		# output dim
		self.out_dim = out_dim

		# define param initializer
		self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.b_initializer = tf.zeros_initializer()

		# define keep prob
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

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

				z = tf.nn.conv2d(activation, W, strides=[1, self.kernels[i], self.kernels[i], 1],
					padding="SAME")

				z = tf.nn.bias_add(z, b)
				# used batch norm except first layer
				if i != 0:
					with tf.name_scope("batch_norm"):
						mean, variance = tf.nn.moments(z, axes=[0, 1, 2])
						z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-08)
				with tf.name_scope("leaky_relu"):
					activation = tf.nn.leaky_relu(z)

		with tf.variable_scope("dense_layer{}".format(self.n_layer), reuse=reuse):
			w_shape = [self.out_channels[-1] * self.widths[-1] * self.widths[-1], self.out_dim]
			# if variable that have same scope exits, get that variable(reuse)
			W = tf.get_variable("weights", shape=w_shape,
				dtype=tf.float32, initializer=self.w_initializer)
			b = tf.get_variable("bias", shape=[self.out_dim],
				dtype=tf.float32, initializer=self.b_initializer)

			activation = tf.layers.flatten(activation)
			z = tf.matmul(activation, W) + b
			with tf.name_scope("batch_norm"):
				mean, variance = tf.nn.moments(z, axes=[0])
				z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-08)
			with tf.name_scope("sigmoid"):
				activation = tf.nn.sigmoid(z)

		return activation


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
			ans = sess.run(logit, feed_dict={x: xs, conv_net.is_training: False, conv_net.keep_prob: 1.0})
			print(ans.shape)


if __name__ == '__main__':
	main()
