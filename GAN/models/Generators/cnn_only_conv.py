import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, in_dim, channels=[256, 128, 64, 1], widths=[3, 5, 10, 28],
		kernels=[2, 2, 3], strides=[2, 2, 3]):
		# input dimention
		self.in_dim = in_dim
		# convolution's channels
		self.in_channels = channels[:-1]
		self.out_channels = channels[1:]
		self.n_layer = len(channels) - 1
		# image widths each layer
		self.widths = widths
		# conv's kernel size
		self.kernels = kernels
		# conv's stride size
		self.strides = strides

		# define param initializer
		#self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.w_initializer = tf.contrib.layers.xavier_initializer_conv2d()
		self.b_initializer = tf.zeros_initializer()

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

	def inference(self, input_tensor):
		batch_size = tf.shape(input_tensor)[0]
		with tf.variable_scope("layer0_dense"):
			units = self.widths[0] * self.widths[0] * self.in_channels[0]
			w_shape = [self.in_dim, units]
			W = tf.get_variable("weights", shape=w_shape,
				dtype=tf.float32, initializer=self.w_initializer)
			b = tf.get_variable("bias", shape=[units],
					dtype=tf.float32, initializer=self.b_initializer)
			z = tf.matmul(input_tensor, W) + b
			z = tf.reshape(z, shape=[-1, self.widths[0], self.widths[0], self.in_channels[0]])
			with tf.name_scope("batch_norm"):
				#z = tf.layers.batch_normalization(z, training=self.is_training)
				mean, variance = tf.nn.moments(z, axes=[0])
				z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-08)
			with tf.name_scope("relu"):
				activation = tf.nn.relu(z)
		for i in range(self.n_layer):
			with tf.variable_scope("conv{}".format(i + 1)):
				# get weights and bias
				w_shape = [self.kernels[i], self.kernels[i], self.out_channels[i], self.in_channels[i]]
				W = tf.get_variable("weights", shape=w_shape,
					dtype=tf.float32, initializer=self.w_initializer)
				b = tf.get_variable("bias", shape=[self.out_channels[i]],
					dtype=tf.float32, initializer=self.b_initializer)
				# conv
				z = tf.nn.conv2d_transpose(activation, W,
					output_shape=[batch_size, self.widths[i + 1], self.widths[i + 1], self.out_channels[i]],
					strides=[1, self.strides[i], self.strides[i], 1])
				z = tf.nn.bias_add(z, b)
				# use batch norm except last layer
				# activation is relu except last layer, last layer's act is sigmoid
				if i != self.n_layer - 1:
					with tf.name_scope("batch_norm"):
						#z = tf.layers.batch_normalization(z, training=self.is_training)
						mean, variance = tf.nn.moments(z, axes=[0, 1, 2])
						z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-08)
					with tf.name_scope("relu"):
						activation = tf.nn.relu(z)
				else:
					with tf.name_scope("sigmoid"):
						activation = tf.nn.sigmoid(z)
		return activation


def main():
	# summary dir
	__summary_dir = "./summary/cnn"
	dim_noise = 100
	mean = np.zeros(dim_noise)
	cov = np.eye(dim_noise)
	z = np.random.multivariate_normal(mean, cov, 50)
	tf.reset_default_graph()
	with tf.Graph().as_default():
		x = tf.placeholder(tf.float32, shape=(None, dim_noise))
		with tf.name_scope("Generator"):
			conv_net = CNNonlyConv(in_dim=dim_noise)
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
			ans = sess.run(logit, feed_dict={x: z, conv_net.is_training: False})
			print(ans.shape)

if __name__ == '__main__':
	main()
