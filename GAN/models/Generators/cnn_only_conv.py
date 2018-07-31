import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, fullconect_units=[100, 1024], channels=[128, 64, 1], widths=[7, 14, 28],
		kernels=[5, 5], strides=[2, 2],
		w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		b_initializer=tf.zeros_initializer()):
		# full conect layer's units
		fullconect_units.append(widths[0] * widths[0] * channels[0])
		self.in_dense_units = fullconect_units[:-1]
		self.out_dense_units = fullconect_units[1:]
		# convolution's channels
		self.in_channels = channels[:-1]
		self.out_channels = channels[1:]
		# layer num
		self.n_dense_layer = len(fullconect_units) - 1
		self.n_layer = len(channels) - 1
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

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

	def inference(self, input_tensor):
		batch_size = tf.shape(input_tensor)[0]
		activation = input_tensor
		for i in range(self.n_dense_layer):
			with tf.variable_scope("dense{}".format(i)):
				w_shape = [self.in_dense_units[i], self.out_dense_units[i]]
				W = tf.get_variable("weights", shape=w_shape,
					dtype=tf.float32, initializer=self.w_initializer)
				b = tf.get_variable("bias", shape=[self.out_dense_units[i]],
						dtype=tf.float32, initializer=self.b_initializer)
				z = tf.matmul(activation, W) + b
				# batch normalization
				with tf.name_scope("batch_norm"):
					mean, variance = tf.nn.moments(z, axes=[0])
					z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-05)
				with tf.name_scope("relu"):
					activation = tf.nn.relu(z)

				# output that last of dense layer, convert to image
				if i == self.n_dense_layer - 1:
					with tf.name_scope("to_image"):
						activation = tf.reshape(activation, shape=[-1, self.widths[0], self.widths[0], self.in_channels[0]])

		for i in range(self.n_layer):
			with tf.variable_scope("conv{}".format(i + self.n_dense_layer)):
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
						z = tf.nn.batch_normalization(z, mean, variance, None, None, 1e-05)
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
			conv_net = CNNonlyConv()
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
