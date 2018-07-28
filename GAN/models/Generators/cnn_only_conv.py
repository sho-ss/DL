import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, dim_noise):
		# inputs dim
		self.dim_noise = dim_noise
		self.dim_layer1 = 512
		self.dim_layer2 = 256
		self.dim_layer3 = 128
		self.dim_layer4 = 1

		# define param initializer
		self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

		# initialize weight and bias
		with tf.name_scope("variables_of_dense"):
			self.W1 = self._weight_variable(shape=[self.dim_noise, 4 * 4 * self.dim_layer1])
			self.b1 = self._weight_variable(shape=[4 * 4 * self.dim_layer1])
		# conv weights
		with tf.name_scope("filter_of_conv0"):
			self.W2 = self._weight_variable(shape=[3, 3, self.dim_layer2, self.dim_layer1])
		with tf.name_scope("filter_of_conv1"):
			self.W3 = self._weight_variable(shape=[3, 3, self.dim_layer3, self.dim_layer2])
		with tf.name_scope("filter_of_conv2"):
			self.W4 = self._weight_variable(shape=[3, 3, self.dim_layer4, self.dim_layer3])

	def _weight_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def _bias_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def inference(self, name):
		with tf.name_scope(name):
			with tf.name_scope("input"):
				self.x = tf.placeholder(tf.float32, shape=(None, self.dim_noise))
			batch_size = tf.shape(self.x)[0]
			with tf.name_scope("layer0"):
				z = tf.matmul(self.x, self.W1) + self.b1
				z_ = tf.reshape(z, shape=[-1, 4, 4, self.dim_layer1])
				z_bn = tf.layers.batch_normalization(z_, training=self.is_training)
				activation = tf.nn.relu(z_bn)
			# compute batch size
			#batch_size = tf.shape(activation)[0]
			with tf.name_scope("conv_layer0"):
				z1 = tf.nn.conv2d_transpose(activation, self.W2,
					output_shape=[batch_size, 7, 7, self.dim_layer2], strides=[1, 2, 2, 1])
				z1_bn = tf.layers.batch_normalization(z1, training=self.is_training)
				activation = tf.nn.relu(z1_bn)
			with tf.name_scope("conv_layer1"):
				z2 = tf.nn.conv2d_transpose(activation, self.W3,
					output_shape=[batch_size, 14, 14, self.dim_layer3], strides=[1, 2, 2, 1])
				z2_bn = tf.layers.batch_normalization(z2, training=self.is_training)
				activation = tf.nn.relu(z2_bn)
			with tf.name_scope("conv_layer2"):
				z3 = tf.nn.conv2d_transpose(activation, self.W4,
					output_shape=[batch_size, 28, 28, self.dim_layer4], strides=[1, 2, 2, 1])
				activation = tf.nn.sigmoid(z3)
		return activation


def main():
	# summary dir
	__summary_dir = "./summary/cnn"
	dim_noise = 100
	mean = np.zeros(dim_noise)
	cov = np.eye(dim_noise)
	z = np.random.multivariate_normal(mean, cov, 50)

	with tf.Graph().as_default():
		with tf.name_scope("Generator"):
			conv_net = CNNonlyConv(dim_noise=dim_noise)
			logit = conv_net.inference("inference")

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
			ans = sess.run(logit, feed_dict={conv_net.x: z, conv_net.is_training: False})
			print(ans.shape)

if __name__ == '__main__':
	main()
