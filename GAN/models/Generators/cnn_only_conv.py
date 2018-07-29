import tensorflow as tf
import numpy as np
import os
import shutil


class CNNonlyConv():
	def __init__(self, channels=[256, 128, 64, 1], widths=[3, 5, 10, 28], kernels=[2, 2, 3]):
		# inputs dim
		self.in_channels = channels[:-1]
		self.out_channels = channels[1:]
		self.n_layer = len(channels) - 1
		# image widths each layer
		self.widths = widths
		# conv's kernel size also stride size too
		self.kernels = kernels
		# image width in first layer
		self.first_width = 3
		self.kernel_size = 2
		self.kernel_size_last = 3

		self.dim_layer1 = 256
		self.dim_layer2 = 128
		self.dim_layer3 = 64
		self.dim_layer4 = 1

		# define param initializer
		self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.b_initializer = tf.zeros_initializer()

		# define batch norm trainable var
		self.is_training = tf.placeholder(tf.bool, name="bn_bool")

	def _weight_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def _bias_variable(self, shape):
		return tf.Variable(self.initializer(shape=shape))

	def inference(self, input_tensor, name):
		with tf.name_scope(name):
			batch_size = tf.shape(input_tensor)[0]
			with tf.name_scope("layer0_dense"):
				units = self.widths[0] * self.widths[0] * self.in_channels[0]
				z = tf.layers.dense(input_tensor, units=units, kernel_initializer=self.initializer)
				z = tf.reshape(z, shape=[-1, self.widths[0], self.widths[0], self.in_channels[0]])
				z_bn = tf.layers.batch_normalization(z, training=self.is_training)
				activation = tf.nn.relu(z_bn)
			for i in range(self.n_layer):
				with tf.variable_scope("conv{}".format(i + 1)):
					# get weights and bias
					w_shape = [self.kernels[i], self.kernels[i], self.out_channels[i], self.in_channels[i]]
					W = tf.get_variable("weights", shape=w_shape,
						dtype=tf.float32, initializer=self.w_initializer)
					b = tf.get_variable("bias", shape=[self.out_channels[i]])
					# conv
					z = tf.nn.conv2d_transpose(activation, W,
						output_shape=[batch_size, self.widths[i + 1], self.widths[i + 1], self.out_channels[i]],
						strides=[1, self.kernels[i], self.kernels[i], 1])
					z = tf.nn.bias_add(z, b)
					# use batch norm except last layer
					# activation is relu except last layer, last layer's act is sigmoid
					if i != self.n_layer - 1:
						with tf.name_scope("batch_norm"):
							z = tf.layers.batch_normalization(z, training=self.is_training)
						with tf.name_scope("activation"):
							activation = tf.nn.relu(z)
					else:
						with tf.name_scope("activation"):
							activation = tf.nn.sigmoid(z)
		return activation


def main():
	# summary dir
	__summary_dir = "./summary/cnn"
	dim_noise = 100
	mean = np.zeros(dim_noise)
	cov = np.eye(dim_noise)
	z = np.random.multivariate_normal(mean, cov, 50)

	with tf.Graph().as_default():
		x = tf.placeholder(tf.float32, shape=(None, dim_noise))
		with tf.name_scope("Generator"):
			conv_net = CNNonlyConv()
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
			ans = sess.run(logit, feed_dict={x: z, conv_net.is_training: False})
			print(ans.shape)

if __name__ == '__main__':
	main()
