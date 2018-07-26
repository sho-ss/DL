import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import os
import logging
import shutil
from generator import ReluNet
from discriminator import MaxOutNetD, ReluNetD
import settings
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.logging.set_verbosity(old_v)


class GAN():
	def __init__(self, dropout, n_hiddens_generator, discriminator_iter=1, lr=.0002, every_eval=10):
		"""
		@param n_hiddens_generator list
			num of generator's hidden unit per layer
		@param discriminator_iter int
			update step of discriminator at each iteration
		"""
		# noise dimention
		self.__noise_d = 100
		# noise prior
		self.__noise_prior = np.random.multivariate_normal
		self.__mean = np.zeros(self.__noise_d)
		self.__cov = np.eye(self.__noise_d)

		self.dropout = dropout
		self.n_hiddens_generator = n_hiddens_generator

		self.discriminator_iter = discriminator_iter

		# learning rate
		self.lr = lr

		# evaluation each every_eval step
		self.every_eval = every_eval

		# params of plotting
		self.plot_cols = settings.plot_cols
		self.plot_rows = settings.plot_rows

	def __save_pkl(self, obj, file_path):
		max_bytes = 2**31 - 1
		# write
		bytes_out = pickle.dumps(obj)
		with open(file_path, 'wb') as f_out:
			for idx in range(0, len(bytes_out), max_bytes):
				f_out.write(bytes_out[idx:idx + max_bytes])

	def __sample_noize(self, sample_size):
		"""
		@return noise samples: ndarray (sample_size, dim)
		"""
		return self.__noise_prior(mean=self.__mean, cov=self.__cov, size=(sample_size))

	def __sample_data(self, sample_size):
		"""
		@return datas: ndarray (sample_size, data_dim)
		"""
		xs, _ = mnist.train.next_batch(sample_size)
		return xs

	def __loss_discriminator(self, logits_d_data, logits_d_gen, size):
		with tf.name_scope("loss_discriminator"):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.log(logits_d_data) +
				tf.log(tf.ones([size, 1], tf.float32) - logits_d_gen), axis=1))
		return loss

	def __loss_generator(self, logits_d_gen):
		with tf.name_scope("loss_generator"):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.log(logits_d_gen), axis=1))
		return loss

	def __train_step(self, loss, var_list, name):
		with tf.name_scope(name):
			train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss, var_list=var_list)
		return train_step

	def __mean_p_data(self, logits_d_data):
		with tf.name_scope("mean_p_data"):
			p_data = tf.reduce_mean(logits_d_data)
		return p_data

	def __print_grads(self, grads_d_val, grads_g_val, d_vars, g_vars):
		print("#gradient_of_D")
		for v, val in zip(d_vars, grads_d_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)
		print("#gradient_of_G")
		for v, val in zip(g_vars, grads_g_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)

	def train(self, sample_size, max_iter, every_samples):
		# used for ploting generative samples
		epochs = []
		samples = []
		self.every_samples = every_samples

		with tf.Graph().as_default():
			# real data
			Xs = mnist.train.images
			generator_dim = Xs.shape[1]

			# define G and D
			generator = ReluNet(n_in=self.__noise_d, n_out=generator_dim, n_hiddens=self.n_hiddens_generator)
			discriminator = ReluNetD(n_in=generator_dim, n_out=1, n_hiddens=[512, 256], dropout=0.0)
			#MaxOutNetD(n_in=generator_dim, n_out=1, n_channels=[600, 50], n_hiddens=[150, 10],
			#	dropout=self.dropout)

			# input placeholder
			x_data = tf.placeholder(tf.float32, shape=(None, generator_dim))
			# used for cal loss_D
			size = tf.placeholder(tf.int32)

			# inference discriminator and generator
			logits_g = generator.inference()
			logits_d_data = discriminator.inference(x_data, name="data_discrim")
			logits_d_gen = discriminator.inference(logits_g, name="gen_discrim")

			# prob of data lather than noise sample
			p_data = self.__mean_p_data(logits_d_data)
			tf.summary.scalar("prob_data", p_data)

			# loss discriminator and generator
			loss_d = self.__loss_discriminator(logits_d_data, logits_d_gen, size)
			loss_g = self.__loss_generator(logits_d_gen)
			tf.summary.scalar("loss_D", loss_d)
			tf.summary.scalar("loss_G", loss_g)

			# get discriminator vars
			discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
			# get generator vars
			generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
			# train_step discriminator and generator
			train_step_d = self.__train_step(loss_d, var_list=discriminator_vars, name="train_discriminator")
			train_step_g = self.__train_step(loss_g, var_list=generator_vars, name="train_generator")

			# get batch norm ops
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			# used for checking training failure or not
			grads_d = tf.gradients(loss_d, discriminator_vars)
			grads_g = tf.gradients(loss_g, generator_vars)
			grad_check_size = 100
			grad_check_data = Xs[0:grad_check_size]
			grad_check_noise = self.__sample_noize(grad_check_size)

			# create test data
			test_size = 10000
			test_noise = self.__sample_noize(test_size)
			test_data = mnist.test.images[0: test_size]

			# create data for plotting
			num_plot = self.plot_cols * self.plot_rows
			plot_noise = self.__sample_noize(num_plot)

			# define summary
			merged = tf.summary.merge_all()

			# define initializer
			init = tf.global_variables_initializer()

			with tf.Session() as sess:
				if os.path.exists("./summary"):
					shutil.rmtree("./summary")
				writer = tf.summary.FileWriter("./summary",
					sess.graph)
				# initialize variables
				sess.run(init)

				for i in range(max_iter):
					# update step of discriminator
					for k in range(self.discriminator_iter):
						sample_noise = self.__sample_noize(sample_size)
						sample_data = self.__sample_data(sample_size)

						sess.run(train_step_d,
							feed_dict={x_data: sample_data,
										generator.x: sample_noise,
										generator.is_training: False,
										size: sample_size,
										discriminator.keep_prob: 1.0 - discriminator.dropout})

					# update step of generator
					sample_noise = self.__sample_noize(sample_size)
					sess.run([train_step_g, extra_update_ops],
						feed_dict={generator.x: sample_noise,
									generator.is_training: True,
									discriminator.keep_prob: 1.0})

					# eval likelihood
					if i % self.every_eval == 0:
						# eval norm of gradients
						grads_d_val, grads_g_val = sess.run([grads_d, grads_g],
							feed_dict={x_data: grad_check_data,
										generator.x: grad_check_noise,
										generator.is_training: False,
										size: grad_check_size,
										discriminator.keep_prob: 1.0})
						print("##################################################")
						self.__print_grads(grads_d_val, grads_g_val, discriminator_vars, generator_vars)
						# eval test stats
						summary, loss_d_tmp, loss_g_tmp = sess.run([merged, loss_d, loss_g],
							feed_dict={x_data: test_data,
										generator.x: test_noise,
										generator.is_training: False,
										size: test_size,
										discriminator.keep_prob: 1.0})
						writer.add_summary(summary, i)
						print("epoch: {}, loss D: {:.3f}, loss G: {:.3f}".format(i, loss_d_tmp, loss_g_tmp))
						print("##################################################")

					if i % self.every_samples == 0:
						epochs.append(i)
						sample_gen = sess.run(logits_g, feed_dict={generator.x: plot_noise, generator.is_training: False})
						samples.append(sample_gen)

		# save datas
		self.__save_pkl(epochs, "./output/epochs.pkl")
		self.__save_pkl(samples, "./output/samples.pkl")

	def check(self):
		with tf.Graph().as_default():
			generator_dim = mnist.train.images.shape[1]

			# define G and D
			generator = ReluNet(n_in=self.__noise_d, n_out=generator_dim, n_hiddens=self.n_hiddens_generator)
			discriminator = ReluNetD(n_in=generator_dim, n_out=1, n_hiddens=[512, 256], dropout=.5)
			#MaxOutNetD(n_in=generator_dim, n_out=1, n_channel=50, n_hidden=10, dropout=self.dropout)
			logits_g = generator.inference()

			# get discriminator vars
			discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
			# get generator vars
			generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")

			print("D")
			for v in discriminator_vars:
				print(v.name)
			print("G")
			for v in generator_vars:
				print(v.name)

			# get batch norm ops
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print("Extra")
			print(extra_update_ops)


def main():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	tf.reset_default_graph()
	gan = GAN(dropout=0.5, n_hiddens_generator=[256], lr=0.001, every_eval=5000)
	gan.train(sample_size=50, max_iter=100000, every_samples=5000)


def test():
	tf.reset_default_graph()
	gan = GAN(dropout=0.5, n_hiddens_generator=[256])
	gan.check()

if __name__ == '__main__':
	#test()
	main()
