import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
import logging
import shutil
import argparse
from Discriminators import CNNonlyConv as CNNonlyConvD
from Generators import CNNonlyConv as CNNonlyConvG
import settings
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)
tf.logging.set_verbosity(old_v)


class DCGAN():
	def __init__(self, lr=.0002, dropout=.7, every_eval=10):
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

		# learning rate
		self.lr = lr

		# evaluation each every_eval step
		self.every_eval = every_eval

		# params of plotting
		self.plot_cols = settings.plot_cols
		self.plot_rows = settings.plot_rows

		# output dir of summary
		self.summary_dir = os.getcwd() + "/../output/summary/"
		# dir of samples from generator
		self.generate_samples_dir = os.getcwd() + "/../output/"

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
			train_step = tf.train.AdamOptimizer(self.lr, beta1=.5).minimize(loss, var_list=var_list)
		return train_step

	def __print_grads(self, grads_d_val, grads_g_val, d_vars, g_vars):
		print("#gradient_of_D")
		for v, val in zip(d_vars, grads_d_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)
		print("#gradient_of_G")
		for v, val in zip(g_vars, grads_g_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)

	def train(self, batch_size, max_iter, sample_size, every_samples):
		# used for ploting generative samples
		epochs = []
		samples = []
		self.every_samples = every_samples

		with tf.Graph().as_default():
			# real data
			Xs = mnist.train.images
			n_train_data = len(Xs)
			data_indices = np.arange(n_train_data)

			with tf.name_scope("input"):
				x_data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
			# used for cal loss_D
			with tf.name_scope("data_size"):
				size = tf.placeholder(tf.int32)

			# define G and D
			with tf.name_scope("Generator"):
				generator = CNNonlyConvG(dim_noise=self.__noise_d)
				# inference discriminator and generator
				logits_g = generator.inference(name="inference")
				# get batch norm ops
				extra_update_ops_g = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")
			with tf.name_scope("Discriminator"):
				discriminator = CNNonlyConvD(h_in=28, w_in=28, dim_in=1)
				# predictions
				logits_d_data = discriminator.inference(x_data, name="data_discrim")
				logits_d_gen = discriminator.inference(logits_g, name="gen_discrim")
				# get batch norm ops
				extra_update_ops_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")

			# loss discriminator and generator
			with tf.name_scope("loss"):
				loss_g = self.__loss_generator(logits_d_gen)
				tf.summary.scalar("loss_G", loss_g)
				loss_d = self.__loss_discriminator(logits_d_data, logits_d_gen, size)
				tf.summary.scalar("loss_D", loss_d)

			# train_step discriminator and generator
			with tf.name_scope("train"):
				# get discriminator vars
				discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
				# get generator vars
				generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
				train_step_d = self.__train_step(loss_d, var_list=discriminator_vars, name="train_discriminator")
				train_step_g = self.__train_step(loss_g, var_list=generator_vars, name="train_generator")

			# used for checking training failure or not
			with tf.name_scope("stats_for_check"):
				grads_d = tf.gradients(loss_d, discriminator_vars)
				grads_g = tf.gradients(loss_g, generator_vars)
				grad_check_size = 100
				grad_check_data = Xs[0:grad_check_size]
				grad_check_data = np.reshape(grad_check_data, [grad_check_data.shape[0], 28, 28, 1])
				grad_check_noise = self.__sample_noize(grad_check_size)

			# create test data
			test_size = 10000
			test_noise = self.__sample_noize(test_size)
			test_data = mnist.test.images[0: test_size]
			test_data = np.reshape(test_data, [test_data.shape[0], 28, 28, 1])

			# create data for plotting
			num_plot = self.plot_cols * self.plot_rows
			plot_noise = self.__sample_noize(num_plot)

			# define summary
			merged = tf.summary.merge_all()

			# define initializer
			init = tf.global_variables_initializer()

			with tf.Session() as sess:
				# remove existing output dir of summary
				if os.path.exists(self.summary_dir):
					shutil.rmtree(self.summary_dir)
				else:
					os.makedirs(self.summary_dir)
				writer = tf.summary.FileWriter(self.summary_dir,
					sess.graph)
				# initialize variables
				sess.run(init)

				for i in range(max_iter):
					# shuffle indices
					np.random.shuffle(data_indices)
					# cal num batch iter
					max_iter_batch = -(-n_train_data // batch_size)
					# batch learn
					for i_batch in range(max_iter_batch):
						start = i_batch * batch_size
						end = start + batch_size
						x_batch = Xs[data_indices[start: end]]
						x_batch = np.reshape(x_batch, [x_batch.shape[0], 28, 28, 1])
						noise_batch = self.__sample_noize(batch_size)
						# train Discriminator at even
						# train Generator at odd
						if i_batch % 2 == 0:
							sess.run([train_step_d, extra_update_ops_d],
								feed_dict={x_data: x_batch,
											generator.x: noise_batch,
											discriminator.keep_prob: 1.0 - self.dropout,
											discriminator.is_training: True,
											generator.is_training: False,
											size: batch_size})
						if i_batch % 2 != 0:
							# update step of generator
							sess.run([train_step_g, extra_update_ops_g],
								feed_dict={generator.x: noise_batch,
											discriminator.keep_prob: 1.0,
											generator.is_training: True,
											discriminator.is_training: False})

					# evaluation
					if i % self.every_eval == 0:
						# eval norm of gradients
						grads_d_val, grads_g_val = sess.run([grads_d, grads_g],
							feed_dict={x_data: grad_check_data,
										generator.x: grad_check_noise,
										discriminator.keep_prob: 1.0,
										discriminator.is_training: False,
										generator.is_training: False,
										size: grad_check_size})
						print("##################################################")
						self.__print_grads(grads_d_val, grads_g_val, discriminator_vars, generator_vars)
						# eval test stats
						summary, loss_d_tmp, loss_g_tmp = sess.run([merged, loss_d, loss_g],
							feed_dict={x_data: test_data,
										generator.x: test_noise,
										discriminator.keep_prob: 1.0,
										discriminator.is_training: False,
										generator.is_training: False,
										size: test_size})
						writer.add_summary(summary, i)
						print("epoch: {}, loss D: {:.3f}, loss G: {:.3f}".format(i, loss_d_tmp, loss_g_tmp))
						print("##################################################")

					if i % self.every_samples == 0:
						epochs.append(i)
						sample_gen = sess.run(logits_g, feed_dict={generator.x: plot_noise, generator.is_training: False})
						samples.append(sample_gen)

		# save datas
		self.__save_pkl(epochs, self.generate_samples_dir + "epochs.pkl")
		self.__save_pkl(samples, self.generate_samples_dir + "samples.pkl")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
	parser.add_argument("--dropout", type=float, default=0.7, help="dropout rate")
	parser.add_argument("--batch-size", type=int, default=128, help="num of data per batch")
	parser.add_argument("--max-iter", type=int, default=100, help="num of training of GAN")
	parser.add_argument("--sample-size", type=int, default=50, help="num of samples per iteration")
	parser.add_argument("--every-samples", type=int, default=1, help="sample generated image per this value")
	parser.add_argument("--every-eval", type=int, default=10, help="cal test stats per this value")
	args = parser.parse_args()

	tf.reset_default_graph()
	gan = DCGAN(lr=args.lr, dropout=args.dropout, every_eval=args.every_eval)
	gan.train(batch_size=args.batch_size, max_iter=args.max_iter,
		sample_size=args.sample_size, every_samples=args.every_samples)


if __name__ == '__main__':
	#test()
	main()
