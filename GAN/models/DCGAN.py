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
	def __init__(self, lr=.0002, beta1=.5, dropout=.7, n_discrim_per_update=1, every_eval=10, every_sample=10):
		"""
		@param n_hiddens_generator list
			num of generator's hidden unit per layer
		@param discriminator_iter int
			update step of discriminator at each iteration
		"""
		# noise dimention
		self.dim_noise = 100
		# noise prior
		self._noise_prior = np.random.multivariate_normal
		self._mean = np.zeros(self.dim_noise)
		self._cov = np.eye(self.dim_noise)

		self.dropout = dropout

		# learning rate
		self.lr = lr
		# param of adams
		self.beta1 = beta1

		# evaluation each every_eval step
		self.every_eval = every_eval
		# sampling sapan from generator
		self.every_sample = every_sample

		# params of plotting
		self.plot_cols = settings.plot_cols
		self.plot_rows = settings.plot_rows

		self.discrim_update_num = n_discrim_per_update

		# output dir of summary
		self.summary_dir = os.getcwd() + "/../output/summary/"
		# dir of samples from generator
		self.generate_samples_dir = os.getcwd() + "/../output/"

	def _save_pkl(self, obj, file_path):
		max_bytes = 2**31 - 1
		# write
		bytes_out = pickle.dumps(obj)
		with open(file_path, 'wb') as f_out:
			for idx in range(0, len(bytes_out), max_bytes):
				f_out.write(bytes_out[idx:idx + max_bytes])

	def _sample_noize(self, sample_size):
		"""
		@return noise samples: ndarray (sample_size, dim)
		"""
		return self._noise_prior(mean=self._mean, cov=self._cov, size=(sample_size))

	def _sample_data(self, sample_size):
		"""
		@return datas: ndarray (sample_size, data_dim)
		"""
		xs, _ = mnist.train.next_batch(sample_size)
		return xs

	def _weight_decay(self, weight_vars):
		weight_decay = tf.nn.l2_loss(weight_vars[0])
		for v in weight_vars[1:]:
			weight_decay += tf.nn.l2_loss(v)
		return weight_decay

	def _loss_discriminator(self, logits_d_data, logits_d_gen, size, var_list):
		with tf.name_scope("loss_discriminator"):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.log(logits_d_data) +
				tf.log(tf.ones([size, 1], tf.float32) - logits_d_gen), axis=1))
			#+ lambda_2 * weight_decay
		return loss

	def _loss_generator(self, logits_d_gen, var_list):
		with tf.name_scope("loss_generator"):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.log(logits_d_gen), axis=1))
			#+ lambda_2 * weight_decay
		return loss

	def _train_step_adam(self, loss, var_list, lr, beta1, name):
		with tf.name_scope(name):
			train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(loss, var_list=var_list)
		return train_step

	def _extract_weight_bias(self, var_list):
		ans = []
		for v in var_list:
			if "batch_normalization" not in v.name:
				ans.append(v)
		return ans

	def _train_step_sgd(self, loss, var_list, lr, name):
		with tf.name_scope(name):
			train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=var_list)
		return train_step

	def _print_grads(self, grads_d_val, grads_g_val, d_vars, g_vars):
		print("#gradient_of_D")
		for v, val in zip(d_vars, grads_d_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)
		print("#gradient_of_G")
		for v, val in zip(g_vars, grads_g_val):
			norm = np.linalg.norm(val)
			print(v.name + ": ", norm)

	def train(self, batch_size, max_iter):
		# used for ploting generative samples
		epochs = []
		samples = []

		with tf.Graph().as_default():
			# real data
			Xs = mnist.train.images
			n_train_data = len(Xs)
			data_indices = np.arange(n_train_data)

			with tf.name_scope("input"):
				input_data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="data")
				input_noise = tf.placeholder(tf.float32, shape=(None, self.dim_noise), name="noise")
			# used for cal loss_D
			with tf.name_scope("data_size"):
				size = tf.placeholder(tf.int32)

			# define G and D
			with tf.variable_scope("Generator"):
				generator = CNNonlyConvG(fullconect_units=[self.dim_noise, 1024],
					channels=[128, 64, 1], widths=[7, 14, 28],
					kernels=[5, 5], strides=[2, 2],
					w_initializer=tf.random_normal_initializer(0.0, 0.02))
				# inference discriminator and generator
				with tf.name_scope("inference"):
					logits_g = generator.inference(input_noise)
			with tf.variable_scope("Discriminator"):
				discriminator = CNNonlyConvD(fullconect_units=[256, 1],
					channels=[1, 64, 128], widths=[28, 14, 7],
					kernels=[5, 5], strides=[2, 2],
					w_initializer=tf.random_normal_initializer(0.0, 0.02))
				with tf.name_scope("inference_data"):
					logits_d_data = discriminator.inference(input_data, reuse=False)
				with tf.name_scope("inference_sample"):
					logits_d_gen = discriminator.inference(logits_g, reuse=True)

			# get D, G vars respectively
			discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
			generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
			generator_vars = self._extract_weight_bias(generator_vars)

			# get batch norm ops
			extra_update_ops_g = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")

			# loss discriminator and generator
			with tf.name_scope("loss"):
				loss_g = self._loss_generator(logits_d_gen, generator_vars)
				tf.summary.scalar("loss_G", loss_g)
				loss_d = self._loss_discriminator(logits_d_data, logits_d_gen, size,
					discriminator_vars)
				tf.summary.scalar("loss_D", loss_d)

			# train_step discriminator and generator
			with tf.name_scope("train"):
				train_step_d = self._train_step_adam(loss_d, var_list=discriminator_vars,
					lr=self.lr, beta1=self.beta1, name="train_discriminator")
				train_step_g = self._train_step_adam(loss_g, var_list=generator_vars,
					lr=self.lr, beta1=self.beta1, name="train_generator")

			# used for checking training failure or not
			with tf.name_scope("stats_for_check"):
				grads_d = tf.gradients(loss_d, discriminator_vars)
				grads_g = tf.gradients(loss_g, generator_vars)
				grad_check_size = 100
				grad_check_data = Xs[0: grad_check_size]
				grad_check_data = np.reshape(grad_check_data, [grad_check_data.shape[0], 28, 28, 1])
				grad_check_noise = self._sample_noize(grad_check_size)

			# create test data
			test_size = 10000
			test_noise = self._sample_noize(test_size)
			test_data = mnist.test.images[0: test_size]
			test_data = np.reshape(test_data, [test_data.shape[0], 28, 28, 1])

			# create data for plotting
			num_plot = self.plot_cols * self.plot_rows
			plot_noise = self._sample_noize(num_plot)

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
					discrim_update_count = 0
					# batch learn
					for i_batch in range(max_iter_batch):
						start = i_batch * batch_size
						end = start + batch_size
						x_batch = Xs[data_indices[start: end]]
						x_batch_size = x_batch.shape[0]
						x_batch = np.reshape(x_batch, [x_batch_size, 28, 28, 1])
						noise_batch = self._sample_noize(x_batch_size)
						# train Discriminator at even
						# train Generator at odd
						if discrim_update_count != self.discrim_update_num:
							sess.run([train_step_d],
								feed_dict={input_data: x_batch,
											input_noise: noise_batch,
											size: x_batch_size,
											generator.is_training: False})
							discrim_update_count += 1
						else:
							# update step of generator
							sess.run([train_step_g, extra_update_ops_g],
								feed_dict={input_noise: noise_batch,
											generator.is_training: True})
							discrim_update_count = 0

					# evaluation
					if i % self.every_eval == 0 or i == max_iter - 1:
						grads_d_val, grads_g_val = sess.run([grads_d, grads_g],
							feed_dict={input_data: grad_check_data,
										input_noise: grad_check_noise,
										size: grad_check_size,
										generator.is_training: False})
						print("##################################################")
						self._print_grads(grads_d_val, grads_g_val, discriminator_vars, generator_vars)
						# eval test stats
						summary, loss_d_tmp, loss_g_tmp = sess.run([merged, loss_d, loss_g],
							feed_dict={input_data: test_data,
										input_noise: test_noise,
										size: test_size,
										generator.is_training: False})
						writer.add_summary(summary, i)
						print("epoch: {}, loss D: {:.3f}, loss G: {:.3f}".format(i, loss_d_tmp, loss_g_tmp))
						print("##################################################")

					if i % self.every_sample == 0 or i == max_iter - 1:
						epochs.append(i)
						sample_gen = sess.run(logits_g, feed_dict={input_noise: plot_noise, generator.is_training: False})
						samples.append(sample_gen)
						# save datas
						self._save_pkl(epochs, self.generate_samples_dir + "epochs.pkl")
						self._save_pkl(samples, self.generate_samples_dir + "samples.pkl")

		# save datas
		self._save_pkl(epochs, self.generate_samples_dir + "epochs.pkl")
		self._save_pkl(samples, self.generate_samples_dir + "samples.pkl")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
	parser.add_argument("--beta1", type=float, default=0.5, help="learning rate")
	parser.add_argument("--dropout", type=float, default=0.7, help="dropout rate")
	parser.add_argument("--n-discrim-per-update", type=int, default=1,
		help="discriminator update this times per iter")
	parser.add_argument("--batch-size", type=int, default=32, help="num of data per batch")
	parser.add_argument("--max-iter", type=int, default=100, help="num of training of GAN")
	parser.add_argument("--every-sample", type=int, default=1, help="sample generated image per this value")
	parser.add_argument("--every-eval", type=int, default=10, help="cal test stats per this value")
	args = parser.parse_args()

	tf.reset_default_graph()
	gan = DCGAN(lr=args.lr, beta1=args.beta1, dropout=args.dropout,
		n_discrim_per_update=args.n_discrim_per_update, every_eval=args.every_eval,
		every_sample=args.every_sample)
	gan.train(batch_size=args.batch_size, max_iter=args.max_iter)
	print("finished.")


if __name__ == '__main__':
	#test()
	main()
