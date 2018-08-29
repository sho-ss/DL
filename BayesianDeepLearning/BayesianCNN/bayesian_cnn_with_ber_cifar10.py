import tensorflow as tf
import numpy as np

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python import debug as tf_debug

import argparse


def inference(input_tensor, keep_prob_conv, keep_prob_dense, wd):
	"""
	define calcurate graph


	Parameters
	----------
	input_tensor : Tensor
		placeholder for input.

	keep_prob_conv : Tensor
		feed value of keep probability at drop out in Convolution layer.

	keep_prob_dense : Tensor
		feed value of keep probability at drop out in fully connected layer.

	Returns
	-------
	logits : Tensor
		CNN output.
	"""
	k_initializer = tf.contrib.layers.xavier_initializer_conv2d()
	b_initializer = tf.zeros_initializer()
	w_initializer = tf.contrib.layers.xavier_initializer_conv2d()

	strides_kernel = [1, 1, 1, 1]
	strides_pooling = [1, 2, 2, 1]
	filter_pooling = [1, 2, 2, 1]

	input_shape = input_tensor.get_shape().as_list()
	with tf.variable_scope("layer0_conv"):
		# [height, width, input_channel_num, output_channel_num]
		kernel_shape = [5, 5, input_shape[-1], 64]
		bias_shape = [64]

		kernel = tf.get_variable("kernel", shape=kernel_shape,
			dtype=tf.float32, initializer=k_initializer)
		b = tf.get_variable("bias", shape=bias_shape,
			dtype=tf.float32, initializer=b_initializer)

		conv1 = tf.nn.conv2d(input_tensor, kernel, strides_kernel, padding="VALID") + b
		d_conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob_conv)
		pool1 = tf.nn.max_pool(d_conv1, ksize=filter_pooling, strides=strides_pooling,
			padding="SAME")

		# add weight decay
		weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)
		weight_decay = tf.multiply(tf.nn.l2_loss(b), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)

	with tf.variable_scope("layer1_conv"):
		# [height, width, input_channel_num, output_channel_num]
		kernel_shape = [5, 5, 64, 128]
		bias_shape = [128]

		kernel = tf.get_variable("kernel", shape=kernel_shape,
			dtype=tf.float32, initializer=k_initializer)
		b = tf.get_variable("bias", shape=bias_shape,
			dtype=tf.float32, initializer=b_initializer)

		conv2 = tf.nn.conv2d(pool1, kernel, strides_kernel, padding="VALID") + b
		d_conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob_conv)
		pool2 = tf.nn.max_pool(d_conv2, ksize=filter_pooling, strides=strides_pooling,
			padding="SAME")

		# add weight decay
		weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)
		weight_decay = tf.multiply(tf.nn.l2_loss(b), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)

	with tf.variable_scope("laeyer3_dense"):
		w_shape = [5 * 5 * 128, 1000]
		b_shape = [1000]

		W = tf.get_variable("weight", shape=w_shape,
			dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable("bias", shape=b_shape,
			dtype=tf.float32, initializer=b_initializer)

		pool2_reshaped = tf.reshape(pool2, shape=[-1, 5 * 5 * 128])

		activation = tf.nn.relu(tf.matmul(pool2_reshaped, W) + b)
		activation_drop = tf.nn.dropout(activation, keep_prob=keep_prob_dense)

		# add weight decay
		weight_decay = tf.multiply(tf.nn.l2_loss(W), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)
		weight_decay = tf.multiply(tf.nn.l2_loss(b), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)

	with tf.variable_scope("layer4_dense"):
		w_shape = [1000, 10]
		b_shape = [10]

		W = tf.get_variable("weight", shape=w_shape,
			dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable("bias", shape=b_shape,
			dtype=tf.float32, initializer=b_initializer)

		logits = tf.matmul(activation_drop, W) + b

		# add weidth decay
		weight_decay = tf.multiply(tf.nn.l2_loss(W), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)
		weight_decay = tf.multiply(tf.nn.l2_loss(b), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)

	return logits


def add_loss(t, logits):
	"""
	add loss

	Parameters
	----------
	t : Placeholder
		placeholder of true label
	logits : Tensor
		Output of CNN
	"""
	cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits))
	tf.add_to_collection("losses", cross_entropy_mean)
	return tf.add_n(tf.get_collection("losses"), name="total_loss")


def add_accuracy(t, samples):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(t, axis=1), tf.argmax(samples, axis=1)),
		dtype=tf.float32))
	return accuracy


def add_mc_drop_accuracy(t, samples):
	mc_drop_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(t, axis=1), tf.argmax(samples, axis=1)),
		dtype=tf.float32))
	return mc_drop_accuracy


def add_train(loss, learning_rate, momentum):
	train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
	return train_step


def main(_):
	tf.logging.set_verbosity(tf.logging.DEBUG)
	np.random.seed(42)
	# load data
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
	# normalization
	train_images = train_images.astype("float32")
	test_images = test_images.astype("float32")
	train_images /= 255.0
	test_images /= 255.0
	train_labels = np.reshape(train_labels, newshape=[len(train_labels), ])
	test_labels = np.reshape(test_labels, newshape=[len(test_labels), ])

	tf.logging.info("test shape: {}".format(test_labels.shape))
	tf.logging.info("train shape: {}".format(train_labels.shape))
	tf.logging.info("images: {}".format(train_images[0]))

	#perms = np.arange(len(test_labels))
	#np.random.shuffle(perms)
	#test_images = test_images[perms[0:5000]]
	#test_labels = test_labels[perms[0:5000]]

	perms = np.arange(len(train_labels))
	np.random.shuffle(perms)
	train_data_size = len(train_labels)
	total_processed = 0

	def cifar10_next_batch(batch_size):
		nonlocal total_processed
		nonlocal perms

		indices = perms[total_processed: total_processed + batch_size]
		x_batch = train_images[indices]
		y_batch = train_labels[indices]
		total_processed += len(y_batch)
		if total_processed >= train_data_size:
			np.random.shuffle(perms)
			total_processed = 0
			# if batch size not enough, then add the difference
			num_not_enough = batch_size - len(y_batch)
			if num_not_enough > 0:
				indices = perms[total_processed: total_processed + num_not_enough]
				x_tmp = train_images[indices]
				y_tmp = train_labels[indices]
				x_batch = np.vstack((x_batch, x_tmp))
				y_batch = np.hstack((y_batch, y_tmp))

				total_processed += len(indices)

		return x_batch, y_batch

	with tf.Graph().as_default():
		with tf.name_scope("bayesian_cnn"):
			x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input_place")
			t = tf.placeholder(tf.int32, shape=[None, ], name="target_place")
			t_onehot = tf.one_hot(t, depth=10, axis=-1)
			# t_onehot = tf.reduce_sum(t_onehot_3d, axis=1)
			# relate to drop out
			keep_prob_conv = tf.placeholder(tf.float32, shape=[], name="dropout_of_conv_layer")
			keep_prob_dense = tf.placeholder(tf.float32, shape=[], name="dropout_of_dense_layer")
			# relate to learning rate
			learning_rate = tf.placeholder(tf.float32, shape=[], name="larning_rate")
			logits = inference(x, keep_prob_conv, keep_prob_dense, FLAGS.weight_decay)
			softmax = tf.nn.softmax(logits, axis=1)
			# used for mc drop
			samples = tf.placeholder(tf.float32, shape=[None, 10], name="MC_drop")

		with tf.name_scope("train_stats"):
			with tf.name_scope("loss"):
				loss = add_loss(t_onehot, logits)
			with tf.name_scope("accuracy"):
				accuracy = add_accuracy(t_onehot, samples)

		with tf.name_scope("train"):
			train_step = add_train(loss, learning_rate, FLAGS.momentum)

		# sumary
		tf.summary.scalar("loss_test", loss)
		tf.summary.scalar("accuracy", accuracy)
		tf.summary.scalar("error", 1.0 - accuracy)
		merged = tf.summary.merge_all()

		with tf.Session() as sess:
			if tf.gfile.Exists(FLAGS.logdir):
				tf.gfile.DeleteRecursively(FLAGS.logdir)
			mc_writer = tf.summary.FileWriter(FLAGS.logdir + "/mc_drop", sess.graph)
			standard_writer = tf.summary.FileWriter(FLAGS.logdir + "/standard", sess.graph)

			sess.run(tf.global_variables_initializer())

			total_iter = 0

			for epoch in range(FLAGS.max_iter):
				x_batch, y_batch = cifar10_next_batch(FLAGS.batch_size)
				lr = FLAGS.base_lr * (1.0 + FLAGS.gamma * total_iter) ** (-FLAGS.power)
				_, loss_train = sess.run([train_step, loss],
					feed_dict={x: x_batch, t: y_batch,
					keep_prob_conv: 1.0 - FLAGS.dropout_conv,
					keep_prob_dense: 1.0 - FLAGS.dropout_dense,
					learning_rate: lr})
				total_iter += 1

				if (epoch + 1) % FLAGS.every_test == 0 or epoch == 0:
					# MC drop
					sample_val = np.zeros((len(test_labels), 10))
					for sample_iter in range(FLAGS.T):
						softmax_val = sess.run(softmax, feed_dict={x: test_images,
							keep_prob_conv: 1.0 - FLAGS.dropout_conv,
							keep_prob_dense: 1.0 - FLAGS.dropout_dense})
						sample_val += softmax_val
					sample_val /= FLAGS.T

					summary, accuracy_mc, loss_test = sess.run([merged, accuracy, loss],
						feed_dict={t: test_labels, samples: sample_val,
						x: test_images,
						keep_prob_conv: 1.0,
						keep_prob_dense: 1.0})
					mc_writer.add_summary(summary, epoch)

					std_samples = sess.run(logits, feed_dict={x: test_images,
						t: test_labels,
						keep_prob_conv: 1.0,
						keep_prob_dense: 1.0})

					summary, accuracy_std, loss_test = sess.run([merged, accuracy, loss],
						feed_dict={t: test_labels, samples: std_samples,
						x: test_images,
						keep_prob_conv: 1.0,
						keep_prob_dense: 1.0})
					standard_writer.add_summary(summary, epoch)

					tf.logging.info("[test] epoch {}: loss {:.3f}".format(epoch, loss_test))
					tf.logging.info("accuracy {:.3f}".format(accuracy_std))
					tf.logging.info("mc_drop_accuracy {:.3f}".format(accuracy_mc))
					# open memory
					del sample_val


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--logdir",
		type=str,
		default="./logdir")
	parser.add_argument("--every_test",
		type=int,
		default=5000)
	parser.add_argument("--max_iter",
		type=int,
		default=100000)
	parser.add_argument("--weight_decay",
		type=float,
		default=0.0005)
	parser.add_argument("--base_lr",
		type=float,
		default=0.01,
		help="this code use learning rate policy. \
		base_lr * (1 + gamma * iter)^(-power)")
	parser.add_argument("--gamma",
		type=float,
		default=0.0001)
	parser.add_argument("--power",
		type=float,
		default=0.75)
	parser.add_argument("--momentum",
		type=float,
		default=0.9)
	parser.add_argument("--batch_size",
		type=int,
		default=64)
	parser.add_argument("--dropout_conv",
		type=float,
		default=0.5)
	parser.add_argument("--dropout_dense",
		type=float,
		default=0.5)
	parser.add_argument("--T",
		type=int,
		default=50,
		help="num of samples in MC drop")
	FLAGS = parser.parse_args()

	tf.reset_default_graph()
	tf.app.run()
