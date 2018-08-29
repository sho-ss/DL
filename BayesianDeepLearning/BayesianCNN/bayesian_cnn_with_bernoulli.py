import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import argparse


def inference(input_tensor, keep_prob_conv, keep_prob_dense):
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
	w_initializer = tf.contrib.layers.xavier_initializer()

	strides_kernel = [1, 1, 1, 1]
	strides_pooling = [1, 2, 2, 1]
	filter_pooling = [1, 2, 2, 1]

	input_shape = input_tensor.get_shape().as_list()
	with tf.variable_scope("layer0_conv"):
		# [height, width, input_channel_num, output_channel_num]
		kernel_shape = [5, 5, input_shape[-1], 32]
		bias_shape = [32]

		kernel = tf.get_variable("kernel", shape=kernel_shape,
			dtype=tf.float32, initializer=k_initializer)
		b = tf.get_variable("bias", shape=bias_shape,
			dtype=tf.float32, initializer=b_initializer)

		conv1 = tf.nn.conv2d(input_tensor, kernel, strides_kernel, padding="VALID") + b
		d_conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob_conv)
		pool1 = tf.nn.max_pool(d_conv1, ksize=filter_pooling, strides=strides_pooling,
			padding="SAME")

	with tf.variable_scope("layer1_conv"):
		# [height, width, input_channel_num, output_channel_num]
		kernel_shape = [5, 5, 32, 64]
		bias_shape = [64]

		kernel = tf.get_variable("kernel", shape=kernel_shape,
			dtype=tf.float32, initializer=k_initializer)
		b = tf.get_variable("bias", shape=bias_shape,
			dtype=tf.float32, initializer=b_initializer)

		conv2 = tf.nn.conv2d(pool1, kernel, strides_kernel, padding="VALID") + b
		d_conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob_conv)
		pool2 = tf.nn.max_pool(d_conv2, ksize=filter_pooling, strides=strides_pooling,
			padding="SAME")

	with tf.variable_scope("laeyer3_dense"):
		w_shape = [4 * 4 * 64, 500]
		b_shape = [500]

		W = tf.get_variable("weight", shape=w_shape,
			dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable("bias", shape=b_shape,
			dtype=tf.float32, initializer=b_initializer)

		pool2_reshaped = tf.reshape(pool2, shape=[-1, 4 * 4 * 64])

		activation = tf.nn.relu(tf.matmul(pool2_reshaped, W) + b)
		activation_drop = tf.nn.dropout(activation, keep_prob=keep_prob_dense)

	with tf.variable_scope("layer4_dense"):
		w_shape = [500, 10]
		b_shape = [10]

		W = tf.get_variable("weight", shape=w_shape,
			dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable("bias", shape=b_shape,
			dtype=tf.float32, initializer=b_initializer)

		logits = tf.matmul(activation_drop, W) + b

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
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits))
	return loss


def add_accuracy(t, logits):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(t, axis=1), tf.argmax(logits, axis=1)),
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
	# load data
	mnist = input_data.read_data_sets(FLAGS.mnist_dir, one_hot=True)

	tf.logging.set_verbosity(tf.logging.DEBUG)
	np.random.seed(42)

	test_images = mnist.test.images
	test_labels = mnist.test.labels
	perms = np.arange(len(test_labels))
	np.random.shuffle(perms)
	test_images = test_images[perms[0:5000]]
	test_labels = test_labels[perms[0:5000]]

	with tf.Graph().as_default():
		with tf.name_scope("bayesian_cnn"):
			x = tf.placeholder(tf.float32, shape=[None, 28 * 28 * 1], name="input_place")
			x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])
			t = tf.placeholder(tf.int32, shape=[None, 10], name="target_place")
			# relate to drop out
			keep_prob_conv = tf.placeholder(tf.float32, shape=[], name="dropout_of_conv_layer")
			keep_prob_dense = tf.placeholder(tf.float32, shape=[], name="dropout_of_dense_layer")
			# relate to learning rate
			learning_rate = tf.placeholder(tf.float32, shape=[], name="larning_rate")
			logits = inference(x_reshape, keep_prob_conv, keep_prob_dense)
			softmax = tf.nn.softmax(logits, axis=1)
			# used for mc drop
			samples = tf.placeholder(tf.float32, shape=[None, 10], name="MC_drop")

		with tf.name_scope("train_stats"):
			with tf.name_scope("loss"):
				loss = add_loss(t, logits)
			with tf.name_scope("accuracy"):
				accuracy = add_accuracy(t, logits)
			with tf.name_scope("MC_drop"):
				mc_drop_accuracy = add_mc_drop_accuracy(t, samples)

		with tf.name_scope("train"):
			train_step = add_train(loss, learning_rate, FLAGS.momentum)

		# sumary
		tf.summary.scalar("loss_test", loss)
		tf.summary.scalar("standard_drop_accuracy", accuracy)
		tf.summary.scalar("standard_drop_error", 1.0 - accuracy)
		tf.summary.scalar("mc_drop_accuracy", mc_drop_accuracy)
		tf.summary.scalar("mc_drop_error", 1.0 - mc_drop_accuracy)
		merged = tf.summary.merge_all()

		with tf.Session() as sess:
			if tf.gfile.Exists(FLAGS.logdir):
				tf.gfile.DeleteRecursively(FLAGS.logdir)
			writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

			sess.run(tf.global_variables_initializer())

			total_iter = 0

			for epoch in range(FLAGS.max_iter):
				x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size, shuffle=True)
				lr = FLAGS.base_lr * (1.0 + FLAGS.gamma * total_iter) ** (-FLAGS.power)
				sess.run(train_step,
					feed_dict={x: x_batch, t: y_batch,
					keep_prob_conv: 1.0 - FLAGS.dropout_conv,
					keep_prob_dense: 1.0 - FLAGS.dropout_dense,
					learning_rate: lr})
				total_iter += 1

				if (epoch + 1) % FLAGS.every_test == 0 or epoch == 0:
					# MC drop
					sample_val = np.zeros((5000, 10))
					for sample_iter in range(FLAGS.T):
						softmax_val = sess.run(softmax, feed_dict={x: test_images,
							keep_prob_conv: 1.0 - FLAGS.dropout_conv,
							keep_prob_dense: 1.0 - FLAGS.dropout_dense})
						sample_val += softmax_val
					sample_val /= FLAGS.T

					summary, loss_test, accuracy_test, mc_drop_val = sess.run([merged, loss, accuracy, mc_drop_accuracy],
						feed_dict={x: test_images, t: test_labels,
						samples: sample_val,
						keep_prob_conv: 1.0,
						keep_prob_dense: 1.0})
					writer.add_summary(summary, epoch)
					tf.logging.info("[test] epoch {}: loss {:.3f}".format(epoch, loss_test))
					tf.logging.info("accuracy {:.3f}".format(accuracy_test))
					tf.logging.info("mc_drop_accuracy {:.3f}".format(mc_drop_val))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--logdir",
		type=str,
		default="./logdir")
	parser.add_argument("--mnist_dir",
		type=str,
		required=True)
	parser.add_argument("--every_test",
		type=int,
		default=5000)
	parser.add_argument("--max_iter",
		type=int,
		default=100000)
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
