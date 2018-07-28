import matplotlib.pyplot as plt
import pickle
import os


def load_pkl(file_path):
	max_bytes = 2**31 - 1
	bytes_in = bytearray(0)
	input_size = os.path.getsize(file_path)
	with open(file_path, 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	data = pickle.loads(bytes_in)
	return data


def create_png(samples, cols, rows):
	# prepare for creating animation
	fig, axes = plt.subplots(cols, rows)
	axes = axes.reshape(cols * rows)
	for im, ax in zip(samples, axes):
		ax.imshow(im.reshape(28, 28), cmap="gray", interpolation="nearest")
		ax.axis("off")
	plt.suptitle("latest output")
	plt.savefig(os.getcwd() + "/output/generate_imgae.png")


def main():
	samples = load_pkl(os.getcwd() + "/output/samples.pkl")
	plot_cols = 5
	plot_rows = 8
	create_png(samples[-1], cols=plot_cols, rows=plot_rows)

main()
