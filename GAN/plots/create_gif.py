import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import sys
sys.path.append(os.getcwd())
from models import settings


def load_pkl(file_path):
	max_bytes = 2**31 - 1
	bytes_in = bytearray(0)
	input_size = os.path.getsize(file_path)
	with open(file_path, 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	data = pickle.loads(bytes_in)
	return data


def create_gif(samples_gen, epochs, cols, rows):
	# for initialize axes
	def init_visualize(xs, axes):
		xs_ = xs[0]
		ims = []
		for x, ax in zip(xs_, axes):
			im = ax.imshow(x.reshape(28, 28), cmap="gray", interpolation="nearest", animated=True)
			ax.axis("off")
			ims.append(im)
		return ims

	# used for creating animation
	def visualize(i, xs, title, epochs, ims):
		xs_ = xs[i]
		# plot images
		for x, im in zip(xs_, ims):
			im.set_array(x.reshape(28, 28))
		# plot fig title
		plt.suptitle(title + str(epochs[i]), color="blue", fontsize=16)

	# prepare for creating animation
	fig, axes = plt.subplots(cols, rows)
	axes = axes.reshape(cols * rows)
	# init
	ims = init_visualize(samples_gen, axes)
	#create start
	ani = animation.FuncAnimation(fig, visualize, fargs=(samples_gen, "epoch: ", epochs, ims),
		interval=800, frames=len(samples_gen))
	ani.save('./output/generate_image.gif', writer="imagemagick")


def main():
	samples = load_pkl("./output/samples.pkl")
	epochs = load_pkl("./output/epochs.pkl")
	print(len(samples))
	print(len(epochs))
	create_gif(samples_gen=samples, epochs=epochs, cols=settings.plot_cols, rows=settings.plot_rows)

main()
