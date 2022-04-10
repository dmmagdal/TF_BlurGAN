# blurgan.py
# Implement the BlurGAN model for de-blurring images. This is a form of
# image super resolution.
# Source: https://medium.com/sicara/keras-generative-adversarial-
# networks-image-deblurring-45e3ab6977b5
# Source (Dataset Site): https://seungjunnah.github.io/Datasets/gopro
# Source (Abridged Dataset): https://drive.google.com/file/d/
# 1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view
# Source (Full Dataset): https://drive.google.com/file/d/
# 1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from tqdm import tqdm


class ResBlock(layers.Layer):
	def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), 
			use_dropout=False, **kwargs):
		super().__init__()
		# self.pad_reflect = layers.ReflectionPadding2D((1, 1))
		self.pad_reflect = layers.ZeroPadding2D((1, 1))
		self.conv = layers.Conv2D(
			filters, kernel_size=kernel_size, strides=strides
		)
		self.batch_norm = layers.BatchNormalization()
		self.relu = layers.ReLU()

		self.use_dropout = use_dropout
		self.dropout = layers.Dropout(0.5)

		# self.pad_reflect2 = layers.ReflectionPadding2D((1, 1))
		self.pad_reflect2 = layers.ZeroPadding2D((1, 1))
		self.conv2 = layers.Conv2D(
			filters, kernel_size=kernel_size, strides=strides
		)
		self.batch_norm2 = layers.BatchNormalization()

		self.add = layers.Add()


	def call(self, inputs):
		x = self.pad_reflect(inputs)
		x = self.conv(x)
		x = self.batch_norm(x)
		x = self.relu(x)

		if self.use_dropout:
			x = self.dropout(x)

		x = self.pad_reflect2(x)
		x = self.conv2(x)
		x = self.batch_norm2(x)
		outs = self.add([inputs, x])
		return outs


	def get_config(self):
		config = super(ResBlock, self).get_config()
		config.update({
			"use_dropout": self.use_dropout,
		})
		return config


def create_generator(inputs, num_blocks=9):
	ngf = 64 # Number of generator fileters
	input_nc = 3 # Number of input channels (RGB)
	output_nc = 3 # Number of output channels (RGB)

	# x = layers.ReflectionPadding2D((3, 3))(inputs)
	x = layers.ZeroPadding2D((3, 3))(inputs)
	x = layers.Conv2D(ngf,kernel_size=(7, 7), padding="valid")(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	n_downsampling = 2
	for i in range(n_downsampling):
		multi = 2 ** i
		x = layers.Conv2D(
			ngf * multi * 2, kernel_size=(3, 3), strides=2, 
			padding="same"
		)(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)

	mult = 2 ** n_downsampling
	for i in range(num_blocks):
		x = ResBlock(ngf * mult, use_dropout=True)(x)

	for i in range(n_downsampling):
		mult = 2 ** (n_downsampling - 1)
		x = layers.Conv2DTranspose(
			int(ngf * multi / 2), kernel_size=(3, 3), strides=2, 
			padding="same"
		)(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)

	# x = layers.ReflectionPadding2D((3, 3))(x)
	x = layers.ZeroPadding2D((3, 3))(x)
	x = layers.Conv2D(
		output_nc, kernel_size=(7, 7), padding="valid"
	)(x)
	x = layers.Activation("tanh")(x)

	x = layers.Add()([x, inputs])
	outputs = layers.Lambda(lambda x:x / 2)(x)

	return Model(inputs, outputs, name="generator")


def create_discriminator(inputs):
	ndf = 64 # Number of discriminator filters
	n_layers, use_sigmoid = 3, False

	x = layers.Conv2D(
		filters=ndf, kernel_size=(4, 4), strides=2, padding="same"
	)(inputs)
	x = layers.LeakyReLU(alpha=0.2)(x)

	nf_mult, nf_mult_prev = 1, 1
	for n in range(n_layers):
		nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
		x = layers.Conv2D(
			filters=ndf * nf_mult, kernel_size=(4, 4), strides=2,
			padding="same"
		)(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU(alpha=0.2)(x)

	nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
	x = layers.Conv2D(
		filters=ndf * nf_mult, kernel_size=(4, 4), strides=1,
		padding="same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU(alpha=0.2)(x)

	x = layers.Conv2D(
		filters=1, kernel_size=(4, 4), strides=1, padding="same"
	)(x)
	if use_sigmoid:
		x = layers.Activation("sigmoid")(x)

	x = layers.Flatten()(x)
	x = layers.Dense(1024, activation="tanh")(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)

	return Model(inputs, outputs, name="discriminator")


def build_vgg19(hr_shape):
	vgg = VGG19(
		weights="imagenet", include_top=False, input_shape=hr_shape
	)
	block3_conv4 = 10
	block5_conv4 = 20

	return Model(
		inputs=vgg.inputs, outputs=vgg.layers[block5_conv4].output, 
		name="vgg19"
	)


def create_gan(generator, discriminator):
	input_nc = 3 # Number of input channels (RGB)
	inputs = layers.Input(shape=(256, 256, input_nc))
	generated_images = generator(inputs)
	outputs = discriminator(generated_images)
	return Model(inputs, [generated_images, outputs], name="gan")


def perceptual_loss(y_true, y_pred):
	vgg = VGG19(
		include_top=False, weights="imagenet", 
		input_shape=(256, 256, 3)
	)
	model = Model(
		inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
	)
	model.trainable = False
	y_true_features = vgg(y_true)
	y_pred_features = vgg(y_pred)
	mse = keras.losses.MeanSquaredError()
	return mse(y_true_features, y_pred_features)


def wasserstein_loss(y_true, y_pred):
	return keras.metrics.Mean(y_true, y_pred)


def load_images(blur_file, sharp_file):
	blur = tf.io.read_file(blur_file)
	blur_image = tf.io.decode_image(
		blur, channels=3, expand_animations=False
	)
	blur_image = tf.image.resize(
		blur_image, (256, 256), method="bicubic"
	)
	sharp = tf.io.read_file(sharp_file)
	sharp_image = tf.io.decode_image(
		sharp, channels=3, expand_animations=False
	)
	sharp_image = tf.image.resize(
		sharp_image, (256, 256), method="bicubic"
	)
	return {"blur": blur_image, "sharp": sharp_image}


'''
def main():
	# Get the list of files.
	blur_files = []
	sharp_files = []
	for root, _, files in os.walk("."):
		if len(files) > 0:
			if "blur" in root and "blur_gamma" not in root:
				blur_files += [
					os.path.join(root, file) for file in files
					if file.endswith(".png")
				]
			elif "sharp" in root:
				sharp_files += [
					os.path.join(root, file) for file in files
					if file.endswith(".png")
				]

	# Create datasets from the image files.
	train_blur = [file for file in blur_files if "train" in file]
	train_sharp = [file for file in sharp_files if "train" in file]
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(train_blur, train_sharp)
	)
	train_dataset = train_dataset.map(load_images)

	test_blur = [file for file in blur_files if "test" in file]
	test_sharp = [file for file in sharp_files if "test" in file]
	test_dataset = tf.data.Dataset.from_tensor_slices(
		(test_blur, test_sharp)
	)
	test_dataset = test_dataset.map(load_images)

	# Load dataset.
	# data = load_images("./images/train", n_images)
	# y_train, x_train = data["B"], data["A"]

	# Initialize models.
	input_nc = 3 # Number of input channels (RGB)
	output_nc = 3 # Number of output channels (RGB)
	gen_inputs = layers.Input(shape=(256, 256, input_nc))
	disc_inputs = layers.Input(shape=(256, 256, output_nc))
	gen = create_generator(gen_inputs)
	disc = create_discriminator(disc_inputs)
	gan = create_gan(gen, disc)
	gen.summary()
	disc.summary()

	# Initialize optimizers.
	gen_opt = keras.optimizers.Adam(
		lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
	)
	disc_opt = keras.optimizers.Adam(
		lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
	)
	gan_opt = keras.optimizers.Adam(
		lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
	)

	# Compile models.
	disc.trainable = True
	disc.compile(optimizer=disc_opt, loss=wasserstein_loss)
	disc.trainable = False
	loss = [perceptual_loss, wasserstein_loss]
	loss_weights = [100, 1]
	gan.compile(
		optimizer=gan_opt, loss=loss, loss_weights=loss_weights
	)
	disc.trainable = True

	# Training loop.
	batch_size = 4
	num_epochs = 50
	critic_updates = 5
	for epoch in range(num_epochs):
		print("Epoch: {}/{}".format(epoch + 1, num_epochs))
		# print("batches: {}".format(x_train.shape[0] / batch_size))
		print("batches: {}".format(int(len(train_dataset) / batch_size)))

		# Randomize into batches.
		# permutated_indexes = np.random.permutation(x_train.shape[0])
		train_dataset.shuffle(buffer_size=100)
		train = list(train_dataset.as_numpy_iterator())

		# for index in range(int(x_train.shape[0] / batch_size)):
		for index in tqdm(range(int(len(train_dataset) / batch_size))):
			# Batch preparation.
			# batch_indexes = permutated_indexes[index * batch_size: (index + 1) * batch_size]
			# image_blur_batch = x_train[batch_indexes]
			# image_full_batch = y_train[batch_indexes]
			permutated_data = train[
				index * batch_size: (index + 1) * batch_size
			]
			image_blur_batch = tf.convert_to_tensor([
				image["blur"] for image in permutated_data
			])
			image_full_batch = tf.convert_to_tensor([
				image["sharp"] for image in permutated_data	
			])
			output_false_batch = tf.zeros((batch_size, 1))
			output_true_batch = tf.ones((batch_size, 1))

			# Generate fake images.
			generated_images = gen.predict(
				x=image_blur_batch,	batch_size=batch_size
			)

			# Train multiple times on real and fake inputs.
			for _ in range(critic_updates):
				d_loss_real = disc.train_on_batch(
					image_full_batch, output_true_batch
				)
				d_loss_fake = disc.train_on_batch(
					generated_images, output_false_batch
				)
				d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

			disc.trainable = False
			# Train generator on discriminator's decision and generated
			# images.
			gan_loss = gan.train_on_batch(
				# image_blur_batch, 
				# [image_full_batch, output_true_batch]
				tf.concat(image_blur_batch, image_full_batch, axis=0),
				tf.concat(output_true_batch, output_true_batch, axis=0)
			)
			disc.trainable = False

	# Exit the program.
	exit(0)
'''


if __name__ == '__main__':
	main()