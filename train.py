# train.py
# Initialize and train a BlurGAN model for image de-blurring (a form of
# image super resolution).
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
from blurgan import ResBlock
from blurgan import create_generator, create_discriminator
from blurgan import build_vgg19
from utils import resize_images, scale_images, save_images
from matplotlib import pyplot as plt


class BlurGAN(keras.Model):
	def __init__(self, generator, discriminator, vgg, **kwargs):
		super(BlurGAN, self).__init__(**kwargs)

		# Models.
		self.discriminator = discriminator
		self.generator = generator
		self.vgg = vgg

		# Mean squared error for VGG loss.
		self.mse = keras.losses.MeanSquaredError()


	def vgg_loss(self, y_true, y_pred):
		# Pass both the real and fake (generated) high res images
		# through the VGG19 model.
		real_features = self.vgg(y_true)
		fake_features = self.vgg(y_pred)

		# Return the MSE between the real and generated images.
		return self.mse(real_features, fake_features)


	def compile(self, gen_opt, disc_opt):
		super(BlurGAN, self).compile()
		self.disc_optimizer = disc_opt
		self.gen_optimizer = gen_opt
		self.d_loss_metric = keras.metrics.Mean(name="d_loss")
		self.g_loss_metric = keras.metrics.Mean(name="g_loss")
		self.d_loss_fn = keras.losses.BinaryCrossentropy()
		self.g_loss_fn = self.vgg_loss # content loss
		self.g_loss_fn2 = keras.losses.MeanSquaredError()
		self.custom_loss_weights = [1e-3, 1]


	@property
	def metrics(self):
		return [self.d_loss_metric, self.g_loss_metric]


	def train_step(self, data):
		# Get the input (low resolution/lr and high resolution/hr
		# images). Also extract the batch size.
		# lr_imgs = data["lr"]
		# hr_imgs = data["hr"]
		lr_imgs = data["blur"]
		hr_imgs = data["sharp"]
		batch_size = tf.shape(hr_imgs)[0]

		# Use the gradient tapes for both the discrinimator and
		# generator to track gradients for the respective models.
		with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
			# Generate fake images.
			fake_imgs = self.generator(lr_imgs)

			# Combine with real images.
			combined_imgs = tf.concat([fake_imgs, hr_imgs], axis=0)

			# Initialize labels for the respective images (1 for fake
			# images from the generator and 0 for real hr images).
			labels = tf.concat(
				[tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
				axis=0
			)

			# Add some random noise to the labels (supposed to be an
			# important trick). See DCGAN example from Keras examples.
			#labels += 0.05 * tf.random.uniform(tf.shape(labels))

			# Train discriminator.
			predictions = self.discriminator(combined_imgs)
			d_loss = self.d_loss_fn(labels, predictions)
			grads = disc_tape.gradient(
				d_loss, self.discriminator.trainable_weights
			)
			self.disc_optimizer.apply_gradients(
				zip(grads, self.discriminator.trainable_variables)
			)

			# Train generator (Do NOT update the weights of the
			# discriminator).
			# VGG-MSE loss between features extracted from HR and
			# generated image.
			g_loss = self.g_loss_fn(hr_imgs, fake_imgs) * self.custom_loss_weights[1]
			# Weighted BCE loss between how many times discriminator
			# falsely labeled generated image as "real" (We want the
			# discriminator to try and wrongly label a generated image
			# as real).
			g_loss2 = self.d_loss_fn(
				labels[-batch_size:, :], predictions[:batch_size, :], 
				self.custom_loss_weights[0]
			)
			# Raw MSE loss between HR and generated image.
			# g_loss3 = self.g_loss_fn2(hr_imgs, fake_imgs, self.custom_loss_weights[1])
			grads = gen_tape.gradient(
				# g_loss, self.generator.trainable_weights
				g_loss + g_loss2, self.generator.trainable_weights # This gave best results
				# g_loss + g_loss2 + g_loss3, self.generator.trainable_weights
			)
			self.gen_optimizer.apply_gradients(
				zip(grads, self.generator.trainable_weights)
			)

		# Update metrics.
		self.d_loss_metric.update_state(d_loss)
		self.g_loss_metric.update_state(g_loss)
		return {
			"d_loss": self.d_loss_metric.result(),
			"g_loss": self.g_loss_metric.result(),
		}


	def save(self, path, h5=True):
		if h5:
			self.generator.save(path + "_generator.h5")
			self.discriminator.save(path + "_discriminator.h5")
		else:
			self.generator.save(path + "_generator")
			self.discriminator.save(path + "_discriminator")


class GANMonitor(keras.callbacks.Callback):
	def __init__(self, valid_data, epoch_freq=10):
		self.valid_data = valid_data
		self.epoch_freq = epoch_freq


	def on_epoch_end(self, epoch, logs=None):
		if (epoch + 1) % self.epoch_freq == 0:
			save_images(self.valid_data, self.model.generator, epoch, True)


def load_images(blur_file, sharp_file):
	# Load images.
	blur = tf.io.read_file(blur_file)
	blur_image = tf.io.decode_image(
		blur, channels=3, expand_animations=False
	)
	sharp = tf.io.read_file(sharp_file)
	sharp_image = tf.io.decode_image(
		sharp, channels=3, expand_animations=False
	)

	# Reshape images to the set sizes. Since the task is image
	# deblurring (instead of super resolution), the input and output
	# shapes are the same (256, 256, 3).
	blur_image = tf.image.resize(
		blur_image, (256, 256), method="bicubic"
	)
	sharp_image = tf.image.resize(
		sharp_image, (256, 256), method="bicubic"
	)

	# Normalize (scale) values (divide by 255.0).
	blur_image = blur_image / 255.0
	sharp_image = sharp_image / 255.0
	return {"blur": blur_image, "sharp": sharp_image}


def load_gopro_dataset():
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
	autotune = tf.data.AUTOTUNE
	train_blur = [file for file in blur_files if "train" in file]
	train_sharp = [file for file in sharp_files if "train" in file]
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(train_blur, train_sharp)
	)
	train_dataset = train_dataset.map(
		load_images, num_parallel_calls=autotune
	)

	test_blur = [file for file in blur_files if "test" in file]
	test_sharp = [file for file in sharp_files if "test" in file]
	test_dataset = tf.data.Dataset.from_tensor_slices(
		(test_blur, test_sharp)
	)
	test_dataset = test_dataset.map(
		load_images, num_parallel_calls=autotune
	)

	return train_dataset, test_dataset


def main():
	# Load GoPro dataset.
	train_data, valid_data = load_gopro_dataset()

	single_sample = list(train_data.as_numpy_iterator())[0]
	blur_shape = tf.shape(single_sample["blur"]).numpy()
	blur_shape = (blur_shape[0], blur_shape[1], blur_shape[2])
	sharp_shape = tf.shape(single_sample["sharp"]).numpy()
	sharp_shape = (sharp_shape[0], sharp_shape[1], sharp_shape[2])
	print(blur_shape)
	print(sharp_shape)

	input_shape = (256, 256, 3)
	print(f"Blur and Sharp image shapes match: {blur_shape == sharp_shape}")
	print(f"Shapes match input shape {input_shape}: {blur_shape == input_shape}")

	# Define inputs to the models.
	blur_inputs = layers.Input(shape=input_shape)
	sharp_inputs = layers.Input(shape=input_shape)

	# Initialize models (generator, discriminator, and vgg).
	gen_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.99)
	generator = create_generator(blur_inputs)
	generator.summary()

	disc_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.99)
	discriminator = create_discriminator(sharp_inputs)
	discriminator.summary()

	vgg = build_vgg19(blur_shape)
	vgg.trainable = False
	vgg.summary()

	# Initialize BlurGAN model.
	gan = BlurGAN(generator, discriminator, vgg)
	gan.compile(gen_opt, disc_opt)
	save_callback = GANMonitor(valid_data)

	# Train the GAN.
	autotune = tf.data.AUTOTUNE
	epochs = 300#50#100#300
	batch_size = 4
	train_data = train_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	valid_data = valid_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	history = gan.fit(
		train_data,
		epochs=epochs,
		callbacks=[save_callback]
	)

	# Save the generator from the GAN.
	# gan.save(f"BlurGAN_{epochs}", h5=True)
	gan.save(f"BlurGAN_{epochs}", h5=False)

	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	# src_img = sample["lr"]
	# tar_img = sample["hr"]
	src_img = sample["blur"]
	tar_img = sample["sharp"]

	loaded_generator = load_model(
		# "BlurGAN_" + str(epochs) + "_generator.h5",
		"BlurGAN_" + str(epochs) + "_generator",
		custom_objects={
			"ResBlock": ResBlock,
		}
	)
	gen_img = loaded_generator.predict(src_img)

	# Plot all three images.
	plt.figure(figsize=(16, 8))
	plt.subplot(231)
	# plt.title("LR Image")
	plt.title("Blur Image")
	plt.imshow(src_img[0, :, :, :])
	plt.subplot(232)
	# plt.title("Superresolution")
	plt.title("Deblurring")
	plt.imshow(gen_img[0, :, :, :])
	plt.subplot(233)
	# plt.title("HR Image")
	plt.title("Sharp Image")
	plt.imshow(tar_img[0, :, :, :])
	plt.show()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()