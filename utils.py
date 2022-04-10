# utils.py
# A module containing utilities for BlurGAN including image
# preprocessing and generative sampling.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import random
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


def resize_images(images):
	# Reshape lr & hr images to the set sizes. Since we're using the
	# bicubic x4 subset, make sure that the hr dimensions are 4x the lr
	# dimensions (hr = (128, 128, 3), lr = (32, 32, 3)).
	hr_dims = [128, 128]
	lr_dims = [32, 32]
	return {
		"hr": tf.image.resize(images["hr"], hr_dims, method="bicubic"),
		"lr": tf.image.resize(images["lr"], lr_dims, method="bicubic"),
	}


def scale_images(images):
	# Normalize (scale) values (divide by 255.0).
	return {
		"hr": images["hr"] / 255.0, 
		"lr": images["lr"] / 255.0,
	}


def save_images(valid_data, generator, e, offset_by_one=False):
	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	# src_img = sample["lr"]
	# tar_img = sample["hr"]
	src_img = sample["blur"]
	tar_img = sample["sharp"]
	n_dims = len(tf.shape(src_img))

	# If the sample is not in (batch_size, height, width, channel)
	# format, then it must be in (height, width, channel) format.
	if n_dims == 3:
		src_img = tf.expand_dims(src_img, axis=0)
		tar_img = tf.expand_dims(tar_img, axis=0)

	# Note: matplotlib pyplot is able to print float values from [0, 1]
	# in their original color (as long as they are floats). In other
	# words, the graphing handles the conversion from grayscale to RGB
	# (scalar multiply by 255). If the values are int from [0, 255],
	# then the graphing is already in RGB and is printed out with no
	# need of conversion (no need to multiply by 255).
	gen_img = generator.predict(src_img) #* 255.0

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
	if offset_by_one:
		e += 1
	# plt.savefig(f"SRGAN_Generator_Sample{e}.png")
	plt.savefig(f"BlurGAN_Generator_Sample{e}.png")
	plt.close()


def psnr(y_true, y_pred):
	# Calculate the Peak Signal-to-Noise Ratio (PSNR) score between the
	# two images. Source: https://www.geeksforgeeks.org/python-peak-
	# signal-to-noise-ratio-psnr/
	# PSNR = 10logbase10((L - 1)^2 / (MSE)) where L is the number of
	# maximum possible intensity levels (minimum intensity level is
	# supposed to be 0) in an image.
	mse = keras.metrics.mean_squared_error(y_true, y_pred)

	if mse == 0:
		# MSE == 0 (exactly the same picture) means that no noise is
		# present in the signal, therefore PSNR has not importance.
		return 100

	psnr = 10 * tf.math.log(1 / mse)
	return psnr # units are dB