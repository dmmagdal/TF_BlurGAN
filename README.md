# BlurGAN

Description: This repository takes from multiple examples of the DeblurGAN for image  debluurring (a form of image super resolution) and implements the model in Tensorflow 2. This is  pretty much the same workflow as the TF_SRGAN and TF_ESRGAN repositories you can find on my [GitHub] (https://github.com/dmmagdal/TF_ESRGAN).

### How to Use:

 > Install the required modules from requirements.txt with `pip install -r requirements.txt`. The best way to train BlurGAN from scratch is to use the training loop defined in `train.py`. After downloading and extracting the dataset (linked below), simply run `python train.py` and it will begin training the neural network. You can go inside and alter the training hyperparameters (ie `batch_size`, `epochs`, etc), making this repo very easy to use for training the model from scratch.


### Sources:

 - [GitHub] (https://github.com/raphaelmeudec/deblur-gan)
 - [DeblurGAN Paper] (https://arxiv.org/pdf/1711.07064.pdf)
 - [Medium] (https://medium.com/sicara/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)
 - [Dataset Site] (https://seungjunnah.github.io/Datasets/gopro)
 - [Abridged Dataset] (https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view)
 - [Full Dataset] (https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view)
 - [DGGAN Keras Example] (https://keras.io/examples/generative/dcgan_overriding_train_step/)


### Training Photos:

1) A 300 epoch run with the training loop in train.py. This training loop use the same parameters from the ESRGAN repository, including the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted BCE from discriminator (perceptual loss). Note that there is still some distortion or "glitching" in the images despite being very similar to the high resolution image. 

2) A 300 epoch run with the training loop in train.py. This training loop use the same parameters from the ESRGAN repository, including the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted Wasserstein loss from discriminator (perceptual loss). Note that this resulted in there being grid artifacts on the generated images.

3) A 300 epoch run with the training loop in train.py. This training loop use the same parameters from the ESRGAN repository, including the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted Wasserstein loss from discriminator (perceptual loss). The Conv2DTranspose layers in the generator were also replaced with UpSampling2D + Conv2D layers. This allowed for the removal of grid artifacts from the images.

4) A 300 epoch run with the training loop in train.py. This training loop use the same parameters from the ESRGAN repository, including the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted Wasserstein from discriminator (perceptual loss). Here, the weights between the perceptual loss and content loss were now [100, 1] instead of [1e-3, 1] respectively. Generator still using UpSampling2D + Conv2D layers over Conv2DTranspose. 