# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Image augmentation operations."""

import random
import numpy as np
import cv2
import tensorflow as tf


def random_adjust_brightness(image, min_delta=-0.2, max_delta=0.2):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: See max_delta.
    max_delta: How much to change the brightness. Brightness will change with a
               value between min_delta and max_delta.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    delta = tf.random_uniform([], min_delta, max_delta)
    image = tf.image.adjust_brightness(image, delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.2):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: See max_delta.
    max_delta: How much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    contrast_factor = tf.random_uniform([], min_delta, max_delta)
    image = tf.image.adjust_contrast(image, contrast_factor)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_hue(image, max_delta=0.02):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: Change hue randomly with a value between 0 and max_delta.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    delta = tf.random_uniform([], -max_delta, max_delta)
    image = tf.image.adjust_hue(image, delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.2):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: See max_delta.
    max_delta: How much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    saturation_factor = tf.random_uniform([], min_delta, max_delta)
    image = tf.image.adjust_saturation(image, saturation_factor)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_blur(image, max_sigma=3.0):
  """Randomly blurs the image.

  Randomly blur an image by applying a gaussian filter with a random sigma
  (0., sigma_max).

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    sigma: Float or list of float. Standard deviation for Gaussian kernel.
           The standard deviations of the Gaussian filter are given for each
           axis as a sequence, or as a single number, in which case it is equal
           for all axes.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomBlur', values=[image]):

    def add_random_blur(image, max_sigma):
      sigma = random.uniform(0., max_sigma)
      blurred = cv2.GaussianBlur(np.array(image), (0, 0), sigma)
      # blurred = scipy.ndimage.filters.gaussian_filter(image, sigma)
      return blurred

    blurred = tf.py_func(add_random_blur, [image, max_sigma], tf.float32)
    blurred = tf.reshape(blurred, tf.shape(image))  # Help to infer the shape.
    return blurred


def random_gaussian_noise(image, max_sigma=0.05):
  """Adds gaussian noise with a random sigma from [0., max_sigma].

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    sigma: Float or list of float.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('RandomGuassianNoise', values=[image]):

    def add_random_noise(image, max_sigma):
      sigma = random.uniform(0., max_sigma)

      # Faster than np.random.normal:
      noise = np.zeros(image.size, np.float32)
      noise = cv2.randn(noise, 0.0, sigma).reshape(image.shape)

      noisy = image + noise
      return noisy

    noisy = tf.py_func(add_random_noise, [image, max_sigma], tf.float32)
    noisy = tf.clip_by_value(noisy, clip_value_min=0.0, clip_value_max=1.0)
    noisy = tf.reshape(noisy, tf.shape(image))  # Help to infer the shape.
    return noisy


def jpeg_artifacts(image, min_quality=80):
  """Adds JPEG artifacts with a random JPEG quality.

  Args:
    image: Rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_quality: Minimal JPEG quality.

  Returns:
    image: Image which is the same shape as input image.
  """
  with tf.name_scope('JpegArtifacts', values=[image]):

    image_aug = tf.image.random_jpeg_quality(image, min_quality, 100)
    image_aug = tf.reshape(image_aug, tf.shape(image))  # Infer the shape.
    return image_aug
