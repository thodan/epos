# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""A script with utility functions."""

import re
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables
import cv2
from epos_lib import misc


def convert_px_indices_to_im_coords(px_indices, scale):
  """Converts pixel indices to (possibly scaled) image coordinates.

  With scale = 1.0, pixel index (i, j) corresponds to image coordinates
  (i + 0.5, j + 0.5).

  Args:
    px_indices: [n, 2] ndarray with pixel indices.
    scale: Scale to be applied to the image coordinates.
  Returns:
    [n, 2] ndarray with image coordinates.
  """
  return scale * (px_indices.astype(np.float64) + 0.5)


def resolve_shape(tensor, rank=None, scope=None):
  """Fully resolves the shape of a Tensor.

  Use as much as possible the shape components already known during graph
  creation and resolve the remaining ones during runtime.

  Args:
    tensor: Input tensor whose shape we query.
    rank: The rank of the tensor, provided that we know it.
    scope: Optional name scope.
  Returns:
    The full shape of the tensor.
  """
  with tf.name_scope(scope, 'resolve_shape', [tensor]):
    if rank is not None:
      shape = tensor.get_shape().with_rank(rank).as_list()
    else:
      shape = tensor.get_shape().as_list()

    if None in shape:
      shape_dynamic = tf.shape(tensor)
      for i in range(len(shape)):
        if shape[i] is None:
          shape[i] = shape_dynamic[i]

    return shape


def resize_image_py(image, size, interpolation=None):
  """Resizes an image with a suitable interpolation method (without TensorFlow).

  Args:
    image: Input image.
    size: Size of the output image (width, height).
    interpolation: Interpolation method
  Returns:
    Resized image.
  """
  if interpolation is None:
    if image.shape[0] >= size[1]:
      interpolation = cv2.INTER_AREA
    else:
      interpolation = cv2.INTER_LINEAR
  return cv2.resize(image, size, interpolation=interpolation)


def resize_image_tf(image, size):
  """Resizes an image with a suitable interpolation method (with TensorFlow).

  Args:
    image: Input image.
    size: Size of the output image (width, height).
  Returns:
    Resized image.
  """
  images = tf.expand_dims(image, axis=0)
  images_resized = tf.cond(
    tf.math.greater_equal(misc.resolve_shape(image)[0], size[1]),
    lambda: tf.image.resize_area(
      images, (size[1], size[0]), align_corners=True),
    lambda: tf.image.resize_bilinear(
      images, (size[1], size[0]), align_corners=True))
  return tf.squeeze(images_resized, axis=0)


def resize_bilinear(images, shape, output_dtype=tf.float32):
  """Resizes an image with the billinear interpolation (Tensorflow-based).

  Args:
    images: Tensor of size [batch, height_in, width_in, channels].
    shape: 1-D int32 Tensor of 2 elements: new_height, new_width. The new shape
      for the images.
    output_dtype: Output data type.
  Returns:
    Tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  images = tf.image.resize_bilinear(images, shape, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def crop_image(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  From: research/deeplab/core/preprocess_utils.py

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    The cropped (and resized) image.
  Raises:
    ValueError: if `image` doesn't have rank of 3.
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = resolve_shape(image)

  if len(image.get_shape().as_list()) != 3:
    raise ValueError('input must have rank of 3')
  original_channels = image.get_shape().as_list()[2]

  rank_assertion = tf.Assert(
    tf.equal(tf.rank(image), 3),
    ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
    tf.logical_and(
      tf.greater_equal(original_shape[0], crop_height),
      tf.greater_equal(original_shape[1], crop_width)),
    ['Crop size greater than the image size.'])

  offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  image = tf.reshape(image, cropped_shape)
  image.set_shape([crop_height, crop_width, original_channels])
  return image


def get_variable_dict():
  """Returns a dictionary of all existing tensorflow variables.

  Returns:
    A dictionary mapping the variable name to the variable.
  """
  var_list = variables._all_saveable_objects()
  var_list.append(tf.train.get_or_create_global_step())
  var_dict = {var.op.name: var for var in var_list}
  return var_dict


def filter_variables(variables, filter_regex_list, invert=False):
  """Filters out variables matching any of the specified regular expressions.

  Filters out the variables whose name matches any of the regular expressions
  in filter_regex_list and returns the remaining variables. Optionally, if
  invert=True, the complement set is returned.

  Args:
    variables: A list of tensorflow variables.
    filter_regex_list: A list of string regular expressions.
    invert: If True, returns the complement of the filtered set; that is, all
      variables matching filter_regex are kept.
  Returns:
    A list of filtered variables.
  """
  kept_vars = []
  variables_to_ignore_patterns = list(filter(None, filter_regex_list))
  for var in variables:
    add = True
    for pattern in variables_to_ignore_patterns:
      if re.match(pattern, var.op.name):
        add = False
        break
    if add != invert:
      kept_vars.append(var)
  return kept_vars
