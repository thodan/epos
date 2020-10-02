# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Common utility functions and classes for creating a TFRecord file."""

import collections
import six
import tensorflow as tf


def save_example_list(path, example_list):
  with open(path, 'w') as f:
    txt = ''
    for e in example_list:
      txt += '{} {}'.format(e['scene_id'], e['im_id']) + '\n'
    f.write(txt)


def load_example_list(path):
  example_list = []
  with open(path, 'r') as f:
    for line in f.read().splitlines():
      elems = line.split()
      example_list.append(
        {'scene_id': int(elems[0]), 'im_id': int(elems[1])})
  return example_list


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='jpeg', channels=3):
    """Class constructor.

    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Number of image channels.
    """
    with tf.Graph().as_default():
      self._decode_data = tf.placeholder(dtype=tf.string)
      self._image_format = image_format
      self._session = tf.Session()
      if self._image_format in ('jpeg', 'jpg'):
        self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
      elif self._image_format == 'png':
        self._decode = tf.image.decode_png(self._decode_data, channels=channels)

  def read_image_dims(self, image_data):
    """Reads the image dimensions.

    Args:
      image_data: String of image data.
    Returns:
      Image_height and image_width.
    """
    image = self.decode_image(image_data)
    return image.shape[:2]

  def decode_image(self, image_data):
    """Decodes the image data string.

    Args:
      image_data: String of image data.
    Returns:
      Decoded image data.
    Raises:
      ValueError: Value of image channels not supported.
    """
    image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
    if len(image.shape) != 3 or image.shape[2] not in (1, 3):
      raise ValueError('The image channels not supported.')

    return image


def int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_list_feature(values):
  """Returns a TF-Feature of float_list.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, list):
    values = [values]

  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(val) for val in values]))
