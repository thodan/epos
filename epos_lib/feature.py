# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Extracts features for different model variants.

Based on: deeplab/core/feature_extractor.py
"""

import functools
import tensorflow as tf

# From external/slim/nets:
from nets import resnet_utils
from nets.mobilenet import mobilenet_v2

from epos_lib import net_xception
from epos_lib import net_resnet_v1_beta

slim = tf.contrib.slim


# Default end point for MobileNetv2.
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'

# Names for end point features.
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features.
networks_to_feature_maps = {
    'mobilenet_v2': {
        DECODER_END_POINTS: {
            4: ['layer_4/depthwise_output'],
        },
    },
    'resnet_v1_50': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_50_beta': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_101': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_101_beta': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
        },
    },
    'xception_41': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
    'xception_65': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
    'xception_71': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block3/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
}


def _mobilenet_v2(net,
                  depth_multiplier,
                  output_stride,
                  divisible_by=None,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
  """Auxiliary function to add support for 'reuse' to mobilenet_v2.

  Args:
    net: Input tensor of shape [batch_size, height, width, channels].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the model from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    reuse: Reuse model variables.
    scope: Optional variable scope.
    final_endpoint: The endpoint to construct the model up to.

  Returns:
    Features extracted by MobileNetv2.
  """
  if divisible_by is None:
    divisible_by = 8 if depth_multiplier == 1.0 else 1
  with tf.variable_scope(
      scope, 'MobilenetV2', [net], reuse=reuse) as scope:
    return mobilenet_v2.mobilenet_base(
        net,
        conv_defs=mobilenet_v2.V2_DEF,
        depth_multiplier=depth_multiplier,
        min_depth=8 if depth_multiplier == 1.0 else 1,
        divisible_by=divisible_by,
        final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT,
        output_stride=output_stride,
        scope=scope)


# A map from network name to network function.
networks_map = {
    'mobilenet_v2': _mobilenet_v2,
    'resnet_v1_50': net_resnet_v1_beta.resnet_v1_50,
    'resnet_v1_50_beta': net_resnet_v1_beta.resnet_v1_50_beta,
    'resnet_v1_101': net_resnet_v1_beta.resnet_v1_101,
    'resnet_v1_101_beta': net_resnet_v1_beta.resnet_v1_101_beta,
    'xception_41': net_xception.xception_41,
    'xception_65': net_xception.xception_65,
    'xception_71': net_xception.xception_71,
}

# A map from network name to network arg scope.
arg_scopes_map = {
    'mobilenet_v2': mobilenet_v2.training_scope,
    'resnet_v1_50': resnet_utils.resnet_arg_scope,
    'resnet_v1_50_beta': resnet_utils.resnet_arg_scope,
    'resnet_v1_101': resnet_utils.resnet_arg_scope,
    'resnet_v1_101_beta': resnet_utils.resnet_arg_scope,
    'xception_41': net_xception.xception_arg_scope,
    'xception_65': net_xception.xception_arg_scope,
    'xception_71': net_xception.xception_arg_scope,
}

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_50_beta': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
    'xception_41': 'xception_41',
    'xception_65': 'xception_65',
    'xception_71': 'xception_71',
}

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  num_channels = tf.shape(inputs)[-1]
  # We set mean pixel as 0 for the non-RGB channels.
  mean_rgb_extended = tf.concat(
      [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
  return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, tf.float32) - 1.0
  return tf.cast(preprocessed_inputs, dtype=dtype)


_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
}


def get_network(network_name, preprocess_images,
                preprocessed_images_dtype=tf.float32, arg_scope=None):
  """Gets the network.

  Args:
    network_name: Network name.
    preprocess_images: Preprocesses the images or not.
    preprocessed_images_dtype: The type after the preprocessing function.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.

  Returns:
    A network function that is used to extract features.

  Raises:
    ValueError: network is not supported.
  """
  if network_name not in networks_map:
    raise ValueError('Unsupported network %s.' % network_name)
  arg_scope = arg_scope or arg_scopes_map[network_name]()
  def _identity_function(inputs, dtype=preprocessed_images_dtype):
    return tf.cast(inputs, dtype=dtype)
  if preprocess_images:
    preprocess_function = _PREPROCESS_FN[network_name]
  else:
    preprocess_function = _identity_function
  func = networks_map[network_name]
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(preprocess_function(inputs, preprocessed_images_dtype),
                  *args, **kwargs)
  return network_fn


def extract_features(images,
                     encoder_output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     divisible_by=None,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     preprocessed_images_dtype=tf.float32,
                     num_classes=None,
                     global_pool=False,
                     use_bounded_activation=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    encoder_output_stride: The ratio of input to encoder output resolution.
    multi_grid: Employ a hierarchy of different atrous rates within the model.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the model up to.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    preprocess_images: Performs preprocessing on images or not. Defaults to
      True. Set to False if preprocessing will be done by other functions. We
      supprot two types of preprocessing: (1) Mean pixel substraction and (2)
      Pixel values normalization to be [-1, 1].
    preprocessed_images_dtype: The type after the preprocessing function.
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference. Currently,
      bounded activation is only used in xception model.

  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and encoder_output_stride.
    end_points: A dictionary from components of the model to the corresponding
      activation.

  Raises:
    ValueError: Unrecognized model variant.
  """
  if 'resnet' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay,
        batch_norm_decay=0.95,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True)

    features, end_points = get_network(
        model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=encoder_output_stride,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=name_scope[model_variant])

  elif 'xception' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=1e-3,
        batch_norm_scale=True,
        regularize_depthwise=regularize_depthwise,
        use_bounded_activation=use_bounded_activation)

    features, end_points = get_network(
        model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=encoder_output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=name_scope[model_variant])

  elif 'mobilenet' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        is_training=(is_training and fine_tune_batch_norm),
        weight_decay=weight_decay)

    features, end_points = get_network(
        model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(
            inputs=images,
            depth_multiplier=depth_multiplier,
            divisible_by=divisible_by,
            output_stride=encoder_output_stride,
            reuse=reuse,
            scope=name_scope[model_variant],
            final_endpoint=final_endpoint)

  elif model_variant.startswith('nas'):
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=1e-3)

    features, end_points = get_network(
        model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=encoder_output_stride,
            reuse=reuse,
            scope=name_scope[model_variant])

  else:
    raise ValueError('Unknown network variant %s.' % model_variant)

  return features, end_points
