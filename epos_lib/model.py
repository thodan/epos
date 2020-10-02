# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

r"""Provides EPOS model definition and helper functions.

Based on DeepLab model:
https://github.com/tensorflow/models/blob/master/research/deeplab/model.py
"""

import tensorflow as tf
from epos_lib import common
from epos_lib import feature
from epos_lib import misc

slim = tf.contrib.slim


LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

DECODER_SCOPE = 'decoder'


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
        META_ARCHITECTURE_SCOPE,
    ]


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast(
      (tf.cast(dim, tf.float32) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def reshape_logits(
      logits_name, logits, num_objs, num_frags, frag_cls_agnostic,
      frag_loc_agnostic):
  """Reshapes logits of the specified type.

    Args:
      logits_name: Name of the type of logits.
      logits: Tensor with logits.
      num_objs: Number of objects.
      num_frags: Number of surface fragments per object.
      frag_cls_agnostic: Whether fragment classification is object agnostic.
      frag_loc_agnostic: Whether fragment localization is object agnostic.

    Returns:
      Reshaped logits.
    """
  # Fragment confidences.
  if logits_name == common.PRED_FRAG_CONF:
    pred_classes = 1 if frag_cls_agnostic else num_objs
    shape = tf.shape(logits)
    logits = tf.reshape(
      logits, [shape[0], shape[1], shape[2], pred_classes, num_frags])

  # 3D fragment coordinates.
  elif logits_name == common.PRED_FRAG_LOC:
    pred_classes = 1 if frag_loc_agnostic else num_objs
    shape = tf.shape(logits)
    logits = tf.reshape(
      logits, [shape[0], shape[1], shape[2], pred_classes, num_frags, 3])

  return logits


def extract_encoder_features(images,
                             model_options,
                             weight_decay=0.0001,
                             reuse=None,
                             is_training=False,
                             fine_tune_batch_norm=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and encoder_output_stride.
    end_points: A dictionary from components of the model to the corresponding
      activation.
  """
  features, end_points = feature.extract_features(
      images=images,
      encoder_output_stride=model_options.encoder_output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      divisible_by=model_options.divisible_by,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      preprocessed_images_dtype=model_options.preprocessed_images_dtype,
      fine_tune_batch_norm=fine_tune_batch_norm,
      use_bounded_activation=model_options.use_bounded_activation)

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    # The following code employs the DeepLabv3 ASPP module. Note that we
    # could express the ASPP module as one particular dense prediction
    # cell architecture. We do not do so but leave the following code
    # for backward compatibility.
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    activation_fn = (
        tf.nn.relu6 if model_options.use_bounded_activation else tf.nn.relu)

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):

      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        depth = 256
        branch_logits = []

        if model_options.add_image_level_feature:

          # Global pooling.
          image_feature = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
          resize_height = tf.shape(features)[1]
          resize_width = tf.shape(features)[2]
          image_feature = slim.conv2d(
              image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
          image_feature = misc.resize_bilinear(
              image_feature, [resize_height, resize_width], image_feature.dtype)

          if isinstance(resize_height, tf.Tensor):
              resize_height = None
          if isinstance(resize_width, tf.Tensor):
              resize_width = None
          image_feature.set_shape([None, resize_height, resize_width, depth])
          branch_logits.append(image_feature)

        # Employ a 1x1 convolution.
        branch_logits.append(
          slim.conv2d(features, depth, 1, scope=ASPP_SCOPE + str(0)))

        if model_options.atrous_rates:
          # Employ 3x3 convolutions with different atrous rates.
          for i, rate in enumerate(model_options.atrous_rates, 1):
            scope = ASPP_SCOPE + str(i)
            if model_options.aspp_with_separable_conv:
              aspp_features = split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            branch_logits.append(aspp_features)

        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = slim.conv2d(
            concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
        concat_logits = slim.dropout(
            concat_logits,
            keep_prob=0.9,
            is_training=is_training,
            scope=CONCAT_PROJECTION_SCOPE + '_dropout')

        return concat_logits, end_points


def extract_decoder_features(features,
                             end_points,
                             im_size=None,
                             decoder_output_stride=None,
                             decoder_use_separable_conv=False,
                             model_variant=None,
                             weight_decay=0.0001,
                             reuse=None,
                             is_training=False,
                             fine_tune_batch_norm=False,
                             use_bounded_activation=False):
  """Adds the decoder to obtain sharper segmentation results.

  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the model to the corresponding
      activation.
    im_size: A tuple [im_width, im_height] specifying whole image size.
    decoder_output_stride: A list of integers specifying the output stride of
      low-level features used in the decoder module.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].

  Raises:
    ValueError: If im_size is None.
  """
  if im_size is None:
    raise ValueError('im_size must be provided when using decoder.')
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu6 if use_bounded_activation else tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):

    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):

        decoder_features = features
        decoder_stage = 0
        scope_suffix = ''

        for stride in decoder_output_stride:
          feature_list = feature.networks_to_feature_maps[
              model_variant][feature.DECODER_END_POINTS][stride]

          # If only one decoder stage, we do not change the scope name in
          # order for backward compactibility.
          if decoder_stage:
            scope_suffix = '_{}'.format(decoder_stage)

          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]

            # MobileNet and NAS variants use different naming convention.
            if 'mobilenet' in model_variant or model_variant.startswith('nas'):
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature.name_scope[model_variant], name)

            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name], 48, 1,
                    scope='feature_projection' + str(i) + scope_suffix))

            # Determine the output size.
            decoder_width = scale_dimension(im_size[0], 1.0 / stride)
            decoder_height = scale_dimension(im_size[1], 1.0 / stride)

            # Resize to decoder_height/decoder_width.
            for j, feat in enumerate(decoder_features_list):
              decoder_features_list[j] = misc.resize_bilinear(
                  feat, [decoder_height, decoder_width], feat.dtype)
              h = (None if isinstance(decoder_height, tf.Tensor)
                   else decoder_height)
              w = (None if isinstance(decoder_width, tf.Tensor)
                   else decoder_width)
              decoder_features_list[j].set_shape([None, h, w, None])

            decoder_depth = 256
            if decoder_use_separable_conv:

              decoder_features = split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0' + scope_suffix)

              decoder_features = split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1' + scope_suffix)

            else:
              num_convs = 2
              decoder_features = slim.repeat(
                  tf.concat(decoder_features_list, 3), num_convs,
                  slim.conv2d, decoder_depth, 3,
                  scope='decoder_conv' + str(i) + scope_suffix)

          decoder_stage += 1
        return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
  """Gets the logits from each network's branch.

  The underlying network is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].

  Raises:
    ValueError: Upon invalid input kernel_size value.
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):

    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):

      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)


def get_logits(images,
               model_options,
               weight_decay=0.0001,
               reuse=None,
               is_training=False,
               fine_tune_batch_norm=False):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features, end_points = extract_encoder_features(
      images=images,
      model_options=model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if model_options.decoder_output_stride is not None:
    features = extract_decoder_features(
        features=features,
        end_points=end_points,
        im_size=model_options.crop_size,
        decoder_output_stride=model_options.decoder_output_stride,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        use_bounded_activation=model_options.use_bounded_activation)

  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_channels):
    outputs_to_logits[output] = get_branch_logits(
        features=features,
        num_classes=model_options.outputs_to_num_channels[output],
        atrous_rates=model_options.atrous_rates,
        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
        kernel_size=model_options.logits_kernel_size,
        weight_decay=weight_decay,
        reuse=reuse,
        scope_suffix=output)

  return outputs_to_logits


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
  """Gets the logits for multi-scale inputs.

  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      object classification) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `image_pyramid` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]

  im_shape = misc.resolve_shape(images, 4)
  im_height = im_shape[1]
  im_width = im_shape[2]

  # Compute the height and width for the output logits.
  if model_options.decoder_output_stride:
    logits_output_stride = min(model_options.decoder_output_stride)
  else:
    # No decoder is used.
    logits_output_stride = model_options.encoder_output_stride

  logits_height = scale_dimension(
      im_height, max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      im_width, max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {
      k: {} for k in model_options.outputs_to_num_channels}
  num_channels = images.get_shape().as_list()[-1]
  for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = scale_dimension(im_height, image_scale)
      scaled_width = scale_dimension(im_width, image_scale)
      scaled_im_size = [scaled_height, scaled_width]
      scaled_images = misc.resize_bilinear(images, scaled_im_size, images.dtype)
      if model_options.crop_size:
        scaled_images.set_shape(
            [None, scaled_height, scaled_width, num_channels])
    else:
      scaled_im_size = model_options.crop_size
      scaled_images = images

    updated_options = model_options._replace(crop_size=scaled_im_size)
    outputs_to_logits = get_logits(
        images=scaled_images,
        model_options=updated_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    # Resize the logits to have the same dimension before merging.
    for output in sorted(outputs_to_logits):
      outputs_to_logits[output] = misc.resize_bilinear(
          outputs_to_logits[output], [logits_height, logits_width],
          outputs_to_logits[output].dtype)

    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_channels):
        outputs_to_scales_to_logits[output][
            MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_channels):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(model_options.outputs_to_num_channels):

    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        tf.expand_dims(logits, axis=4)
        for logits in outputs_to_scales_to_logits[output].values()
    ]
    all_logits = tf.concat(all_logits, 4)

    if model_options.merge_method == 'max':
      merge_fn = tf.reduce_max
    else:
      merge_fn = tf.reduce_mean

    outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=4)

  return outputs_to_scales_to_logits


def predict(
      images, model_options, upsample_logits, image_pyramid, num_objs,
      num_frags, frag_cls_agnostic, frag_loc_agnostic):
  """Prediction.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    upsample_logits: Whether to upsample logits to the input resolution.
    image_pyramid: Input image scales for multi-scale feature extraction.
    num_objs: Number of objects.
    num_frags: Number of surface fragments per object.
    frag_cls_agnostic: Whether fragment classification is object agnostic.
    frag_loc_agnostic: Whether fragment localization is object agnostic.

  Returns:
    A dictionary with keys specifying the output_type and values storing Tensors
    representing predictions.
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images=images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  outputs_to_predictions = {}
  for output_name in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output_name]
    logits = scales_to_logits[MERGED_LOGITS_SCOPE]

    # Bilinear upsampling of logits.
    if upsample_logits:
      logits = misc.resize_bilinear(
        images=logits,
        shape=tf.shape(images)[1:3],
        output_dtype=scales_to_logits[MERGED_LOGITS_SCOPE].dtype)

    # Reshape the logits.
    logits = reshape_logits(
        logits_name=output_name,
        logits=logits,
        num_objs=num_objs,
        num_frags=num_frags,
        frag_cls_agnostic=frag_cls_agnostic,
        frag_loc_agnostic=frag_loc_agnostic)

    # Apply softmax to classification outputs.
    if output_name in [common.PRED_OBJ_CONF, common.PRED_FRAG_CONF]:
      logits = tf.nn.softmax(logits)

    # Postprocess object classification output.
    if output_name == common.PRED_OBJ_CONF:
      outputs_to_predictions[common.PRED_OBJ_CONF] = logits
      outputs_to_predictions[common.PRED_OBJ_LABEL] = tf.argmax(logits, -1)
    else:
      outputs_to_predictions[output_name] = logits

  return outputs_to_predictions
