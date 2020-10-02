# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Provides common names and flags used throughout the code."""

import os
import copy
import collections
import yaml
import tensorflow as tf


# Common names.
# ------------------------------------------------------------------------------
# 6D object pose estimation tasks (see Hodan et al. ECCVW 2018).
LOCALIZATION = 'localization'
DETECTION = 'detection'

# Pose fitting methods.
PROGRESSIVE_X = 'progressive_x'
OPENCV_RANSAC = 'opencv_ransac'

# Predictions.
PRED_OBJ_LABEL = 'pred_obj_label'  # Object labels.
PRED_OBJ_CONF = 'pred_obj_conf'  # Object confidences.
PRED_FRAG_CONF = 'pred_frag_conf'  # Fragment confidences.
PRED_FRAG_LOC = 'pred_frag_loc'  # Precise 3D locations on fragments.
PRED_CORRESP = 'pred_corresp'  # 2D-3D correspondences.

# Ground-truth output.
GT_OBJ_LABEL = 'gt_obj_label'  # Object labels.
GT_FRAG_LABEL = 'gt_frag_label'  # Fragment labels.
GT_FRAG_LOC = 'gt_frag_loc'  # Precise 3D locations on fragments.
GT_FRAG_WEIGHT = 'gt_frag_weight'

# Lists of ground truth annotations (elements at the same position in the lists
# belong to the same annotated object instance).
GT_OBJ_IDS = 'gt_obj_ids'  # List of object ID's.
GT_OBJ_VISIB_FRACT = 'gt_obj_visib_fract'  # List of visibility fractions.
GT_OBJ_MASKS = 'gt_obj_masks'  # List of masks of object instances.
GT_OBJ_QUATS = 'gt_obj_quats'  # List of quaternions.
GT_OBJ_TRANS = 'gt_obj_trans'  # List of translation vectors.

# Other common names.
K = 'K'
IMAGE = 'image'
IMAGE_PATH = 'image_path'
SCENE_ID = 'scene_id'
IM_ID = 'im_id'
TEST_SET = 'test'
PARAMS_FILENAME = 'params.yml'


# Common flags.
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

# EPOS flags.
flags.DEFINE_boolean(
  'frag_cls_agnostic', False,
  'Whether the fragment classification is object agnostic.')
flags.DEFINE_boolean(
  'frag_loc_agnostic', False,
  'Whether the fragment localization is object agnostic.')
flags.DEFINE_integer(
  'num_frags', 64,
  'Number of fragments per object.')
flags.DEFINE_float(
  'min_visib_fract', 0.1,
  'Minimum visibility to consider an annotated object instance.')
flags.DEFINE_float(
  'corr_min_obj_conf', 0.1,
  'Threshold on the object confidence (tau_a in the EPOS paper).')
flags.DEFINE_float(
  'corr_min_frag_rel_conf', 0.5,
  'Threshold on the relative fragment confidence (tau_b in the EPOS paper).')
flags.DEFINE_boolean(
  'corr_project_to_model', False,
  'Whether to project the predicted points to the object model.')

# Other flags.
flags.DEFINE_string(
  'model', None,
  'Name of the model.')
flags.DEFINE_string(
  'dataset', None,
  'Name of the dataset.')
# See feature.py for supported model variants.
flags.DEFINE_string(
  'model_variant', 'xception_65',
  'Deep model variant.')
flags.DEFINE_integer(
  'logits_kernel_size', 1,
  'The kernel size for the convolutional kernel that generates logits.')
flags.DEFINE_multi_float(
  'image_pyramid', None,
  'Input scales for multi-scale feature extraction.')
flags.DEFINE_boolean(
  'add_image_level_feature', True,
  'Add image level feature.')
flags.DEFINE_list(
  'image_pooling_stride', '1,1',
  'Image pooling stride [height, width] used in the ASPP image pooling. ')
flags.DEFINE_boolean(
  'aspp_with_batch_norm', True,
  'Use batch norm parameters for ASPP or not.')
flags.DEFINE_boolean(
  'aspp_with_separable_conv', True,
  'Use separable convolution for ASPP or not.')
# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer(
  'multi_grid', None,
  'Employ a hierarchy of atrous rates for ResNet.')
flags.DEFINE_float(
  'depth_multiplier', 1.0,
  'Multiplier for the depth (number of channels) for all convolution ops used '
  'in MobileNet.')
flags.DEFINE_integer(
  'divisible_by', None,
  'An integer that ensures the layer # channels are divisible by this value. '
  'Used in MobileNet.')
flags.DEFINE_multi_integer(
  'atrous_rates', [12, 24, 36],
  'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_list(
  'decoder_output_stride', [4],
  'Comma-separated list of strings with the number specifying output stride of '
  'low-level features at each model level. Current implementation assumes at '
  'most one output stride (i.e., either None or a list with only one element). '
  'If None, decoder is not used.')
flags.DEFINE_integer(
  'encoder_output_stride', 8,
  'The ratio of input to encoder output spatial resolution.')
flags.DEFINE_boolean(
  'decoder_use_separable_conv', True,
  'Employ separable convolution for decoder or not.')
flags.DEFINE_enum(
  'merge_method', 'max', ['max', 'avg'],
  'Scheme to merge multi scale features.')
flags.DEFINE_boolean(
  'prediction_with_upsampled_logits', True,
  'When performing prediction, there are two options: (1) bilinear upsampling '
  'the logits followed by argmax, or (2) armax followed by nearest upsampling '
  'the predicted labels. The second option may introduce some'
  '"blocking effect", but it is more computationally efficient.')
flags.DEFINE_bool(
  'use_bounded_activation', False,
  'Whether or not to use bounded activations. Bounded activations better lend '
  'themselves to quantized inference.')
flags.DEFINE_boolean(
  'upsample_logits', False,
  'Whether to upsample logits.')


def update_flags(model_params_path):
  """Updates flags with values loaded from a YAML file.

  Args:
    model_params_path: Path to a YAML file.
  """
  if not os.path.exists(model_params_path):
    return

  tf.logging.info('Loading flags from: {}'.format(model_params_path))
  if os.path.basename(model_params_path).split('.')[1] not in ['yml', 'yaml']:
    raise ValueError('Only YAML format is currently supported.')

  with open(model_params_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.CLoader)
  for par_name, par_val in params.items():
    if par_name in FLAGS.__flags.keys():
      if par_name in ['train_crop_size', 'infer_crop_size', 'eval_crop_size']:
        FLAGS.__flags[par_name].value = [int(x) for x in par_val.split(',')]
      else:
        FLAGS.__flags[par_name].value = par_val


def print_flags():
  """Prints all flags and their values."""
  tf.logging.info('Flags:')
  tf.logging.info('----------')
  for flag_name, flag_value in FLAGS.__flags.items():
    tf.logging.info('{}: {}'.format(flag_name, flag_value.value))
  tf.logging.info('----------')


def get_outputs_to_num_channels(num_objs, num_frags):
  """Returns a map from output type to the number of associated channels.

  Args:
    num_objs: Number of objects.
    num_frags: Number of surface fragments per object.
  """
  return {
    PRED_OBJ_CONF:
        num_objs + 1,
    PRED_FRAG_CONF:
        (1 if FLAGS.frag_cls_agnostic else num_objs) * num_frags,
    PRED_FRAG_LOC:
        (1 if FLAGS.frag_cls_agnostic else num_objs) * num_frags * 3,
  }


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_channels',
        'crop_size',
        'atrous_rates',
        'encoder_output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'use_bounded_activation'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_channels,
              crop_size=None,
              atrous_rates=None,
              encoder_output_stride=8,
              preprocessed_images_dtype=tf.float32):
    """Constructor to set default values.

    Args:
      outputs_to_num_channels: A dictionary from output type to the number of
        classes. For example, for the task of object segmentation with 21
        classes, we would have outputs_to_num_channels['semantic']=21.
      crop_size: A tuple [im_height, im_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      encoder_output_stride: The ratio of input to encoder output resolution.
      preprocessed_images_dtype: The type after the preprocessing function.

    Returns:
      A new ModelOptions instance.
    """
    decoder_output_stride = None
    if FLAGS.decoder_output_stride:
      decoder_output_stride = [int(x) for x in FLAGS.decoder_output_stride]
      if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
        raise ValueError('Decoder output stride need to be sorted in the '
                         'descending order.')

    image_pooling_stride = [1, 1]
    if FLAGS.image_pooling_stride:
      image_pooling_stride = [int(x) for x in FLAGS.image_pooling_stride]

    return super(ModelOptions, cls).__new__(
        cls,
        outputs_to_num_channels,
        crop_size,
        atrous_rates,
        encoder_output_stride,
        preprocessed_images_dtype,
        FLAGS.merge_method,
        FLAGS.add_image_level_feature,
        image_pooling_stride,
        FLAGS.aspp_with_batch_norm,
        FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid,
        decoder_output_stride,
        FLAGS.decoder_use_separable_conv,
        FLAGS.logits_kernel_size,
        FLAGS.model_variant,
        FLAGS.depth_multiplier,
        FLAGS.divisible_by,
        FLAGS.prediction_with_upsampled_logits,
        FLAGS.use_bounded_activation)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.outputs_to_num_channels),
                        self.crop_size,
                        self.atrous_rates,
                        self.encoder_output_stride,
                        self.preprocessed_images_dtype)
