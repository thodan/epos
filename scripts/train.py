# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""A script for training.

Example:
python train.py --model=ycbv-bop20-xc65-f64
"""

import os
import tensorflow as tf
from tensorflow.python.ops import math_ops
from epos_lib import common
from epos_lib import config
from epos_lib import datagen
from epos_lib import loss
from epos_lib import misc
from epos_lib import model
from epos_lib import train_utils


# Flags (other common flags are defined in epos_lib/common.py; the flag values
# can be defined on the command line or in params.yml in the model folder).
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.
flags.DEFINE_integer(
  'num_clones', 1,
  'Number of clones to deploy.')
flags.DEFINE_boolean(
  'clone_on_cpu', False,
  'Use CPUs to deploy clones.')
flags.DEFINE_integer(
  'num_replicas', 1,
  'Number of worker replicas.')
flags.DEFINE_integer(
  'startup_delay_steps', 15,
  'Number of training steps between replicas startup.')
flags.DEFINE_integer(
  'num_ps_tasks', 0,
  'The number of parameter servers. If the value is 0, then the parameters are '
  'handled locally by the worker.')
flags.DEFINE_string(
  'master', '',
  'BNS name of the tensorflow server')
flags.DEFINE_integer(
  'task', 0,
  'The task ID.')

# Settings for logging.
flags.DEFINE_integer(
  'log_steps', 100,
  'Display logging information at every log_steps.')
flags.DEFINE_integer(
  'save_interval_steps', 50000,
  'How often, in number of training steps, we save the model to disk.')
flags.DEFINE_integer(
  'max_checkpoints_to_keep', 40,
  'The maximum number of model checkpoints to keep.')
flags.DEFINE_integer(
  'save_summaries_secs', 1200,
  'How often, in seconds, we compute the summaries.')

# Settings for profiling.
flags.DEFINE_string(
  'profile_logdir', None,
  'Where the profile files are stored.')

# Settings for training.
flags.DEFINE_float(
  'obj_cls_loss_weight', 1.0,
  'Weight of the object classification loss.')
flags.DEFINE_float(
  'frag_cls_loss_weight', 1.0,
  'Weight of the fragment classification loss.')
flags.DEFINE_float(
  'frag_loc_loss_weight', 100.0,
  'Weight of the fragment localization loss.')
flags.DEFINE_integer(
  'gt_knn_frags', 1,
  'Number of the closest fragments to which a point on the model surface is '
  'assigned when generating the ground-truth.')
flags.DEFINE_multi_string(
  'freeze_regex_list', None,
  'Variables matching any of the specified regular expressions will be frozen.')
flags.DEFINE_enum(
  'learning_policy', 'poly', ['poly', 'step'],
  'Learning rate policy for training.')
flags.DEFINE_float(
  'base_learning_rate', 0.0001,
  'The base learning rate for model training.')
flags.DEFINE_float(
  'learning_rate_decay_factor', 0.1,
  'The rate to decay the base learning rate.')
flags.DEFINE_integer(
  'learning_rate_decay_step', 2000,
  'Decay the base learning rate at a fixed step.')
flags.DEFINE_float(
  'learning_power', 0.9,
  'The power value used in the poly learning policy.')
flags.DEFINE_integer(
  'train_steps', 2000000,
  'The number of steps used for training')
flags.DEFINE_float(
  'momentum', 0.9,
  'The momentum value to use')
flags.DEFINE_integer(
  'train_batch_size', 1,
  'The number of images in each batch during training.')
# Use 0.00004 for MobileNet-V2 or Xception model and 0.0001 for ResNet model.
flags.DEFINE_float(
  'weight_decay', 0.00004,
  'The value of the weight decay for training.')
flags.DEFINE_integer(
  'train_max_height_before_crop', '480',
  'Maximum image height before cropping (the image is downscaled if larger).')
flags.DEFINE_list(
  'train_crop_size', '640,480',
  'Image size [width, height] during training.')
flags.DEFINE_float(
  'last_layer_gradient_multiplier', 1.0,
  'The gradient multiplier for last layers, which is used to boost the '
  'gradient of last layers if the value > 1.')

# Settings for fine-tuning the model.
flags.DEFINE_string(
  'initial_checkpoint', None,
  'The initial checkpoint in the TensorFlow format specified by a path relative'
  'to config.TF_MODELS_PATH.')
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean(
  'initialize_last_layer', True,
  'Initialize the last layer.')
flags.DEFINE_boolean(
  'last_layers_contain_logits_only', False,
  'Only consider logits as last layers or not.')
flags.DEFINE_integer(
  'slow_start_step', 0,
  'Training model with small learning rate for few steps.')
flags.DEFINE_float(
  'slow_start_learning_rate', 0.0001,
  'Learning rate employed during slow start.')
# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean(
  'fine_tune_batch_norm', False,
  'Fine tune the batch norm parameters or not.')

# Dataset settings.
flags.DEFINE_list(
  'train_tfrecord_names', None,
  'Names of tfrecord files (without suffix) used for training.')

# Data augmentations.
flags.DEFINE_string(
  'data_augmentations', None,
  'A string containing a dictionary in the YAML format.')
# ------------------------------------------------------------------------------


def _build_epos_model(
      iterator, num_objs, num_frags, ignore_obj_label,
      outputs_to_num_channels):
  """Builds the EPOS model.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_objs: Number of objects.
    num_frags: Number of surface fragments per object.
    ignore_obj_label: Object segmentation label to be ignored.
    outputs_to_num_channels: A map from output type to the number of channels.
      For example, for the task of object segmentation with 21 classes, we would
      have outputs_to_num_channels['pred_obj_cls'] = 21.
  """
  samples = iterator.get_next()

  # Add names to nodes so we can add them to summary.
  for key, val in samples.items():
    samples[key] = tf.identity(val, name=key)

  model_options = common.ModelOptions(
      outputs_to_num_channels=outputs_to_num_channels,
      crop_size=list(map(int, FLAGS.train_crop_size)),
      atrous_rates=FLAGS.atrous_rates,
      encoder_output_stride=FLAGS.encoder_output_stride)

  # Calculate logits.
  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      image_pyramid=FLAGS.image_pyramid,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

  # Object classification loss.
  loss.add_obj_cls_loss(
    scales_to_logits=outputs_to_scales_to_logits[common.PRED_OBJ_CONF],
    targets=samples[common.GT_OBJ_LABEL],
    num_classes=outputs_to_num_channels[common.PRED_OBJ_CONF],
    ignore_obj_label=ignore_obj_label,
    loss_weight=FLAGS.obj_cls_loss_weight,
    upsample_logits=FLAGS.upsample_logits,
    scope=common.PRED_OBJ_CONF)

  # Fragment classification loss.
  loss.add_frag_cls_loss(
    scales_to_logits=outputs_to_scales_to_logits[common.PRED_FRAG_CONF],
    targets=samples[common.GT_FRAG_LABEL],
    target_weights=samples[common.GT_FRAG_WEIGHT],
    obj_cls_map=samples[common.GT_OBJ_LABEL],
    ignore_obj_label=ignore_obj_label,
    num_objs=num_objs,
    num_frags=num_frags,
    class_agnostic=FLAGS.frag_cls_agnostic,
    loss_weight=FLAGS.frag_cls_loss_weight,
    upsample_logits=FLAGS.upsample_logits,
    scope=common.PRED_FRAG_CONF)

  # Fragment localization loss.
  loss.add_frag_loc_loss(
    scales_to_logits=outputs_to_scales_to_logits[common.PRED_FRAG_LOC],
    frag_cls_targets=samples[common.GT_FRAG_LABEL],
    frag_loc_targets=samples[common.GT_FRAG_LOC],
    frag_weights=samples[common.GT_FRAG_WEIGHT],
    obj_cls_map=samples[common.GT_OBJ_LABEL],
    ignore_obj_label=ignore_obj_label,
    num_objs=num_objs,
    num_frags=num_frags,
    class_agnostic=FLAGS.frag_loc_agnostic,
    loss_weight=FLAGS.frag_loc_loss_weight,
    upsample_logits=FLAGS.upsample_logits,
    scope=common.PRED_FRAG_LOC)

  # Add summaries for model variables.
  for model_var in tf.model_variables():
    tf.summary.histogram(model_var.op.name, model_var)


def _tower_loss(
      iterator, num_objs, num_frags, ignore_obj_label, scope, reuse_variable):
  """Calculates the total loss on a single tower running the EPOS model.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_objs: Number of objects.
    num_frags: Number of surface fragments per object.
    ignore_obj_label: Object segmentation label to be ignored.
    scope: Unique prefix string identifying the deeplab tower.
    reuse_variable: If the variable should be reused.

  Returns:
     The total loss for a batch of data.
  """
  with tf.variable_scope(
      tf.get_variable_scope(), reuse=True if reuse_variable else None):

    # A map from output type to the number of channels.
    outputs_to_num_channels =\
      common.get_outputs_to_num_channels(num_objs, num_frags)

    # Builds the computational graph for the EPOS model.
    _build_epos_model(
      iterator, num_objs, num_frags, ignore_obj_label,
      outputs_to_num_channels)

  # Collect losses and create summaries.
  losses = tf.losses.get_losses(scope=scope)
  for loss in losses:
    tf.summary.scalar('losses/%s' % loss.op.name, loss)

  # Regularization loss.
  regularization_loss = tf.losses.get_regularization_loss(scope=scope)
  tf.summary.scalar('losses/%s' % regularization_loss.op.name,
                    regularization_loss)

  # Total loss.
  total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
  total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
  tf.summary.scalar('losses/total_loss', total_loss)

  return total_loss


def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.

  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads


def _train_epos_model(iterator,
                      num_objs,
                      num_frags,
                      ignore_obj_label,
                      freeze_regex_list):
  """Trains the EPOS model.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and annotations.
    num_objs: Number of objects.
    num_frags: Number of surface fragments per object.
    ignore_obj_label: Object segmentation label to be ignored.

  Returns:
    train_tensor: A tensor to update the model variables.
    summary_op: An operation to log the summaries.
  """
  global_step = tf.train.get_or_create_global_step()

  learning_rate = train_utils.get_model_learning_rate(
      FLAGS.learning_policy, FLAGS.base_learning_rate,
      FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
      FLAGS.train_steps, FLAGS.learning_power,
      FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

  tower_losses = []
  tower_grads = []
  for i in range(FLAGS.num_clones):
    with tf.device('/gpu:%d' % i):
      # First tower has default name scope.
      name_scope = ('clone_%d' % i) if i else ''
      with tf.name_scope(name_scope) as scope:
        loss = _tower_loss(
            iterator=iterator,
            num_objs=num_objs,
            num_frags=num_frags,
            ignore_obj_label=ignore_obj_label,
            scope=scope,
            reuse_variable=(i != 0))
        tower_losses.append(loss)

  for i in range(FLAGS.num_clones):
    with tf.device('/gpu:%d' % i):
      name_scope = ('clone_%d' % i) if i else ''
      with tf.name_scope(name_scope) as scope:
        grads = optimizer.compute_gradients(tower_losses[i])
        tower_grads.append(grads)

  with tf.device('/cpu:0'):
    grads_and_vars = _average_gradients(tower_grads)

    # Freeze some variables (by setting their gradients to zero).
    if freeze_regex_list is not None:
      grads_and_vars = train_utils.freeze_gradients_matching_regex(
        grads_and_vars, freeze_regex_list, invert=True)

    # Multiply gradients for biases and last layer variables.
    if FLAGS.last_layer_gradient_multiplier != 1.0:
      last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = tf.contrib.training.multiply_gradients(
            grads_and_vars, grad_mult)

    # Create gradient update op.
    grad_updates = optimizer.apply_gradients(
      grads_and_vars, global_step=global_step)

    # Collect update ops.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)

    # Print losses to the terminal.
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    print_inputs = ['step:', global_step, 'total_loss:', total_loss]
    losses = tf.losses.get_losses(scope=scope)
    for loss in losses:
      name = loss.op.name
      for short_name in ['obj_cls', 'frag_cls', 'frag_loc']:
        if short_name in name:
          name = short_name
          break
      print_inputs += [name + ':', loss]

    should_log = math_ops.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
    print_op = tf.cond(
        should_log,
        lambda: tf.print(*print_inputs),
        lambda: tf.no_op())

    # Add a summary for the total loss.
    # tf.summary.scalar('losses/total_loss', total_loss)
    with tf.control_dependencies([update_op, print_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Excludes summaries from towers other than the first one.
    summary_op = tf.summary.merge_all(scope='(?!clone_)')

  return train_tensor, summary_op


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Model folder.
  model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)

  # Update flags with parameters loaded from the model folder.
  common.update_flags(os.path.join(model_dir, common.PARAMS_FILENAME))

  # Print the flag values.
  common.print_flags()

  # Folder where the checkpoint and training logs are stored.
  train_dir = os.path.join(model_dir, 'train')
  tf.gfile.MakeDirs(train_dir)

  # TFRecord files used for training.
  tfrecord_names = FLAGS.train_tfrecord_names
  if not isinstance(FLAGS.train_tfrecord_names, list):
    tfrecord_names = [FLAGS.train_tfrecord_names]

  # Stride of the final output.
  if FLAGS.upsample_logits:
    # The stride is 1 if the logits are upsampled to the input resolution.
    output_stride = 1
  else:
    assert (len(FLAGS.decoder_output_stride) == 1)
    output_stride = FLAGS.decoder_output_stride[0]

  # Per-clone batch size.
  if FLAGS.train_batch_size % FLAGS.num_clones != 0:
    raise ValueError('Batch size not divisible by number of clones (GPUs).')
  clone_batch_size = FLAGS.train_batch_size // FLAGS.num_clones

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.num_ps_tasks)):

      # Dataset provider.
      dataset = datagen.Dataset(
          dataset_name=FLAGS.dataset,
          tfrecord_names=tfrecord_names,
          model_dir=model_dir,
          model_variant=FLAGS.model_variant,
          batch_size=clone_batch_size,
          max_height_before_crop=FLAGS.train_max_height_before_crop,
          crop_size=list(map(int, FLAGS.train_crop_size)),
          num_frags=FLAGS.num_frags,
          min_visib_fract=FLAGS.min_visib_fract,
          gt_knn_frags=FLAGS.gt_knn_frags,
          output_stride=output_stride,
          is_training=True,
          return_gt_orig=False,
          return_gt_maps=True,
          should_shuffle=True,
          should_repeat=True,
          prepare_for_projection=False,
          data_augmentations=FLAGS.data_augmentations)

      # Construct the training graph.
      train_tensor, summary_op = _train_epos_model(
        iterator=dataset.get_one_shot_iterator(),
        num_objs=dataset.num_objs,
        num_frags=dataset.num_frags,
        ignore_obj_label=dataset.ignore_obj_label,
        freeze_regex_list=FLAGS.freeze_regex_list)

      # Soft placement allows placing on CPU ops without GPU implementation.
      tf_config = tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False)

      # tf_config.gpu_options.allow_growth = True  # Only necessary GPU memory.
      tf_config.gpu_options.allow_growth = False

      # Nodes that can use multiple threads to parallelize their execution
      # will schedule the individual pieces into this pool.
      tf_config.intra_op_parallelism_threads = 3

      # All ready nodes are scheduled in this pool.
      tf_config.inter_op_parallelism_threads = 3

      # Last available checkpoint for the current model.
      checkpoint_path = tf.train.latest_checkpoint(train_dir)

      # Function to initialize from initial_checkpoint (if no checkpoint for the
      # current model exists yet).
      init_fn = None
      if FLAGS.initial_checkpoint and not checkpoint_path:
        last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
        init_fn = train_utils.get_model_init_fn(
            train_dir,
            FLAGS.initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True)

      # Get dictionary of variables to restore from the last checkpoint.
      var_dict = None
      if init_fn is None:
        var_dict = misc.get_variable_dict()

      # Training scaffold.
      scaffold = tf.train.Scaffold(
          init_fn=init_fn,
          summary_op=summary_op,
          saver=tf.train.Saver(
            max_to_keep=FLAGS.max_checkpoints_to_keep,
            var_list=var_dict))

      # Hook for stopping after the specified number of training steps.
      stop_hook = tf.train.StopAtStepHook(
          last_step=FLAGS.train_steps)

      # Create a folder for profiling.
      profile_dir = FLAGS.profile_logdir
      if profile_dir is not None:
        tf.gfile.MakeDirs(profile_dir)

      # Start training.
      with tf.contrib.tfprof.ProfileContext(
            enabled=profile_dir is not None, profile_dir=profile_dir):
        with tf.train.MonitoredTrainingSession(
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            config=tf_config,
            scaffold=scaffold,
            checkpoint_dir=train_dir,
            summary_dir=train_dir,
            log_step_count_steps=FLAGS.log_steps,
            save_summaries_steps=FLAGS.save_summaries_secs,
            save_checkpoint_steps=FLAGS.save_interval_steps,
            hooks=[stop_hook]
        ) as sess:
          while not sess.should_stop():
            sess.run([train_tensor])


if __name__ == '__main__':
  tf.app.run()
