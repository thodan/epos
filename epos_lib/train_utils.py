# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Utility functions for training."""

import os
import tensorflow as tf
from epos_lib import config
from epos_lib import misc


def freeze_gradients_matching_regex(grads_and_vars, regex_list, invert=True):
  """Freezes gradients whose variable names match a regular expression.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    regex_list: A list of string regular expressions.
    invert: If True, the complement of the filtered set is frozen.
  Returns:
    A filtered list of gradient to variable pairs (tuples).
  """
  variables = [pair[1] for pair in grads_and_vars]
  matching_vars = misc.filter_variables(variables, regex_list, invert=invert)
  kept_grads_and_vars = [pair for pair in grads_and_vars
                         if pair[1] not in matching_vars]
  for var in matching_vars:
    tf.logging.info('Freezing variable [%s]', var.op.name)
  return kept_grads_and_vars


def get_model_init_fn(train_logdir,
                      initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization, other checkpoint exists.')
    return None

  initial_checkpoint_path =\
    os.path.join(config.TF_MODELS_PATH, initial_checkpoint)
  tf.logging.info('Initializing model from path: %s', initial_checkpoint_path)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = tf.contrib.framework.get_variables_to_restore(
      exclude=exclude_list)

  if variables_to_restore:
    init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
        initial_checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
    global_step = tf.train.get_or_create_global_step()

    def restore_fn(unused_scaffold, sess):
      sess.run(init_op, init_feed_dict)
      sess.run([global_step])

    return restore_fn

  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model variables.
  To fine-tune the models, we usually set larger (e.g., 10 times larger)
  learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in tf.model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            train_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none'):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / train_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    train_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  adjusted_global_step = global_step

  if slow_start_burnin_type != 'none':
    adjusted_global_step -= slow_start_step

  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        adjusted_global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        adjusted_global_step,
        train_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  adjusted_slow_start_learning_rate = slow_start_learning_rate
  if slow_start_burnin_type == 'linear':
    # Do linear burnin. Increase linearly from slow_start_learning_rate and
    # reach base_learning_rate after (global_step >= slow_start_steps).
    adjusted_slow_start_learning_rate = (
        slow_start_learning_rate +
        (base_learning_rate - slow_start_learning_rate) *
        tf.cast(global_step, tf.float32) / slow_start_step)
  elif slow_start_burnin_type != 'none':
    raise ValueError('Unknown burnin type.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step,
                  adjusted_slow_start_learning_rate, learning_rate)
