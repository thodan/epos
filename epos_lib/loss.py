# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Utility functions for training."""

import six
import tensorflow as tf
from epos_lib import misc


def resize_logits(logits, target_shape):
  """Resizes a tensor with logits.

  Args:
    logits: Tensor with logits.
    target_shape: New size.
  Returns:
    Tensor with resized logits.
  """
  return tf.image.resize_bilinear(logits, target_shape, align_corners=True)


def create_index_map(
      batch_size, height, width, seg_mask, agnostic=False, fg_only=True):
  """Creates index map with (batch, y, x, cls_id) at each pixel.

  :param batch_size: Batch size.
  :param height: Map height.
  :param width: Map width.
  :param seg_mask: Segmentation mask of shape [batch_size, height, width].
  :param agnostic: Whether the index map should be class agnostic.
  :param fg_only: If true, the first foreground class ID is 0, otherwise 1.
  :return: Index map of shape [batch_size, height, width, 4].
  """
  # Batch map of shape [batch_size, height, width]. All elements of
  # [batch, :, :] have value == batch.
  batch_map = tf.tile(
    tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1, 1]),
    [1, height, width])

  # Map of XY coordinates of shape [im_height, im_width, 2].
  coords_map = tf.stack(tf.meshgrid(tf.range(width), tf.range(height)), axis=2)

  # XY map to YX map.
  coords_map = tf.reverse(coords_map, axis=[2])

  # Replicate the grid of XY coordinates to [batch_size, height, width, 2].
  coords_map = tf.cast(tf.tile(
    tf.expand_dims(coords_map, axis=0), [batch_size, 1, 1, 1]), tf.int32)

  # Mask of class ID's.
  if agnostic:
    cls_map = tf.cast(tf.not_equal(seg_mask, 0), tf.int32)
  else:
    cls_map = tf.cast(seg_mask, tf.int32)

  if fg_only:
    cls_map -= 1

  # Concatenate the batch map, the coordinate map and the segmentation mask
  # to get the index map of shape [batch_size, height, width, 4].
  index_map = tf.concat(
    [tf.expand_dims(batch_map, axis=3), coords_map,
     tf.expand_dims(cls_map, axis=3)], axis=3)

  return index_map


def get_fg_indices(mask, ignore_label, class_agnostic):
  """Collects indices (batch, y, x, obj_id) of the foreground examples.

  :param mask: [batch_size, height, width]
  :param ignore_label: Label to ignore.
  :param class_agnostic: Whether the indices should be class agnostic.
  :return: [num_idx, 4]
  """
  # Get shape.
  shape = misc.resolve_shape(mask, 3)
  batch_size, height, width = shape[0], shape[1], shape[2]

  # Get foreground mask of shape [batch_size, height, width].
  fg_mask = tf.logical_and(
    tf.not_equal(mask, 0), tf.not_equal(mask, ignore_label))
  fg_mask = tf.reshape(fg_mask, [batch_size, height, width])

  # Create an index map of shape [batch_size, height, width, 4].
  index_map = create_index_map(
    batch_size, height, width, mask, class_agnostic, fg_only=True)

  # Indices (batch, y, x, obj_id) of shape [num_idx, 4] of logits in fg mask.
  pred_idx = tf.boolean_mask(index_map, fg_mask)

  # Indices of shape [num_idx, 3] of GT targets corresponding to the fg mask.
  gt_idx = tf.slice(pred_idx, [0, 0], [-1, 3])

  return pred_idx, gt_idx


def add_obj_cls_loss(scales_to_logits,
                     targets,
                     num_classes,
                     ignore_obj_label,
                     loss_weight=1.0,
                     upsample_logits=True,
                     scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    targets: Groundtruth labels of shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_obj_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if targets is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      targets_shape = misc.resolve_shape(targets, 4)[1:3]
      logits = resize_logits(logits, targets_shape)

    not_ignore_mask = tf.not_equal(targets, ignore_obj_label)

    one_hot_targets = tf.one_hot(
      targets, num_classes, on_value=1.0, off_value=0.0)

    not_ignore_mask = tf.reshape(not_ignore_mask, shape=[-1])
    one_hot_targets = tf.reshape(one_hot_targets, shape=[-1, num_classes])

    # Compute the loss for all pixels.
    losses = tf.losses.softmax_cross_entropy(
      one_hot_targets,
      tf.reshape(logits, shape=[-1, num_classes]),
      weights=tf.cast(not_ignore_mask, tf.float32) * loss_weight,
      scope=loss_scope,
      loss_collection=None,
      reduction='none')
    loss = tf.reduce_mean(losses)
    tf.losses.add_loss(tf.identity(loss, name='obj_cls_loss'))


def add_frag_cls_loss(scales_to_logits,
                      targets,
                      target_weights,
                      obj_cls_map,
                      ignore_obj_label,
                      num_objs,
                      num_frags,
                      class_agnostic=False,
                      loss_weight=1.0,
                      upsample_logits=True,
                      scope=None):
  """Adds loss for fragment classification of each scale.
  """
  # Get indices of the foreground examples.
  pred_idx, gt_idx =\
    get_fg_indices(obj_cls_map, ignore_obj_label, class_agnostic)

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    # Shape of targets.
    shape = misc.resolve_shape(targets, 4)

    # Upsample logits.
    if upsample_logits:
      logits = resize_logits(logits, shape)

    # Reshape logits.
    num_cls = 1 if class_agnostic else num_objs
    logits = tf.reshape(
      logits, [shape[0], shape[1], shape[2], num_cls, 1, num_frags])
    top_k = shape[3]  # Number of assigned GT fragments.
    logits = tf.tile(logits, [1, 1, 1, 1, top_k, 1])

    # Gather the foreground examples.
    # [num_elems, top_k]
    targets_fg = tf.gather_nd(targets, gt_idx)
    target_weights_fg = tf.gather_nd(target_weights, gt_idx)
    # [num_elems, top_k, num_frags]
    logits_fg = tf.gather_nd(logits, pred_idx)

    # Target distributions of shape [num_elems * top_k, num_frags].
    num_rows = tf.size(targets_fg)
    sparse_indices = tf.cast(tf.concat(
      [tf.reshape(tf.range(num_rows), [-1, 1]),
       tf.reshape(targets_fg, [-1, 1])], axis=1), tf.int64)
    distrib_targets_fg = tf.sparse.to_dense(tf.SparseTensor(
      indices=sparse_indices,
      values=tf.reshape(target_weights_fg, [-1]),
      dense_shape=[num_rows, num_frags]
    ), default_value=0, validate_indices=True)

    # Make sure that each row represents a proper distribution.
    distrib_targets_fg = tf.divide(
      distrib_targets_fg,
      tf.reduce_sum(distrib_targets_fg, axis=1, keepdims=True))

    # Logits of shape [num_elems * top_k, num_frags].
    logits_fg = tf.reshape(logits_fg, shape=[-1, num_frags])

    # Compute the loss.
    losses = tf.losses.softmax_cross_entropy(
      onehot_labels=distrib_targets_fg,
      logits=logits_fg,
      weights=loss_weight,
      scope=loss_scope,
      loss_collection=None,
      reduction='none')

    loss = tf.reduce_mean(losses)

    # Set loss to zero if there are no foreground examples (to avoid nan).
    loss = tf.cond(tf.greater(tf.size(pred_idx), 0), lambda: loss, lambda: 0.0)

    tf.losses.add_loss(tf.identity(loss, name='frag_cls_loss'))


def add_frag_loc_loss(scales_to_logits,
                      frag_cls_targets,
                      frag_weights,
                      frag_loc_targets,
                      obj_cls_map,
                      ignore_obj_label,
                      num_objs,
                      num_frags,
                      class_agnostic=False,
                      loss_weight=1.0,
                      upsample_logits=True,
                      scope=None):
  """Adds loss for fragment localization of each scale.
  """
  # Get indices of the foreground examples.
  pred_idx, gt_idx =\
    get_fg_indices(obj_cls_map, ignore_obj_label, class_agnostic)

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    # Shape of targets.
    shape = misc.resolve_shape(frag_loc_targets, 5)

    # Upsample logits.
    if upsample_logits:
      logits = resize_logits(logits, shape)

    # Reshape logits.
    num_cls = 1 if class_agnostic else num_objs
    logits = tf.reshape(
      logits, [shape[0], shape[1], shape[2], num_cls, num_frags, 3])

    # Gather the fragment targets and weights [num_elems, top_k].
    frag_cls_targets_fg = tf.gather_nd(frag_cls_targets, gt_idx)
    frag_weights_fg = tf.gather_nd(frag_weights, gt_idx)

    # Gather the localization targets of shape [num_elems, top_k, 3].
    frag_loc_targets_fg = tf.gather_nd(frag_loc_targets, gt_idx)

    # Reshape targets to [num_elems * top_k, 3].
    frag_loc_targets_fg = tf.reshape(frag_loc_targets_fg, shape=[-1, 3])

    # Extend the prediction indices by the index of the fragment.
    top_k = shape[3]  # Number of the assigned GT fragments.
    idx_size = tf.shape(pred_idx)[1]
    pred_idx = tf.reshape(tf.tile(pred_idx, [1, top_k]), [-1, idx_size])
    pred_idx = tf.concat(
      [pred_idx, tf.reshape(frag_cls_targets_fg, [-1, 1])], axis=1)

    # Gather logits of shape [num_elems * top_k, 3].
    logits_fg = tf.gather_nd(logits, pred_idx)

    # Compute Smooth-L1 loss.
    losses = tf.losses.huber_loss(
      labels=frag_loc_targets_fg,
      predictions=logits_fg,
      weights=loss_weight,
      scope=loss_scope,
      loss_collection=None,
      reduction='none')

    # Weight the losses.
    losses *= tf.reshape(frag_weights_fg, [-1, 1])
    loss = tf.reduce_mean(losses)

    # Set loss to zero if there are no foreground examples (to avoid nan).
    loss = tf.cond(tf.greater(tf.size(pred_idx), 0), lambda: loss, lambda: 0.0)

    tf.losses.add_loss(tf.identity(loss, name='frag_loc_loss'))
