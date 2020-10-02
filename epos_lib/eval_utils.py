# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Utility functions for evaluation."""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
from tensorflow.python.training import training_util
from bop_toolkit_lib import misc
from epos_lib import common


class EvalHook(tf.train.SessionRunHook):
  """An evaluation hook (currently evaluates only object segmentation).

  Based on SummaryAtEndHook from:
  tensorflow/contrib/training/python/training/eval_utils.py
  """

  def __init__(
        self, log_dir, tensors_for_eval, num_objs, ignore_label):

    """Constructs the evaluation hook.

    Args:
      log_dir: The directory where the evaluation results (including summary
        events) are saved to.
      tensors_for_eval: Tensors to be evaluated in the hook.
      num_objs: Number of objects.
      ignore_label: Segmentation label to be ignored.
    """
    self.log_dir = log_dir
    self.tensors_for_eval = tensors_for_eval
    self.num_objs = num_objs
    self.ignore_label = ignore_label

    self.global_step = None
    self.summary_writer = None

    # Confusion matrix of per-pixel object classification.
    self.num_cls = self.num_objs + 1
    self.cm = np.zeros((self.num_cls, self.num_cls), np.int64)

  def begin(self):
    self.global_step = training_util.get_or_create_global_step()

  def after_create_session(self, session, coord):
    self.summary_writer = tf.summary.FileWriterCache.get(self.log_dir)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self.tensors_for_eval)

  def after_run(self, run_context, run_values):
    output = run_values.results
    num_imgs = output[common.PRED_OBJ_LABEL].shape[0]

    for im_id in range(num_imgs):

      # Mask of valid pixels.
      valid_mask = output[common.GT_OBJ_LABEL][im_id] != self.ignore_label

      # Confusion matrix.
      gt_labels = output[common.GT_OBJ_LABEL][im_id][valid_mask].flatten()
      pred_labels = output[common.PRED_OBJ_LABEL][im_id][valid_mask].flatten()
      inds = np.stack([gt_labels, pred_labels], axis=0).T
      unique, unique_counts = np.unique(inds, return_counts=True, axis=0)
      self.cm[unique[:, 0], unique[:, 1]] += unique_counts

  def add_scalar_summary(self, name, value, global_step):
    tf.logging.info('Adding summary: {} = {}'.format(name, value))
    summary = tf.Summary(value=[
      tf.Summary.Value(tag=name, simple_value=value)])
    self.summary_writer.add_summary(summary, global_step)

  def end(self, session):
    global_step = training_util.global_step(session, self.global_step)

    # Save the confusion matrix to a text file.
    df = pd.DataFrame(self.cm)
    cm_table = tabulate(df, headers='keys', tablefmt='psql')
    cm_path = os.path.join(self.log_dir, 'cm_{}.txt'.format(global_step))
    misc.ensure_dir(os.path.dirname(cm_path))
    with open(cm_path, 'w') as f:
      f.write(cm_table)

    # Calculate mIoU of object segmentation.
    bg_iou = 1.0
    fg_ious = []
    for cls in range(self.num_cls):
      intersection = self.cm[cls, cls]
      union = np.sum(self.cm[cls, :]) + np.sum(self.cm[:, cls]) - intersection
      if union > 0:
        iou = intersection / float(union)
        if cls == 0:
          bg_iou = iou
        else:
          fg_ious.append(iou)

    if len(fg_ious):
      miou_fg = np.mean(fg_ious)
      miou_all = np.mean(fg_ious + [bg_iou])
    else:
      miou_fg = 0.0
      miou_all = 0.0

    # mIoU calculated over foreground and background classes.
    self.add_scalar_summary('eval/obj_cls_miou_all', miou_all, global_step)

    # mIoU calculated only over foreground classes.
    self.add_scalar_summary('eval/obj_cls_miou_fg', miou_fg, global_step)

    self.summary_writer.flush()
