# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""A script for evaluation.

Note that this script currently evaluates only object segmentation. The
evaluation results are logged into TF summaries which can be visualized with
TensorBoard (see script launch_tensorboard.py).

Use script infer.py if you want to estimate 6D object poses and save them to
a format expected by the BOP evaluation system (http://bop.felk.cvut.cz/).

Example:
python eval.py --model=ycbv-bop20-xc65-f64
"""

import os
import time
import json
import tensorflow as tf
from epos_lib import common, model
from epos_lib import misc
from epos_lib import eval_utils
from epos_lib import config
from epos_lib import datagen


# Flags (other common flags are defined in epos_lib/common.py; the flag values
# can be defined on the command line or in params.yml in the model folder).
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'master', '',
  'BNS name of the tensorflow server')
flags.DEFINE_integer(
  'eval_max_height_before_crop', '480',
  'Maximum image height before cropping (the image is downscaled if larger).')
flags.DEFINE_list(
  'eval_crop_size', '640,480',
  'Image size [height, width] for evaluation.')
flags.DEFINE_integer(
  'eval_interval_secs', 3600,
  'How often (in seconds) to run evaluation.')
flags.DEFINE_list(
  'eval_tfrecord_names', None,
  'Names of tfrecord files (without suffix) used for evaluation')
# ------------------------------------------------------------------------------


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Model folder.
  model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)

  # Update flags with parameters loaded from the model folder.
  common.update_flags(os.path.join(model_dir, common.PARAMS_FILENAME))

  # Print the flag values.
  common.print_flags()

  # Folder where the checkpoint and training logs are stored.
  checkpoint_dir = os.path.join(model_dir, 'train')

  # Path to the latest checkpoint.
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

  # Folder where the evaluation results will be stored.
  eval_dir = os.path.join(model_dir, 'eval')
  tf.gfile.MakeDirs(eval_dir)

  # Skip the evaluation if this checkpoint has already been evaluated or
  # not enough time has passed since the last evaluation.
  last_evaluation_fpath = os.path.join(eval_dir, 'last_evaluation.json')
  if os.path.exists(last_evaluation_fpath):
    with open(last_evaluation_fpath, 'r') as f:
      eval_info = json.load(f)

      if checkpoint_path == eval_info['checkpoint_path']:
        tf.logging.info(
          'Skipping evaluation (checkpoint {} has been evaluated).'.format(
            checkpoint_path))
        return

      seconds_from_last_eval = time.time() - eval_info['time']
      if seconds_from_last_eval < FLAGS.eval_interval_secs:
        tf.logging.info(
          'Skipping evaluation (only {} s from the last evaluation).'.format(
            seconds_from_last_eval))
        return

  tf.logging.info('Evaluating on: {}'.format(FLAGS.eval_tfrecord_names))

  split_names = FLAGS.eval_tfrecord_names
  if not isinstance(FLAGS.eval_tfrecord_names, list):
    split_names = [FLAGS.eval_tfrecord_names]

  # Stride of the final output.
  if FLAGS.upsample_logits:
    # The stride is 1 if the logits are upsampled to the input resolution.
    output_stride = 1
  else:
    assert (len(FLAGS.decoder_output_stride) == 1)
    output_stride = FLAGS.decoder_output_stride[0]

  # Only support batch_size = 1, because tensors with GT's for different images
  # are of different sizes (there may be a different number of GT's per image)
  # and cannot be batched together (maybe they could be made sparse).
  batch_size = 1

  dataset = datagen.Dataset(
      dataset_name=FLAGS.dataset,
      tfrecord_names=split_names,
      model_dir=model_dir,
      model_variant=FLAGS.model_variant,
      batch_size=batch_size,
      max_height_before_crop=FLAGS.eval_max_height_before_crop,
      crop_size=list(map(int, FLAGS.eval_crop_size)),
      num_frags=FLAGS.num_frags,
      min_visib_fract=None,
      gt_knn_frags=1,
      output_stride=output_stride,
      is_training=False,
      return_gt_orig=True,
      return_gt_maps=True,
      should_shuffle=False,
      should_repeat=False,
      prepare_for_projection=False,
      data_augmentations=None)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    outputs_to_num_channels = common.get_outputs_to_num_channels(
      dataset.num_objs, dataset.model_store.num_frags)

    model_options = common.ModelOptions(
        outputs_to_num_channels=outputs_to_num_channels,
        crop_size=list(map(int, FLAGS.eval_crop_size)),
        atrous_rates=FLAGS.atrous_rates,
        encoder_output_stride=FLAGS.encoder_output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape([batch_size, int(FLAGS.eval_crop_size[1]),
                                     int(FLAGS.eval_crop_size[0]), 3])

    predictions = model.predict(
      images=samples[common.IMAGE],
      model_options=model_options,
      upsample_logits=FLAGS.upsample_logits,
      image_pyramid=FLAGS.image_pyramid,
      num_objs=dataset.num_objs,
      num_frags=dataset.num_frags,
      frag_cls_agnostic=FLAGS.frag_cls_agnostic,
      frag_loc_agnostic=FLAGS.frag_loc_agnostic)

    # Tensors used in the evaluation hook.
    samples_keys_for_eval = [
      # common.K,
      common.GT_OBJ_LABEL,
      # common.GT_OBJ_IDS,
      # common.GT_OBJ_MASKS,
      # common.GT_OBJ_QUATS,
      # common.GT_OBJ_TRANS,
      # common.GT_FRAG_LABEL,
      # common.GT_FRAG_LOC,
    ]
    pred_keys_for_eval = [
      common.PRED_OBJ_LABEL,
      # common.PRED_OBJ_CONF,
      # common.PRED_FRAG_CONF,
      # common.PRED_FRAG_LOC,
    ]
    tensors_for_eval = {}
    for key in samples_keys_for_eval:
      if key in samples:
        tensors_for_eval[key] = samples[key]
    for key in pred_keys_for_eval:
      if key in predictions:
        tensors_for_eval[key] = predictions[key]

    # Evaluation hook.
    eval_summary_hook = eval_utils.EvalHook(
      log_dir=eval_dir,
      tensors_for_eval=tensors_for_eval,
      num_objs=dataset.num_objs,
      ignore_label=dataset.ignore_obj_label)

    # Summary hook.
    summary_op = tf.summary.merge_all()
    summary_hook = tf.contrib.training.SummaryAtEndHook(
      log_dir=eval_dir, summary_op=summary_op)

    # State printing hook.
    class StatePrinterHook(tf.train.SessionRunHook):

      def __init__(self):
        self.batch_counter = 0

      def before_run(self, run_context):
        self.batch_counter += 1
        if self.batch_counter % 100 == 0:
          tf.logging.info('Evaluating batch {}'.format(self.batch_counter))

    hooks = [
      eval_summary_hook,
      summary_hook,
      StatePrinterHook(),
      # tf.contrib.training.StopAfterNEvalsHook(10)
    ]

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # Only necessary GPU memory.

    # Nodes that can use multiple threads to parallelize their execution will
    # schedule the individual pieces into this pool.
    tf_config.intra_op_parallelism_threads = 5

    # All ready nodes are scheduled in this pool.
    tf_config.inter_op_parallelism_threads = 5

    scaffold = tf.train.Scaffold(
      saver=tf.train.Saver(var_list=misc.get_variable_dict()))

    tf.contrib.training.evaluate_once(
      config=tf_config,
      scaffold=scaffold,
      master=FLAGS.master,
      checkpoint_path=checkpoint_path,
      eval_ops=[tf.identity(0)],
      hooks=hooks)

    # Save info about the finished evaluation.
    with open(last_evaluation_fpath, 'w') as f:
      eval_info = {
        'time': time.time(),
        'checkpoint_path': checkpoint_path
      }
      json.dump(eval_info, f)


if __name__ == '__main__':
  tf.app.run()
