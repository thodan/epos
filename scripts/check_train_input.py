# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""A script for checking the training input.

Example:
python check_train_input.py --model=ycbv-bop20-xc65-f64
"""

import os
import time
import numpy as np
import tensorflow as tf
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc as misc_bop
from bop_toolkit_lib import transform
from epos_lib import common
from epos_lib import config
from epos_lib import datagen
from epos_lib import misc
from epos_lib import vis


# Flags (other common flags are defined in epos_lib/common.py; the flag values
# can be defined on the command line or in params.yml in the model folder).
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(
  'num_batches_to_check', 10,
  'Number of batches to check.')
flags.DEFINE_boolean(
  'print_shapes', False,
  'Whether to print shapes of unput tensors.')
flags.DEFINE_integer(
  'gt_knn_frags', 1,
  'Number of the closest fragments to which a point on the model surface is '
  'assigned when generating the ground-truth.')
flags.DEFINE_integer(
  'train_max_height_before_crop', '480',
  'Maximum image height before cropping.')
flags.DEFINE_list(
  'train_crop_size', '640,480',
  'Image size [width, height] during training.')
flags.DEFINE_list(
  'train_tfrecord_names', None,
  'Names of tfrecord files (without suffix) used for training.')
flags.DEFINE_string(
  'data_augmentations', None,
  'A string containing a dictionary in the YAML format.')
# ------------------------------------------------------------------------------


def check_gt_frag(samples, im_id, output_scale, model_store):

  # Consider the first (i.e. the closest) fragment.
  frag_id = 0

  for gt_id, obj_id in enumerate(samples[common.GT_OBJ_IDS][im_id]):

    # GT object mask.
    obj_mask = samples[common.GT_OBJ_MASKS][im_id][gt_id]

    # Get fragment label and coordinates.
    frag_ids = samples[common.GT_FRAG_LABEL][im_id][obj_mask][:, frag_id]
    frag_centers = model_store.frag_centers[obj_id][frag_ids]
    frag_coords = samples[common.GT_FRAG_LOC][im_id][obj_mask][:, frag_id, :]

    # Scale coordinates by the fragment sizes.
    frag_scales = model_store.frag_sizes[obj_id][frag_ids]
    frag_coords *= np.expand_dims(frag_scales, 1)

    # Reconstruct XYZ.
    xyz = frag_centers + frag_coords

    # Coordinates (y, x) of shape [mask_pts, 2].
    field_yx_coords = np.stack(np.nonzero(obj_mask), axis=0).T

    # Convert to (x, y) image coordinates.
    im_coords = np.flip(field_yx_coords, axis=1)
    im_coords =\
      misc.convert_px_indices_to_im_coords(im_coords, 1.0 / output_scale)

    # GT pose.
    K = samples[common.K][im_id].reshape((3, 3))
    R = transform.quaternion_matrix(
      samples[common.GT_OBJ_QUATS][im_id][gt_id])[:3, :3]
    t = samples[common.GT_OBJ_TRANS][im_id][gt_id].reshape((3, 1))

    # Project the 3D points.
    im_coords_proj = misc_bop.project_pts(xyz, K, R, t)

    # Compare the projection and the original image coordinates.
    mean_proj_error =\
      np.mean(np.linalg.norm(im_coords - im_coords_proj, axis=1))

    tf.logging.info(
      'Mean GT reprojection error: {:.5f} px'.format(mean_proj_error))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Model folder.
  model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)

  # Update flags with parameters loaded from the model folder.
  common.update_flags(os.path.join(model_dir, common.PARAMS_FILENAME))

  # Print the flag values.
  common.print_flags()

  # Folder for visualizations.
  vis_dir = os.path.join(model_dir, 'check_train_input')
  tf.gfile.MakeDirs(vis_dir)

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

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=0)):

      # Dataset provider.
      dataset = datagen.Dataset(
          dataset_name=FLAGS.dataset,
          tfrecord_names=tfrecord_names,
          model_dir=model_dir,
          model_variant=FLAGS.model_variant,
          batch_size=1,
          max_height_before_crop=FLAGS.train_max_height_before_crop,
          crop_size=list(map(int, FLAGS.train_crop_size)),
          num_frags=FLAGS.num_frags,
          min_visib_fract=FLAGS.min_visib_fract,
          gt_knn_frags=FLAGS.gt_knn_frags,
          output_stride=output_stride,
          is_training=True,
          return_gt_orig=True,
          return_gt_maps=True,
          should_shuffle=True,
          should_repeat=True,
          prepare_for_projection=False,
          data_augmentations=FLAGS.data_augmentations,
          buffer_size=min(FLAGS.num_batches_to_check, 50))

      iterator = dataset.get_one_shot_iterator()

      # Soft placement allows placing on CPU ops without GPU implementation.
      tf_config = tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False)

      tf_config.gpu_options.allow_growth = True  # Only necessary GPU memory.

      # Nodes that can use multiple threads to parallelize their execution
      # will schedule the individual pieces into this pool.
      tf_config.intra_op_parallelism_threads = 5

      # All ready nodes are scheduled in this pool.
      tf_config.inter_op_parallelism_threads = 5

      # Get input samples.
      samples = iterator.get_next()
      tf.logging.info('Sample keys: {}'.format(samples.keys()))

      sess = tf.Session(config=tf_config)
      tf.logging.info('Preparing input batches...')
      batch_counter = 0
      while True:
        if batch_counter == FLAGS.num_batches_to_check:
          break
        batch_counter += 1

        t_start = time.time()
        out = sess.run(samples)
        tf.logging.info('Image: {}'.format(out[common.IMAGE_PATH]))
        tf.logging.info(
          'Preparation of one batch took: {} s'.format(time.time() - t_start))

        # Print shapes.
        if FLAGS.print_shapes:
          for name, val in out.items():
            tf.logging.info('Shape of {}: {}'.format(name, val.shape))

        # Size of the output fields.
        output_scale = 1.0 / output_stride
        output_size = (
          int(output_scale * dataset.crop_size[0]),
          int(output_scale * dataset.crop_size[1]))

        num_images = out[common.IMAGE].shape[0]
        for i in range(num_images):
          tf.logging.info('Visualization for {}'.format(
            out[common.IMAGE_PATH][i].decode('utf8')))

          image = np.squeeze(out[common.IMAGE][i])

          # Input RGB image.
          # --------------------------------------------------------------------
          inout.save_im(
            os.path.join(vis_dir, '{}_{}_rgb.png'.format(batch_counter, i)),
            image.astype(np.uint8))

          # Object masks.
          # --------------------------------------------------------------------
          for gt_id, obj_mask in enumerate(out[common.GT_OBJ_MASKS][i]):
            obj_id = out[common.GT_OBJ_IDS][i][gt_id]
            fname = '{}_{}_gt_mask_{}_obj_{}.png'.format(
              batch_counter, i, gt_id, obj_id)
            inout.save_im(
              os.path.join(vis_dir, fname), (255 * obj_mask).astype(np.uint8))

          # Object labels.
          # --------------------------------------------------------------------
          gt_obj_labels = np.squeeze(out[common.GT_OBJ_LABEL][i])

          seg_vis_orig =\
            vis.colorize_label_map(gt_obj_labels).astype(np.uint8)
          fname = '{}_{}_gt_obj_label.png'.format(batch_counter, i)
          inout.save_im(
            os.path.join(vis_dir, fname), seg_vis_orig.astype(np.uint8))
          tf.logging.info('Unique object labels: {}'.format(
            set(gt_obj_labels.flatten().tolist())))

          # Fragment labels and coordinates.
          # --------------------------------------------------------------------
          vis_prefix = '{}_{}'.format(batch_counter, i)

          vis.visualize_gt_frag(
            gt_obj_ids=out[common.GT_OBJ_IDS][0],
            gt_obj_masks=out[common.GT_OBJ_MASKS][0],
            gt_frag_labels=out[common.GT_FRAG_LABEL][0],
            gt_frag_weights=out[common.GT_FRAG_WEIGHT][0],
            gt_frag_coords=out[common.GT_FRAG_LOC][0],
            output_size=output_size,
            model_store=dataset.model_store,
            vis_prefix=vis_prefix,
            vis_dir=vis_dir)

          check_gt_frag(out, i, output_scale, dataset.model_store)


if __name__ == '__main__':
  tf.app.run()
