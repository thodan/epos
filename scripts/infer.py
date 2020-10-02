# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""A script for inference/visualization.

Example:
python infer.py --model=ycbv-bop20-xc65-f64
"""

import os
import os.path
import time
import numpy as np
import cv2
import tensorflow as tf
import pyprogressivex
import bop_renderer
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import transform
from bop_toolkit_lib import visualization
from epos_lib import common
from epos_lib import config
from epos_lib import corresp
from epos_lib import datagen
from epos_lib import misc
from epos_lib import model
from epos_lib import vis


# Flags (other common flags are defined in epos_lib/common.py; the flag values
# can be defined on the command line or in params.yml in the model folder).
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'master', '',
  'BNS name of the tensorflow server')
flags.DEFINE_boolean(
  'cpu_only', False,
  'Whether to run the inference on CPU only.')
flags.DEFINE_string(
  'task_type', common.LOCALIZATION,  # LOCALIZATION, DETECTION
  'Type of the 6D object pose estimation task.')
flags.DEFINE_list(
  'infer_tfrecord_names', None,
  'Names of tfrecord files (without suffix) used for inference.')
flags.DEFINE_integer(
  'infer_max_height_before_crop', '480',
  'Maximum image height before cropping (the image is downscaled if larger).')
flags.DEFINE_list(
  'infer_crop_size', '640,480',
  'Image size [height, width] during inference.')
flags.DEFINE_string(
  'checkpoint_name', None,
  'Name of the checkpoint to evaluate (e.g. "model.ckpt-1000000"). The latest '
  'available checkpoint is used if None.')
flags.DEFINE_boolean(
  'project_to_surface', False,
  'Whether to project the predicted 3D locations to the object model.')
flags.DEFINE_boolean(
  'save_estimates', True,
  'Whether to save pose estimates in format expected by the BOP Challenge.')
flags.DEFINE_boolean(
  'save_corresp', False,
  'Whether to save established correspondences to text files.')
flags.DEFINE_string(
  'infer_name', None,
  'Name of the inference used in the filename of the saved estimates.')

# Pose fitting parameters.
flags.DEFINE_string(
  'fitting_method', common.PROGRESSIVE_X,  # PROGRESSIVE_X, OPENCV_RANSAC
  'Pose fitting method.')
flags.DEFINE_float(
  'inlier_thresh', 4.0,
  'Tau_r in the CVPR 2020 paper. Inlier threshold [px] on the '
  'reprojection error.')
flags.DEFINE_float(
  'neighbour_max_dist', 20.0,
  'Tau_d in the CVPR 2020 paper.')
flags.DEFINE_float(
  'min_hypothesis_quality', 0.5,
  'Tau_q in the CVPR 2020 paper')
flags.DEFINE_float(
  'required_progx_confidence', 0.5,
  'The required confidence used to calculate the number of Prog-X iterations.')
flags.DEFINE_float(
  'required_ransac_confidence', 1.0,
  'The required confidence used to calculate the number of RANSAC iterations.')
flags.DEFINE_float(
  'min_triangle_area', 0.0,
  'Tau_t in the CVPR 2020 paper.')
flags.DEFINE_boolean(
  'use_prosac', False,
  'Whether to use the PROSAC sampler.')
flags.DEFINE_integer(
  'max_model_number_for_pearl', 5,
  'Maximum number of instances to optimize by PEARL. PEARL is turned off if '
  'there are more instances to find.')
flags.DEFINE_float(
  'spatial_coherence_weight', 0.1,
  'Weight of the spatial coherence in Graph-Cut RANSAC.')
flags.DEFINE_float(
  'scaling_from_millimeters', 0.1,
  'Scaling factor of 3D coordinates when constructing the neighborhood graph. '
  '0.1 will convert mm to cm. See the CVPR 2020 paper for details.')
flags.DEFINE_float(
  'max_tanimoto_similarity', 0.9,
  'See the Progressive-X paper.')
flags.DEFINE_integer(
  'max_correspondences', None,
  'Maximum number of correspondences to use for fitting. Not applied if None.')
flags.DEFINE_integer(
  'max_instances_to_fit', None,
  'Maximum number of instances to fit. Not applied if None.')
flags.DEFINE_integer(
  'max_fitting_iterations', 400,
  'The maximum number of fitting iterations.')

# Visualization parameters.
flags.DEFINE_boolean(
  'vis', False,
  'Global switch for visualizations.')
flags.DEFINE_boolean(
  'vis_gt_poses', True,
  'Whether to visualize the GT poses.')
flags.DEFINE_boolean(
  'vis_pred_poses', True,
  'Whether to visualize the predicted poses.')
flags.DEFINE_boolean(
  'vis_gt_obj_labels', True,
  'Whether to visualize the GT object labels.')
flags.DEFINE_boolean(
  'vis_pred_obj_labels', True,
  'Whether to visualize the predicted object labels.')
flags.DEFINE_boolean(
  'vis_pred_obj_confs', False,
  'Whether to visualize the predicted object confidences.')
flags.DEFINE_boolean(
  'vis_gt_frag_fields', False,
  'Whether to visualize the GT fragment fields.')
flags.DEFINE_boolean(
  'vis_pred_frag_fields', False,
  'Whether to visualize the predicted fragment fields.')
# ------------------------------------------------------------------------------


def visualize(
      samples, predictions, pred_poses, im_ind, crop_size, output_scale,
      model_store, renderer, vis_dir):
  """Visualizes estimates from one image.

  Args:
    samples: Dictionary with input data.
    predictions: Dictionary with predictions.
    pred_poses: Predicted poses.
    im_ind: Image index.
    crop_size: Image crop size (width, height).
    output_scale: Scale of the model output w.r.t. the input (output / input).
    model_store: Store for 3D object models of class ObjectModelStore.
    renderer: Renderer of class bop_renderer.Renderer().
    vis_dir: Directory where the visualizations will be saved.
  """
  tf.logging.info('Visualization for: {}'.format(
    samples[common.IMAGE_PATH][0].decode('utf8')))

  # Size of a visualization grid tile.
  tile_size = (300, 225)

  # Extension of the saved visualizations ('jpg', 'png', etc.).
  vis_ext = 'jpg'

  # Font settings.
  font_size = 10
  font_color = (0.8, 0.8, 0.8)

  # Intrinsics.
  K = samples[common.K][0]
  output_K = K * output_scale
  output_K[2, 2] = 1.0

  # Tiles for the grid visualization.
  tiles = []

  # Size of the output fields.
  output_size =\
    int(output_scale * crop_size[0]), int(output_scale * crop_size[1])

  # Prefix of the visualization names.
  vis_prefix = '{:06d}'.format(im_ind)

  # Input RGB image.
  rgb = np.squeeze(samples[common.IMAGE][0])
  vis_rgb = visualization.write_text_on_image(
    misc.resize_image_py(rgb, tile_size).astype(np.uint8),
    [{'name': '', 'val': 'input', 'fmt': ':s'}],
    size=font_size, color=font_color)
  tiles.append(vis_rgb)

  # Visualize the ground-truth poses.
  if FLAGS.vis_gt_poses:

    gt_poses = []
    for gt_id, obj_id in enumerate(samples[common.GT_OBJ_IDS][0]):
      q = samples[common.GT_OBJ_QUATS][0][gt_id]
      R = transform.quaternion_matrix(q)[:3, :3]
      t = samples[common.GT_OBJ_TRANS][0][gt_id].reshape((3, 1))
      gt_poses.append({'obj_id': obj_id, 'R': R, 't': t})

    vis_gt_poses = vis.visualize_object_poses(rgb, K, gt_poses, renderer)
    vis_gt_poses = visualization.write_text_on_image(
      misc.resize_image_py(vis_gt_poses, tile_size),
      [{'name': '', 'val': 'gt poses', 'fmt': ':s'}],
      size=font_size, color=font_color)
    tiles.append(vis_gt_poses)

  # Visualize the estimated poses.
  if FLAGS.vis_pred_poses:
    vis_pred_poses = vis.visualize_object_poses(rgb, K, pred_poses, renderer)
    vis_pred_poses = visualization.write_text_on_image(
      misc.resize_image_py(vis_pred_poses, tile_size),
      [{'name': '', 'val': 'pred poses', 'fmt': ':s'}],
      size=font_size, color=font_color)
    tiles.append(vis_pred_poses)

  # Ground-truth object labels.
  if FLAGS.vis_gt_obj_labels and common.GT_OBJ_LABEL in samples:
    obj_labels = np.squeeze(samples[common.GT_OBJ_LABEL][0])
    obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
    obj_labels = vis.colorize_label_map(obj_labels)
    obj_labels = visualization.write_text_on_image(
      misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
      [{'name': '', 'val': 'gt obj labels', 'fmt': ':s'}],
      size=font_size, color=font_color)
    tiles.append(obj_labels)

  # Predicted object labels.
  if FLAGS.vis_pred_obj_labels:
    obj_labels = np.squeeze(predictions[common.PRED_OBJ_LABEL][0])
    obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
    obj_labels = vis.colorize_label_map(obj_labels)
    obj_labels = visualization.write_text_on_image(
      misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
      [{'name': '', 'val': 'predicted obj labels', 'fmt': ':s'}],
      size=font_size, color=font_color)
    tiles.append(obj_labels)

  # Predicted object confidences.
  if FLAGS.vis_pred_obj_confs:
    num_obj_labels = predictions[common.PRED_OBJ_CONF].shape[-1]
    for obj_label in range(num_obj_labels):
      obj_confs = misc.resize_image_py(np.array(
        predictions[common.PRED_OBJ_CONF][0, :, :, obj_label]), tile_size)
      obj_confs = (255.0 * obj_confs).astype(np.uint8)
      obj_confs = np.dstack([obj_confs, obj_confs, obj_confs])  # To RGB.
      obj_confs = visualization.write_text_on_image(
        obj_confs, [{'name': 'cls', 'val': obj_label, 'fmt': ':d'}],
        size=font_size, color=font_color)
      tiles.append(obj_confs)

  # Visualization of ground-truth fragment fields.
  if FLAGS.vis_gt_frag_fields and common.GT_OBJ_IDS in samples:
    vis.visualize_gt_frag(
      gt_obj_ids=samples[common.GT_OBJ_IDS][0],
      gt_obj_masks=samples[common.GT_OBJ_MASKS][0],
      gt_frag_labels=samples[common.GT_FRAG_LABEL][0],
      gt_frag_weights=samples[common.GT_FRAG_WEIGHT][0],
      gt_frag_coords=samples[common.GT_FRAG_LOC][0],
      output_size=output_size,
      model_store=model_store,
      vis_prefix=vis_prefix,
      vis_dir=vis_dir)

  # Visualization of predicted fragment fields.
  if FLAGS.vis_pred_frag_fields:
    vis.visualize_pred_frag(
      frag_confs=predictions[common.PRED_FRAG_CONF][0],
      frag_coords=predictions[common.PRED_FRAG_LOC][0],
      output_size=output_size,
      model_store=model_store,
      vis_prefix=vis_prefix,
      vis_dir=vis_dir,
      vis_ext=vis_ext)

  # Build and save a visualization grid.
  grid = vis.build_grid(tiles, tile_size)
  grid_vis_path = os.path.join(
    vis_dir, '{}_grid.{}'.format(vis_prefix, vis_ext))
  inout.save_im(grid_vis_path, grid)


def save_correspondences(
      scene_id, im_id, im_ind, obj_id, image_path, K, obj_pred, pred_time,
      infer_name, obj_gt_poses, infer_dir):

  # Add meta information.
  txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
  txt += '{}\n'.format(image_path)
  txt += '{} {} {} {}\n'.format(scene_id, im_id, obj_id, pred_time)

  # Add intrinsics.
  for i in range(3):
    txt += '{} {} {}\n'.format(K[i, 0], K[i, 1], K[i, 2])

  # Add ground-truth poses.
  txt += '{}\n'.format(len(obj_gt_poses))
  for pose in obj_gt_poses:
    for i in range(3):
      txt += '{} {} {} {}\n'.format(
        pose['R'][i, 0], pose['R'][i, 1], pose['R'][i, 2], pose['t'][i, 0])

  # Sort the predicted correspondences by confidence.
  sort_inds = np.argsort(obj_pred['conf'])[::-1]
  px_id = obj_pred['px_id'][sort_inds]
  frag_id = obj_pred['frag_id'][sort_inds]
  coord_2d = obj_pred['coord_2d'][sort_inds]
  coord_3d = obj_pred['coord_3d'][sort_inds]
  conf = obj_pred['conf'][sort_inds]
  conf_obj = obj_pred['conf_obj'][sort_inds]
  conf_frag = obj_pred['conf_frag'][sort_inds]

  # Add the predicted correspondences.
  pred_corr_num = len(coord_2d)
  txt += '{}\n'.format(pred_corr_num)
  for i in range(pred_corr_num):
    txt += '{} {} {} {} {} {} {} {} {} {}\n'.format(
      coord_2d[i, 0], coord_2d[i, 1],
      coord_3d[i, 0], coord_3d[i, 1], coord_3d[i, 2],
      px_id[i], frag_id[i], conf[i], conf_obj[i], conf_frag[i])

  # Save the correspondences into a file.
  corr_suffix = infer_name
  if corr_suffix is None:
    corr_suffix = ''
  else:
    corr_suffix = '_' + corr_suffix

  corr_path = os.path.join(
    infer_dir, 'corr{}'.format(corr_suffix),
    '{:06d}_corr_{:02d}.txt'.format(im_ind, obj_id))
  tf.gfile.MakeDirs(os.path.dirname(corr_path))
  with open(corr_path, 'w') as f:
    f.write(txt)


def process_image(
      sess, samples, predictions, im_ind, crop_size, output_scale, model_store,
      renderer, task_type, infer_name, infer_dir, vis_dir):
  """Estimates object poses from one image.

  Args:
    sess: TensorFlow session.
    samples: Dictionary with input data.
    predictions: Dictionary with predictions.
    im_ind: Index of the current image.
    crop_size: Image crop size (width, height).
    output_scale: Scale of the model output w.r.t. the input (output / input).
    model_store: Store for 3D object models of class ObjectModelStore.
    renderer: Renderer of class bop_renderer.Renderer().
    task_type: 6D object pose estimation task (common.LOCALIZATION or
      common.DETECTION).
    infer_name: Name of the current inference.
    infer_dir: Folder for inference results.
    vis_dir: Folder for visualizations.
  """
  # Dictionary for run times.
  run_times = {}

  # Prediction.
  time_start = time.time()
  (samples, predictions) = sess.run([samples, predictions])
  run_times['prediction'] = time.time() - time_start

  # Scene and image ID's.
  scene_id = samples[common.SCENE_ID][0]
  im_id = samples[common.IM_ID][0]

  # Intrinsic parameters.
  K = samples[common.K][0]

  if task_type == common.LOCALIZATION:
    gt_poses = []
    gt_obj_ids = samples[common.GT_OBJ_IDS][0]
    for gt_id in range(len(gt_obj_ids)):
      R = transform.quaternion_matrix(
        samples[common.GT_OBJ_QUATS][0][gt_id])[:3, :3]
      t = samples[common.GT_OBJ_TRANS][0][gt_id].reshape((3, 1))
      gt_poses.append({'obj_id': gt_obj_ids[gt_id], 'R': R, 't': t})
  else:
    gt_poses = None

  # Establish many-to-many 2D-3D correspondences.
  time_start = time.time()
  corr = corresp.establish_many_to_many(
      obj_confs=predictions[common.PRED_OBJ_CONF][0],
      frag_confs=predictions[common.PRED_FRAG_CONF][0],
      frag_coords=predictions[common.PRED_FRAG_LOC][0],
      gt_obj_ids=[x['obj_id'] for x in gt_poses],
      model_store=model_store,
      output_scale=output_scale,
      min_obj_conf=FLAGS.corr_min_obj_conf,
      min_frag_rel_conf=FLAGS.corr_min_frag_rel_conf,
      project_to_surface=FLAGS.project_to_surface,
      only_annotated_objs=(task_type == common.LOCALIZATION))
  run_times['establish_corr'] = time.time() - time_start

  # PnP-RANSAC to estimate 6D object poses from the correspondences.
  time_start = time.time()
  poses = []
  for obj_id, obj_corr in corr.items():
    # tf.logging.info(
    #   'Image path: {}, obj: {}'.format(samples[common.IMAGE_PATH][0], obj_id))

    # Number of established correspondences.
    num_corrs = obj_corr['coord_2d'].shape[0]

    # Skip the fitting if there are too few correspondences.
    min_required_corrs = 6
    if num_corrs < min_required_corrs:
      continue

    # The correspondences need to be sorted for PROSAC.
    if FLAGS.use_prosac:
      sorted_inds = np.argsort(obj_corr['conf'])[::-1]
      for key in obj_corr.keys():
        obj_corr[key] = obj_corr[key][sorted_inds]

    # Select correspondences with the highest confidence.
    if FLAGS.max_correspondences is not None \
          and num_corrs > FLAGS.max_correspondences:
      # Sort the correspondences only if they have not been sorted for PROSAC.
      if FLAGS.use_prosac:
        keep_inds = np.arange(num_corrs)
      else:
        keep_inds = np.argsort(obj_corr['conf'])[::-1]
      keep_inds = keep_inds[:FLAGS.max_correspondences]
      for key in obj_corr.keys():
        obj_corr[key] = obj_corr[key][keep_inds]

    # Save the established correspondences (for analysis).
    if FLAGS.save_corresp:
      obj_gt_poses = []
      if gt_poses is not None:
        obj_gt_poses = [x for x in gt_poses if x['obj_id'] == obj_id]
      pred_time = float(np.sum(list(run_times.values())))
      image_path = samples[common.IMAGE_PATH][0].decode('utf-8')
      save_correspondences(
        scene_id, im_id, im_ind, obj_id, image_path, K, obj_corr, pred_time,
        infer_name, obj_gt_poses, infer_dir)

    # Make sure the coordinates are saved continuously in memory.
    coord_2d = np.ascontiguousarray(obj_corr['coord_2d'].astype(np.float64))
    coord_3d = np.ascontiguousarray(obj_corr['coord_3d'].astype(np.float64))

    if FLAGS.fitting_method == common.PROGRESSIVE_X:
      # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
      # Progressive-X is applied and up to num_instances poses are returned.
      # If num_instances == -1, then Progressive-X is applied and all found
      # poses are returned.
      if task_type == common.LOCALIZATION:
        num_instances = len([x for x in gt_poses if x['obj_id'] == obj_id])
      else:
        num_instances = -1

      if FLAGS.max_instances_to_fit is not None:
        num_instances = min(num_instances, FLAGS.max_instances_to_fit)

      pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
        x1y1=coord_2d,
        x2y2z2=coord_3d,
        K=K,
        threshold=FLAGS.inlier_thresh,
        neighborhood_ball_radius=FLAGS.neighbour_max_dist,
        spatial_coherence_weight=FLAGS.spatial_coherence_weight,
        scaling_from_millimeters=FLAGS.scaling_from_millimeters,
        max_tanimoto_similarity=FLAGS.max_tanimoto_similarity,
        max_iters=FLAGS.max_fitting_iterations,
        conf=FLAGS.required_progx_confidence,
        proposal_engine_conf=FLAGS.required_ransac_confidence,
        min_coverage=FLAGS.min_hypothesis_quality,
        min_triangle_area=FLAGS.min_triangle_area,
        min_point_number=6,
        max_model_number=num_instances,
        max_model_number_for_optimization=FLAGS.max_model_number_for_pearl,
        use_prosac=FLAGS.use_prosac,
        log=False)

      pose_est_success = pose_ests is not None
      if pose_est_success:
        for i in range(int(pose_ests.shape[0] / 3)):
          j = i * 3
          R_est = pose_ests[j:(j + 3), :3]
          t_est = pose_ests[j:(j + 3), 3].reshape((3, 1))
          poses.append({
            'scene_id': scene_id,
            'im_id': im_id,
            'obj_id': obj_id,
            'R': R_est,
            't': t_est,
            'score': pose_qualities[i],
          })

    elif FLAGS.fitting_method == common.OPENCV_RANSAC:
      # This integration of OpenCV-RANSAC can estimate pose of only one object
      # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
      # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
      # in the Progressive-X scheme (as the other methods in that table).
      pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
        objectPoints=coord_3d,
        imagePoints=coord_2d,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=FLAGS.max_fitting_iterations,
        reprojectionError=FLAGS.inlier_thresh,
        confidence=0.99,  # FLAGS.required_ransac_confidence
        flags=cv2.SOLVEPNP_EPNP)

      if pose_est_success:
        poses.append({
          'scene_id': scene_id,
          'im_id': im_id,
          'obj_id': obj_id,
          'R': cv2.Rodrigues(r_est)[0],
          't': t_est,
          'score': 0.0,  # TODO: Define the score.
        })

    else:
      raise ValueError(
        'Unknown pose fitting method ({}).'.format(FLAGS.fitting_method))

  run_times['fitting'] = time.time() - time_start
  run_times['total'] = np.sum(list(run_times.values()))

  # Add the total time to each pose.
  for pose in poses:
    pose['time'] = run_times['total']

  # Visualization.
  if FLAGS.vis:
    visualize(
      samples=samples,
      predictions=predictions,
      pred_poses=poses,
      im_ind=im_ind,
      crop_size=crop_size,
      output_scale=output_scale,
      model_store=model_store,
      renderer=renderer,
      vis_dir=vis_dir)

  return poses, run_times


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Model folder.
  model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)

  # Update flags with parameters loaded from the model folder.
  common.update_flags(os.path.join(model_dir, common.PARAMS_FILENAME))

  # Print the flag values.
  common.print_flags()

  # Folder from which the latest model checkpoint will be loaded.
  checkpoint_dir = os.path.join(model_dir, 'train')

  # Folder for the inference output.
  infer_dir = os.path.join(model_dir, 'infer')
  tf.gfile.MakeDirs(infer_dir)

  # Folder for the visualization output.
  vis_dir = os.path.join(model_dir, 'vis')
  tf.gfile.MakeDirs(vis_dir)

  # TFRecord files used for training.
  tfrecord_names = FLAGS.infer_tfrecord_names
  if not isinstance(FLAGS.infer_tfrecord_names, list):
    tfrecord_names = [FLAGS.infer_tfrecord_names]

  # Stride of the final output.
  if FLAGS.upsample_logits:
    # The stride is 1 if the logits are upsampled to the input resolution.
    output_stride = 1
  else:
    assert (len(FLAGS.decoder_output_stride) == 1)
    output_stride = FLAGS.decoder_output_stride[0]

  with tf.Graph().as_default():

    return_gt_orig = np.any([
      FLAGS.task_type == common.LOCALIZATION,
      FLAGS.vis_gt_poses])

    return_gt_maps = np.any([
      FLAGS.vis_pred_obj_labels,
      FLAGS.vis_pred_obj_confs,
      FLAGS.vis_pred_frag_fields])

    # Dataset provider.
    dataset = datagen.Dataset(
      dataset_name=FLAGS.dataset,
      tfrecord_names=tfrecord_names,
      model_dir=model_dir,
      model_variant=FLAGS.model_variant,
      batch_size=1,
      max_height_before_crop=FLAGS.infer_max_height_before_crop,
      crop_size=list(map(int, FLAGS.infer_crop_size)),
      num_frags=FLAGS.num_frags,
      min_visib_fract=None,
      gt_knn_frags=1,
      output_stride=output_stride,
      is_training=False,
      return_gt_orig=return_gt_orig,
      return_gt_maps=return_gt_maps,
      should_shuffle=False,
      should_repeat=False,
      prepare_for_projection=FLAGS.project_to_surface,
      data_augmentations=None)

    # Initialize a renderer for visualization.
    renderer = None
    if FLAGS.vis_gt_poses or FLAGS.vis_pred_poses:
      tf.logging.info('Initializing renderer for visualization...')

      renderer = bop_renderer.Renderer()
      renderer.init(dataset.crop_size[0], dataset.crop_size[1])

      model_type_vis = 'eval'
      dp_model = dataset_params.get_model_params(
        config.BOP_PATH, dataset.dataset_name, model_type=model_type_vis)
      for obj_id in dp_model['obj_ids']:
        path = dp_model['model_tpath'].format(obj_id=obj_id)
        renderer.add_object(obj_id, path)

      tf.logging.info('Renderer initialized.')

    # Inputs.
    samples = dataset.get_one_shot_iterator().get_next()

    # A map from output type to the number of associated channels.
    outputs_to_num_channels = common.get_outputs_to_num_channels(
      dataset.num_objs, dataset.model_store.num_frags)

    # Options of the neural network model.
    model_options = common.ModelOptions(
        outputs_to_num_channels=outputs_to_num_channels,
        crop_size=list(map(int, FLAGS.infer_crop_size)),
        atrous_rates=FLAGS.atrous_rates,
        encoder_output_stride=FLAGS.encoder_output_stride)

    # Construct the inference graph.
    predictions = model.predict(
        images=samples[common.IMAGE],
        model_options=model_options,
        upsample_logits=FLAGS.upsample_logits,
        image_pyramid=FLAGS.image_pyramid,
        num_objs=dataset.num_objs,
        num_frags=dataset.num_frags,
        frag_cls_agnostic=FLAGS.frag_cls_agnostic,
        frag_loc_agnostic=FLAGS.frag_loc_agnostic)

    # Global step.
    tf.train.get_or_create_global_step()

    # Get path to the model checkpoint.
    if FLAGS.checkpoint_name is None:
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    else:
      checkpoint_path = os.path.join(checkpoint_dir, FLAGS.checkpoint_name)

    time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
    tf.logging.info('Starting inference at: {}'.format(time_str))
    tf.logging.info('Inference with model: {}'.format(checkpoint_path))

    # Scaffold for initialization.
    scaffold = tf.train.Scaffold(
      init_op=tf.global_variables_initializer(),
      saver=tf.train.Saver(var_list=misc.get_variable_dict()))

    # TensorFlow configuration.
    if FLAGS.cpu_only:
      tf_config = tf.ConfigProto(device_count={'GPU': 0})
    else:
      tf_config = tf.ConfigProto()
      # tf_config.gpu_options.allow_growth = True  # Only necessary GPU memory.
      tf_config.gpu_options.allow_growth = False

    # Nodes that can use multiple threads to parallelize their execution will
    # schedule the individual pieces into this pool.
    tf_config.intra_op_parallelism_threads = 10

    # All ready nodes are scheduled in this pool.
    tf_config.inter_op_parallelism_threads = 10

    poses_all = []
    first_im_poses_num = 0

    session_creator = tf.train.ChiefSessionCreator(
        config=tf_config,
        scaffold=scaffold,
        master=FLAGS.master,
        checkpoint_filename_with_path=checkpoint_path)
    with tf.train.MonitoredSession(
          session_creator=session_creator, hooks=None) as sess:

      im_ind = 0
      while not sess.should_stop():

        # Estimate object poses for the current image.
        poses, run_times = process_image(
            sess=sess,
            samples=samples,
            predictions=predictions,
            im_ind=im_ind,
            crop_size=dataset.crop_size,
            output_scale=(1.0 / output_stride),
            model_store=dataset.model_store,
            renderer=renderer,
            task_type=FLAGS.task_type,
            infer_name=FLAGS.infer_name,
            infer_dir=infer_dir,
            vis_dir=vis_dir)

        # Note that the first image takes longer time (because of TF init).
        tf.logging.info(
          'Image: {}, prediction: {:.3f}, establish_corr: {:.3f}, '
          'fitting: {:.3f}, total time: {:.3f}'.format(
            im_ind, run_times['prediction'], run_times['establish_corr'],
            run_times['fitting'], run_times['total']))

        poses_all += poses
        if im_ind == 0:
          first_im_poses_num = len(poses)
        im_ind += 1

    # Set the time of pose estimates from the first image to the average time.
    # Tensorflow takes a long time on the first image (because of init).
    time_avg = 0.0
    for pose in poses_all:
      time_avg += pose['time']
    if len(poses_all) > 0:
      time_avg /= float((len(poses_all)))
    for i in range(first_im_poses_num):
      poses_all[i]['time'] = time_avg

    # Save the estimated poses in the BOP format:
    # https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#formatofresults
    if FLAGS.save_estimates:
      suffix = ''
      if FLAGS.infer_name is not None:
        suffix = '_{}'.format(FLAGS.infer_name)
      poses_path = os.path.join(
        infer_dir, 'estimated-poses{}.csv'.format(suffix))
      tf.logging.info('Saving estimated poses to: {}'.format(poses_path))
      inout.save_bop_results(poses_path, poses_all, version='bop19')

    time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
    tf.logging.info('Finished inference at: {}'.format(time_str))


if __name__ == '__main__':
  tf.app.run()
