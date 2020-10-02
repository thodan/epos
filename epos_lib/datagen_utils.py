# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

import numpy as np
import tensorflow as tf
from scipy import spatial
from bop_toolkit_lib import transform


def filter_gt_ids_tf(gt_obj_ids, keep_obj_ids):
  """Returns ID's of GT annotations associated with the specified object ID's.

  Args:
    gt_obj_ids: Object ID's of the ground-truth annotations.
    keep_obj_ids: Object ID's for which annotations are kept.
  Returns:
    ID's of kept ground-truth annotations.
  """
  def filter_gt_ids_py(gt_obj_ids, keep_obj_ids):
    kept_gt_ids = []
    for gt_id, obj_id in enumerate(gt_obj_ids):
      if obj_id in keep_obj_ids:
        kept_gt_ids.append(gt_id)
    return np.array(kept_gt_ids, np.int64)

  return tf.py_func(filter_gt_ids_py, [gt_obj_ids, keep_obj_ids], tf.int64)


def filter_visib_tf(gt_obj_visib_fracts, min_visib_fract):
  """Returns ID's of GT annotations which are visible enough.

  Args:
    gt_obj_visib_fracts: Visible fractions of the ground-truth annotations.
    min_visib: Minimum visible fraction to keep an annotation.
  Returns:
    ID's of kept ground-truth annotations.
  """
  def filter_visib_py(gt_obj_visib_fracts, min_visib_fract):
    kept_gt_ids = []
    for gt_id, visib_fract in enumerate(gt_obj_visib_fracts):
      if visib_fract >= min_visib_fract:
        kept_gt_ids.append(gt_id)
    return np.array(kept_gt_ids, np.int64)

  return tf.py_func(
    filter_visib_py, [gt_obj_visib_fracts, min_visib_fract], tf.int64)


def make_masks_exclusive_tf(
      gt_obj_masks, gt_obj_ids, gt_obj_quats, gt_obj_trans, K, renderer):
  """Makes a set of object instance masks mutually exclusive.

    Args:
      gt_obj_masks: Instance masks of the same size.
      gt_obj_ids: Object ID's.
      gt_obj_quats: Quaternions representing orientation of the objects.
      gt_obj_trans: Translation vectors of the objects.
      K: Camera intrinsic matrix.
      renderer: Renderer of class bop_renderer.Renderer().
    Returns:
      Mutually exlusive instance masks of the same shape as gt_obj_masks.
      If more masks overlap at a pixel, the pixel will be kept in the latest
      mask (i.e. the one with the highest index in gt_obj_masks).
  """
  def make_masks_exclusive_py(
        K, gt_obj_ids, gt_obj_quats, gt_obj_translations, gt_obj_masks):

    num_gts, height, width = gt_obj_masks.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Mask of still available pixels (not included in any instance mask yet).
    avail_mask = np.ones([height, width], np.bool)

    # Reverse order to be consistent with datagen_utils.construct_seg_mask_py.
    for gt_id in range(num_gts)[::-1]:
      obj_id = gt_obj_ids[gt_id]

      # Object pose.
      q = gt_obj_quats[gt_id]
      R_list = transform.quaternion_matrix(q)[:3, :3].flatten().tolist()
      t = gt_obj_translations[gt_id]
      t_list = t.flatten().tolist()

      # Render the model to make sure the mask is aligned with the rendering.
      renderer.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
      depth = renderer.get_depth_image(obj_id).astype(np.float32)

      obj_mask = np.logical_and(
        np.logical_and(gt_obj_masks[gt_id], avail_mask), depth > 0)

      avail_mask = np.logical_and(np.logical_not(obj_mask), avail_mask)
      gt_obj_masks[gt_id] = obj_mask

    return gt_obj_masks

  return tf.py_func(
    make_masks_exclusive_py,
    [K, gt_obj_ids, gt_obj_quats, gt_obj_trans, gt_obj_masks], tf.bool)


def construct_seg_mask_py(masks, gts, height, width):
  """Construct a segmentation mask from a list of object masks.

  Args:
    masks: Masks of object instances.
    gts: Ground-truth annotations.
    height: Height of the masks.
    width: Width of the masks.
  Returns:
    Segmantation mask.
  """
  seg_mask = np.zeros((height, width), np.uint16)
  for gt_id, mask_visib in enumerate(masks):
    seg_mask[mask_visib.astype(np.bool)] = gts[gt_id]['obj_id']
  return seg_mask


def set_background_to_ignore_label_py(
      im_path, active_path_patterns, label_map, ignore_label):
  """Replaces the background label (0) by the ignore label.

  Args:
    im_path: Path to the image.
    active_path_patterns: List of path patterns that activate this function.
    label_map: 2D label map.
    ignore_label: Label which is not considered when calculating the loss.
  Returns:
    Potentially modified label map.
  """
  active = False
  for active_path_pattern in active_path_patterns:
    if active_path_pattern.decode('utf8') in im_path.decode('utf8'):
      active = True
      break

  if active:
    label_map[label_map == 0] = ignore_label
    return label_map.astype(np.int32)
  else:
    return label_map


class FragmentFieldGenerator(object):
  """Generates GT fields with fragment labels and 3D fragment coordinates."""

  def __init__(self, frag_centers, frag_sizes, renderer, knn_frags=1):
    """Initializes the fragment field generator.

    Args:
      frag_centers: Map from obj ID to [num_bins, 3] ndarray with frag. centers.
      frag_sizes: A map from obj ID to [num_bins] ndarray with fragment sizes.
      knn_frags: Number of the nearest fragments to assign to each point on the
        model surface.
      renderer: Renderer of class bop_renderer.Renderer().
    """
    self.frag_centers = frag_centers
    self.frag_sizes = frag_sizes
    self.renderer = renderer
    self.knn_frags = knn_frags

  def assign_to_frags_py(self, obj_id, xyz):
    """Assigns XYZ points to fragments.

    Args:
      obj_id: Object ID.
      xyz: [n, 3] ndarray with 3D points.
    Returns:
      Assigned fragment labels, 3D fragment coordinates and fragment weights.
    """
    num_pts = xyz.shape[0]

    # Find the closest fragment centers.
    # Note: nn_dists and nn_ids are sorted by the distance in ascending order.
    nn_index = spatial.cKDTree(self.frag_centers[obj_id])
    nn_dists, nn_ids = nn_index.query(xyz, k=self.knn_frags)
    nn_weights = np.ones((num_pts, self.knn_frags), np.float32)

    # Duplicate each point knn_frags times to get shape [n * knn_frags, 3].
    xyz_dup = np.tile(xyz, [1, self.knn_frags]).reshape((-1, 3))

    # Centers of the nearest fragments with shape [n * knn_frags, 3].
    nn_centers = self.frag_centers[obj_id][nn_ids.flatten()]

    # Deltas w.r.t. the nearest fragment centers with shape [n, knn_frags, 3].
    nn_coords = xyz_dup - nn_centers

    # Normalize the coordinates.
    nn_scales = self.frag_sizes[obj_id][nn_ids.flatten()]
    nn_coords = np.divide(nn_coords, np.expand_dims(nn_scales, 1))

    # Reshape and cast the outputs.
    nn_ids = nn_ids.astype(np.int32).reshape(
      (num_pts, self.knn_frags))
    nn_coords = nn_coords.astype(np.float32).reshape(
      (num_pts, self.knn_frags, 3))
    nn_weights = nn_weights.astype(np.float32).reshape(
      (num_pts, self.knn_frags))

    return nn_ids, nn_coords, nn_weights

  def construct_frag_fields_py(
        self, width, height, K, gt_obj_ids, gt_obj_quats, gt_obj_trans,
        gt_obj_masks):
    """Constructs GT fragment fields.

    Args:
      See construct_frag_fields_tf.
    Returns:
      See construct_frag_fields_tf.
    """
    frag_ids = np.zeros((height, width, self.knn_frags), np.int32)
    frag_coords = np.zeros((height, width, self.knn_frags, 3), np.float32)
    frag_weights = np.zeros((height, width, self.knn_frags), np.float32)

    for gt_id, obj_id in enumerate(gt_obj_ids):
      # Quaternion, translation, and object mask.
      obj_q = gt_obj_quats[gt_id]
      obj_t = gt_obj_trans[gt_id]
      obj_mask = gt_obj_masks[gt_id].astype(np.bool)

      # Render XYZ object coordinates.
      fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
      R_list = transform.quaternion_matrix(obj_q)[:3, :3].flatten().tolist()
      t_list = obj_t.flatten().tolist()
      self.renderer.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
      xyz = self.renderer.get_local_pos_image(obj_id).astype(np.float32)

      # Assign the XYZ points to fragments (the masks are assumed exclusive).
      frag_ids[obj_mask], frag_coords[obj_mask], frag_weights[obj_mask] =\
        self.assign_to_frags_py(obj_id, xyz[obj_mask])

    return frag_ids, frag_coords, frag_weights

  def construct_frag_fields_tf(
        self, width, height, K, gt_obj_ids, gt_obj_quats, gt_obj_trans,
        gt_obj_masks):
    """Constructs GT fragment fields.

    Args:
      width: Width of the fragment fields.
      height: Height of the fragment fields.
      K: Intrinsic camera matrix.
      gt_obj_ids: Object ID's.
      gt_obj_quats: Quaternions.
      gt_obj_trans: 3D translation vectors.
      gt_obj_masks: Masks of the object instances.
    Returns:
      Fields with fragment labels, 3D fragment coordinates and fragment weights.
    """
    frag_ids, frag_coords, frag_weights = tf.py_func(
      self.construct_frag_fields_py,
      [width, height, K, gt_obj_ids, gt_obj_quats, gt_obj_trans, gt_obj_masks],
      [tf.int32, tf.float32, tf.float32])

    # Set shape (it is unknown from tf.py_func).
    frag_ids = tf.reshape(frag_ids, [height, width, self.knn_frags])
    frag_coords = tf.reshape(frag_coords, [height, width, self.knn_frags, 3])
    frag_weights = tf.reshape(frag_weights, [height, width, self.knn_frags])

    return frag_ids, frag_coords, frag_weights
