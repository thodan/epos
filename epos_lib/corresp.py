# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

import numpy as np
from epos_lib import common
from epos_lib import misc


def establish_many_to_many(
      obj_confs, frag_confs, frag_coords, gt_obj_ids, model_store, output_scale,
      min_obj_conf, min_frag_rel_conf, project_to_surface, only_annotated_objs):
  """Establishes many-to-many 2D-3D correspondences.

  Args:
    obj_confs: Object confidences of shape [output_h, output_w, num_objs].
    frag_confs: Fragment confidences of shape [output_h, output_w, num_objs,
      num_frags].
    frag_coords: 3D fragment coordinates of shape [output_h, output_w, num_objs,
      num_frags, 3].
    gt_obj_ids: List of object ID's of the ground-truth annotations.
    model_store: Store of object models (of class ObjectModelStore).
    output_scale: Scale of the final output w.r.t. the input (output / input).
    min_obj_conf: Threshold on object confidence (tau_a in the EPOS paper).
    min_frag_rel_conf: Threshold on relative fragment confidence (tau_b in the
      EPOS paper).
    project_to_surface: Whether to project the predicted 3D locations to the
      object model.
    only_annotated_objs: Whether to establish correspondences only for annotated
      object ID's.

  Returns:
    A list of established correspondences.
  """
  # Postprocess fragment predictions.
  # ----------------------------------------------------------------------------
  corresp = {}

  # 0 is reserved for background.
  for obj_id in model_store.dp_model['obj_ids']:

    # Consider predictions only for annotated objects.
    if only_annotated_objs and obj_id not in gt_obj_ids:
      continue

    # Mask of pixels with high enough confidence for the current class.
    obj_conf = obj_confs[:, :, obj_id]
    obj_mask = obj_conf > min_obj_conf

    if np.any(obj_mask):

      # Coordinates (y, x) of shape [num_mask_px, 2].
      obj_mask_yx_coords = np.stack(np.nonzero(obj_mask), axis=0).T

      # Convert to (x, y) image coordinates.
      im_coords = np.flip(obj_mask_yx_coords, axis=1)
      im_coords = misc.convert_px_indices_to_im_coords(
        im_coords, 1.0 / output_scale)

      # Fragment confidences of shape [num_obj_mask_px, num_frags].
      frag_conf_masked = frag_confs[obj_mask][:, obj_id - 1, :]

      # Select all fragments with a high enough confidence.
      frag_conf_max = np.max(frag_conf_masked, axis=1, keepdims=True)
      frag_mask = frag_conf_masked > (frag_conf_max * min_frag_rel_conf)

      # Indices (y, x) of positive frags of shape [num_frag_mask_px, 2].
      frag_inds = np.stack(np.nonzero(frag_mask), axis=0).T

      # Collect 2D-3D correspondences.
      corr_2d = im_coords[frag_inds[:, 0]]
      corr_3d = model_store.frag_centers[obj_id][frag_inds[:, 1]]

      # Add the predicted 3D fragment coordinates.
      frag_scales = np.expand_dims(
        model_store.frag_sizes[obj_id][frag_inds[:, 1]], 1)
      corr_3d_local = frag_coords[obj_mask][:, obj_id - 1, :, :][frag_mask]
      corr_3d_local *= frag_scales
      corr_3d += corr_3d_local

      # The confidence of the correspondence is given by:
      # P(fragment, object) = P(fragment | object) * P(object)
      corr_conf_obj = obj_conf[obj_mask][frag_inds[:, 0]]
      corr_conf_frag = frag_conf_masked[frag_mask]
      corr_conf = corr_conf_obj * corr_conf_frag

      # Project the predicted 3D points to the object model.
      if project_to_surface:
        corr_3d = model_store.project_pts_to_model(corr_3d, obj_id)

      # Save the correspondences.
      corresp[obj_id] = {
        'px_id': frag_inds[:, 0],
        'frag_id': frag_inds[:, 1],
        'coord_2d': corr_2d,
        'coord_3d': corr_3d,
        'conf': corr_conf,
        'conf_obj': corr_conf_obj,
        'conf_frag': corr_conf_frag
      }

  return corresp
