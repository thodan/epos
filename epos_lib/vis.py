# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Visualization functions."""

import os
import math
import numpy as np
from PIL import Image
from PIL import ImageDraw
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc as misc_bop
from bop_toolkit_lib import visualization
from epos_lib import misc


# A label colormap used in ADE20K segmentation benchmark.
colormap_ade20k = np.asarray([
  [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
  [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
  [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
  [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4],
  [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51],
  [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
  [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
  [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255],
  [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41],
  [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
  [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
  [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255],
  [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245],
  [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
  [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255],
  [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153],
  [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0],
  [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
  [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255],
  [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0],
  [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0],
  [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
  [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0],
  [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255],
  [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143],
  [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122],
  [255, 245, 0], [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
  [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255],
  [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255],
  [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0],
  [25, 194, 194], [102, 255, 0], [92, 0, 255]
])


def build_grid(tiles, tile_size, grid_rows=None, grid_cols=None):
  """Creates a grid image from a list of tiles.

  Args:
    tiles: List of tiles.
    tile_size: Size of each tile (height, width).
    grid_rows: Number of grid rows.
    grid_cols: Number of grid columns.
  Return:
    Grid image.
  """
  if grid_rows is None or grid_cols is None:
    grid_rows = int(math.sqrt(len(tiles)))
    grid_cols = int(math.ceil(len(tiles) / grid_rows))

  grid = np.zeros(
    (grid_rows * tile_size[1], grid_cols * tile_size[0], 3), np.uint8)
  for tile_id, tile in enumerate(tiles):
    assert(tile.shape[0] == tile_size[1] and tile.shape[1] == tile_size[0])
    yy = int(tile_id / grid_cols)
    xx = tile_id % grid_cols
    grid[(yy * tile_size[1]):((yy + 1) * tile_size[1]),
    (xx * tile_size[0]):((xx + 1) * tile_size[0]), :] = tile
  return grid


def colorize_label_map(label):
  """Colorizes a label map.

  Args:
    label: A 2D array with integer type, storing the segmentation label.
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the color map.
  Raises:
    ValueError: If label is not of rank 2.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

  colormap = colormap_ade20k
  label_mod = np.mod(label, len(colormap))
  return colormap[label_mod].astype(np.uint8)


def colorize_xyz(xyz):
  """Colorizes 3D points by putting them into an RGB box.

  Args:
    xyz: [num_pts, 3] ndarray with 3D points.
  Return:
    [num_pts, 3] ndarray with point colors.
  """
  xyz_vis = xyz - xyz.min()
  return (255 * xyz_vis / xyz_vis.max()).astype(np.uint8)


def visualize_coordinate_frame(im, K, R, t, vis_size_in_px=15):
  """Draws a coordinate frame on top of an image.

  Args:
    im: Input image.
    K: 3x3 intrinsic matrix.
    R: 3x3 rotation matrix.
    t: 3x1 translation vector.
    vis_size_in_px: Length of each axis in pixels.
  Return:
    The input image augmented with the coordinate frame.
  """
  f = 0.5 * (K[0, 0] + K[1, 1])
  depth = 500.  # [mm]
  a = depth * vis_size_in_px / f
  pts_3d = np.array([[0., 0., 0.], [a, 0., 0.], [0., a, 0.], [0., 0., a]])
  pts_im = misc_bop.project_pts(pts_3d, K, R, t)

  im_pil = Image.fromarray(im)
  draw = ImageDraw.Draw(im_pil)
  for i in range(1, 4):
    color = [0, 0, 0]
    color[i - 1] = 255
    pts = tuple(
      map(int, [pts_im[0, 0], pts_im[0, 1], pts_im[i, 0], pts_im[i, 1]]))
    draw.line(pts, fill=tuple(color), width=2)
  del draw
  return np.asarray(im_pil)


def visualize_object_poses(rgb, K, poses, renderer):
  """Visualizes object poses on top of an image.

  Args:
    rgb: Input image.
    K: 3x3 intrinsic matrix.
    poses: List of poses, each given by a dictionary with these items:
      obj_id: Object ID.
      R: 3x3 rotation matrix.
      t: 3x1 translation vector.
    renderer: Renderer of class bop_renderer.Renderer().
  Return:
    The input image augmented with the visualization of the object poses.
  """
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  ren_rgb = np.zeros(rgb.shape, np.uint8)

  # Render the pose estimates one by one.
  for pose in poses:

    # Rendering.
    R_list = pose['R'].flatten().tolist()
    t_list = pose['t'].flatten().tolist()
    renderer.render_object(pose['obj_id'], R_list, t_list, fx, fy, cx, cy)
    m_rgb = renderer.get_color_image(pose['obj_id']).astype(np.float32)

    # Combine the RGB renderings.
    ren_rgb_f = ren_rgb.astype(np.float32) + m_rgb.astype(np.float32)
    ren_rgb_f[ren_rgb_f > 255] = 255
    ren_rgb = ren_rgb_f.astype(np.uint8)

  # Blend the RGB visualization.
  vis_im_rgb = 0.3 * rgb.astype(np.float32) + 0.7 * ren_rgb.astype(np.float32)
  vis_im_rgb[vis_im_rgb > 255] = 255

  return vis_im_rgb.astype(np.uint8)


def visualize_gt_frag(
      gt_obj_ids, gt_obj_masks, gt_frag_labels, gt_frag_weights, gt_frag_coords,
      output_size, model_store, vis_prefix, vis_dir):
  """Visualizes GT fragment fields.

  Args:
    gt_obj_ids: GT object ID's.
    gt_obj_masks: GT object instance masks.
    gt_frag_labels: GT fragment labels.
    gt_frag_weights: GT fragment weights.
    gt_frag_coords: GT fragment coordinates.
    output_size: Size of the output fields.
    model_store: Store of 3D object models.
    vis_dir: Where to save the visualizations.
    vis_prefix: Name prefix of the visualizations.
  """
  # Consider the first (i.e. the closest) fragment.
  frag_ind = 0

  centers_vis = np.zeros((output_size[1], output_size[0], 3))
  for gt_id, obj_id in enumerate(gt_obj_ids):
    obj_mask = gt_obj_masks[gt_id]
    obj_frag_labels = gt_frag_labels[obj_mask][:, frag_ind]
    centers_vis[obj_mask] = model_store.frag_centers[obj_id][obj_frag_labels]

  weights_vis = gt_frag_weights[:, :, frag_ind]
  weights_vis /= weights_vis.max()

  coords_vis = np.zeros((output_size[1], output_size[0], 3))
  for gt_id, obj_id in enumerate(gt_obj_ids):

    obj_mask = gt_obj_masks[gt_id]
    obj_frag_labels = gt_frag_labels[obj_mask][:, frag_ind]
    obj_frag_coords = gt_frag_coords[obj_mask][:, frag_ind, :]

    # Scale by fragment sizes.
    frag_scales = model_store.frag_sizes[obj_id][obj_frag_labels]
    obj_frag_coords *= np.expand_dims(frag_scales, 1)

    coords_vis[obj_mask] = obj_frag_coords

  # Reconstruct the XYZ object coordinates.
  xyz_vis = centers_vis + coords_vis

  # Normalize the visualizations.
  centers_vis = centers_vis - centers_vis.min()
  centers_vis /= centers_vis.max()

  coords_vis = coords_vis - coords_vis.min()
  coords_vis /= coords_vis.max()

  xyz_vis = xyz_vis - xyz_vis.min()
  xyz_vis /= xyz_vis.max()

  # Save the visualizations.
  inout.save_im(
    os.path.join(vis_dir, '{}_gt_frag_labels.png'.format(vis_prefix)),
    (255.0 * centers_vis).astype(np.uint8))

  inout.save_im(
    os.path.join(vis_dir, '{}_gt_frag_coords.png'.format(vis_prefix)),
    (255.0 * coords_vis).astype(np.uint8))

  inout.save_im(
    os.path.join(vis_dir, '{}_gt_frag_reconst.png'.format(vis_prefix)),
    (255.0 * xyz_vis).astype(np.uint8))

  inout.save_im(
    os.path.join(vis_dir, '{}_gt_frag_weights.png'.format(vis_prefix)),
    (255.0 * weights_vis).astype(np.uint8))


def visualize_pred_frag(
      frag_confs, frag_coords, output_size, model_store, vis_prefix, vis_dir,
      vis_ext='png'):
  """Visualizes predicted fragment fields.

  Args:
    frag_confs: Predicted fragment confidences of shape [output_h, output_w,
      num_objs, num_frags].
    frag_coords: Predicted 3D fragment coordinates of shape [field_h, field_w,
      num_fg_cls, num_bins, 3].
    output_size: Size of the fragment fields.
    model_store: Store of 3D object models.
    vis_prefix: Name prefix of the visualizations.
    vis_dir: Where to save the visualizations.
    vis_ext: Extension of the visualizations ('jpg', 'png', etc.).
  """
  num_objs = frag_confs.shape[2]
  tiles_centers = []
  tiles_coords = []
  tiles_reconst = []
  for obj_id in range(1, num_objs + 1):

    # Fragment confidences of shape [field_h, field_w, num_frags].
    conf_obj = frag_confs[:, :, obj_id - 1, :]
    field_shape = (conf_obj.shape[0], conf_obj.shape[1], 3)

    # Indices of fragments with the highest confidence.
    top_inds = np.argmax(conf_obj, axis=2)
    top_inds_f = top_inds.flatten()

    # Fragment centers.
    top_centers = np.reshape(
      model_store.frag_centers[obj_id][top_inds_f], field_shape)

    # Fragment coordinates of shape [field_h * field_w, num_frags, 3].
    num_frags = frag_coords.shape[3]
    coords_obj = frag_coords[:, :, obj_id - 1, :, :].reshape((-1, num_frags, 3))

    # Top fragment coordinates of shape [field_h * field_w, 3].
    top_coords_rel = coords_obj[np.arange(top_inds.size), top_inds_f]
    top_scales = model_store.frag_sizes[obj_id][top_inds_f]
    top_coords = top_coords_rel * top_scales.reshape((-1, 1))

    # Reshape to [field_h, field_w, 3].
    top_coords = top_coords.reshape(field_shape)

    # Reconstruction of shape [field_h * field_w, 3].
    top_reconst = top_centers + top_coords

    txt_list = [{'name': 'cls', 'val': obj_id, 'fmt': ':d'}]
    tiles_centers.append(visualization.write_text_on_image(
      colorize_xyz(top_centers), txt_list, size=10, color=(1.0, 1.0, 1.0)))
    tiles_coords.append(visualization.write_text_on_image(
      colorize_xyz(top_coords), txt_list, size=10, color=(1.0, 1.0, 1.0)))
    tiles_reconst.append(visualization.write_text_on_image(
      colorize_xyz(top_reconst), txt_list, size=10, color=(1.0, 1.0, 1.0)))

  # Assemble and save the visualization grids.
  fname = '{}_pred_frag_centers.{}'.format(vis_prefix, vis_ext)
  grid = build_grid(tiles_centers, output_size)
  inout.save_im(os.path.join(vis_dir, fname), grid)

  fname = '{}_pred_frag_coords.{}'.format(vis_prefix, vis_ext)
  grid = build_grid(tiles_coords, output_size)
  inout.save_im(os.path.join(vis_dir, fname), grid)

  fname = '{}_pred_frag_reconst.{}'.format(vis_prefix, vis_ext)
  grid = build_grid(tiles_reconst, output_size)
  inout.save_im(os.path.join(vis_dir, fname), grid)
