# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Data preparation functions."""

import os
import numpy as np
import pickle
from functools import partial
import tensorflow as tf
import igl
import bop_renderer
from bop_toolkit_lib import config as config_bop
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from epos_lib import augment
from epos_lib import common
from epos_lib import config as config_epos
from epos_lib import datagen_utils
from epos_lib import fragment
from epos_lib import misc


class ObjectModelStore(object):
  """Stores 3D object models, their fragmentation and other relevant data."""

  def __init__(self,
               dataset_name,
               model_type,
               num_frags,
               models=None,
               models_igl=None,
               frag_centers=None,
               frag_sizes=None,
               prepare_for_projection=False):
    """Initializes the 3D object model store.

    Args:
      dataset_name: Dataset name ('tless', 'ycbv', 'lmo', etc.).
      model_type: Type of object models (see bop_toolkit_lib/dataset_params.py).
      num_frags: Number of surface fragments per object.
      models: Dictionary of models (see load_ply from bop_toolkit_lib/inout.py).
      models_igl: Dictionary of models in the IGL format.
      frag_centers: Fragment centers.
      frag_sizes: Fragment sizes defined as the length of the longest side of
        the 3D bounding box of the fragment.
      prepare_for_projection: Whether to prepare the models for projection to
        the model surface.
    """
    self.dataset_name = dataset_name
    self.model_type = model_type
    self.models = models
    self.models_igl = models_igl
    self.num_frags = num_frags
    self.frag_centers = frag_centers
    self.frag_sizes = frag_sizes
    self.prepare_for_projection = prepare_for_projection
    self.aabb_trees_igl = {}

    # Dataset-specific model parameters.
    self.dp_model = dataset_params.get_model_params(
      config_bop.datasets_path, dataset_name, model_type=model_type)

  @property
  def num_objs(self):
    return len(self.models)

  def load_models(self):
    """Loads 3D object models."""
    tf.logging.info('Loading object models...')

    self.models = {}
    self.models_igl = {}
    for obj_id in self.dp_model['obj_ids']:
      model_fpath = self.dp_model['model_tpath'].format(obj_id=obj_id)
      self.models[obj_id] = inout.load_ply(model_fpath)

      if self.prepare_for_projection:
        # Read the model to the igl format.
        V = igl.eigen.MatrixXd(self.models[obj_id]['pts'])
        F = igl.eigen.MatrixXi(self.models[obj_id]['faces'].astype(np.int32))
        self.models_igl[obj_id] = {'V': V, 'F': F}

    tf.logging.info('Loaded {} object models.'.format(len(self.models)))

  def fragment_models(self):
    """Splits the surface of 3D object models into fragments."""
    tf.logging.info('Fragmenting object models...')

    if self.models is None:
      self.load_models()

    self.frag_centers = {}
    self.frag_sizes = {}
    for obj_id in self.dp_model['obj_ids']:
      tf.logging.info('Fragmenting object {}...'.format(obj_id))

      if self.num_frags == 1:
        # Use the origin (the center of the object) as the fragment center in
        # the case of one fragment.
        num_pts = self.models[obj_id]['pts'].shape[0]
        self.frag_centers[obj_id] = np.array([[0., 0., 0.]])
        pt_frag_ids = np.zeros(num_pts)
      else:
        # Find the fragment centers by the furthest point sampling algorithm.
        assert(len(self.models[obj_id]['pts']) >= self.num_frags)
        self.frag_centers[obj_id], pt_frag_ids =\
          fragment.fragmentation_fps(self.models[obj_id]['pts'], self.num_frags)

      # Calculate fragment sizes defined as the length of the longest side of
      # the 3D bounding box of the fragment.
      self.frag_sizes[obj_id] = []
      for frag_id in range(self.num_frags):
        # Points (i.e. model vertices) belonging to the current fragment.
        frag_pts =\
          self.models[obj_id]['pts'][pt_frag_ids == frag_id]

        # Calculate the 3D bounding box of the fragment and its longest side.
        bb_size = np.max(frag_pts, axis=0) - np.min(frag_pts, axis=0)
        min_frag_size = 5.0  # 5 mm.
        frag_size = max(np.max(bb_size), min_frag_size)
        self.frag_sizes[obj_id].append(frag_size)

      self.frag_sizes[obj_id] = np.array(self.frag_sizes[obj_id])

    tf.logging.info('Object models fragmented.')

  def project_pts_to_model(self, pts, obj_id):
    """Projects 3D points to the model of the specified object.

    Args:
      pts: 3D points to project.
      obj_id: ID of the object model to which the points are projected.

    Returns:
      3D points projected to the model surface.
    """
    # Build AABB tree.
    if obj_id not in self.aabb_trees_igl:
      self.aabb_trees_igl[obj_id] = igl.AABB()
      self.aabb_trees_igl[obj_id].init(
        self.models_igl[obj_id]['V'], self.models_igl[obj_id]['F'])

    # Query points.
    P = igl.eigen.MatrixXd(pts)

    # For each query point, find the closest vertex on the model surface.
    sqrD = igl.eigen.MatrixXd()
    I = igl.eigen.MatrixXi()
    C = igl.eigen.MatrixXd()
    self.aabb_trees_igl[obj_id].squared_distance(
      self.models_igl[obj_id]['V'], self.models_igl[obj_id]['F'], P, sqrD, I, C)

    return np.array(C)


class Dataset(object):
  """Represents input dataset."""

  def __init__(self,
               dataset_name,
               tfrecord_names,
               model_dir,
               model_variant,
               batch_size,
               max_height_before_crop,
               crop_size,
               num_frags,
               min_visib_fract,
               gt_knn_frags,
               output_stride,
               is_training,
               return_gt_orig,
               return_gt_maps,
               should_shuffle,
               should_repeat,
               prepare_for_projection,
               data_augmentations,
               buffer_size=50):
    """Initializes the dataset.

    Args:
      dataset_name: Dataset name.
      tfrecord_names: List of names of tfrecord files to read images from.
      model_dir: Model folder where the model fragmentations are saved.
      model_variant: See feature_extractor.network_map for supported variants.
      batch_size: Batch size.
      max_height_before_crop: Maximum image height before cropping.
      crop_size: Image size (width, height).
      num_frags: Number of surface fragments per object.
      min_visib_fract: Only annotated object instances visible from at least
        min_visib_fract are considered.
      gt_knn_frags: Number of the closest fragments to which a point on the
        model surface is assigned during training.
      output_stride: The ratio of input to encoder output resolution.
      is_training: Boolean, if dataset is for training or not.
      return_gt_orig: Whether to return the original GT annotations.
      return_gt_maps: Whether to return the GT maps used for training.
      should_shuffle: Boolean, if should shuffle the input data.
      should_repeat: Boolean, if should repeat the input data.
      prepare_for_projection: Whether to prepare object models for projecting
        3D points to their surface.
      data_augmentations: Dictionary with image augmentation operations and
        their parameters.
      buffer_size: Size of the buffer.
    """
    if return_gt_orig and batch_size != 1:
      raise ValueError(
        'Only batch_size = 1 (per clone) is supported if the original GT '
        'annotations should be returned.')

    if buffer_size < batch_size:
      raise ValueError('Buffer size must be >= batch size.')

    self.dataset_name = dataset_name
    self.tfrecord_names = tfrecord_names
    self.model_dir = model_dir
    self.model_variant = model_variant
    self.batch_size = batch_size
    self.max_height_before_crop = max_height_before_crop
    self.crop_size = crop_size
    self.num_frags = num_frags
    self.min_visib_fract = min_visib_fract
    self.gt_knn_frags = gt_knn_frags
    self.output_stride = output_stride
    self.is_training = is_training
    self.return_gt_orig = return_gt_orig
    self.return_gt_maps = return_gt_maps
    self.should_shuffle = should_shuffle
    self.should_repeat = should_repeat
    self.prepare_for_projection = prepare_for_projection
    self.data_augmentations = data_augmentations
    self.buffer_size = buffer_size

    # Object segmentation label that is ignored when calculating loss.
    self.ignore_obj_label = 255

    # Type of 3D object models for fragmentation.
    model_type_frag = None  # Original models.
    if self.dataset_name == 'tless':
      # Use the reconstructed models for T-LESS as they include only the outer
      # surface ('cad' and 'eval' include also the inner surface).
      model_type_frag = 'reconst'
    elif self.dataset_name == 'itodd':
      # Use more densely sampled models.
      model_type_frag = 'dense'
    elif self.dataset_name == 'tudl':
      # The 'eval' models have more uniform sampling than the original ones.
      model_type_frag = 'eval'

    # Type of 3D object models for rendering etc.
    model_type = 'eval'  # Decimated models to speed up the rendering.

    # Load/calculate fragmentation of the 3D object models.
    frag_path = os.path.join(self.model_dir, 'fragments.pkl')
    if os.path.exists(frag_path):
      tf.logging.info('Loading fragmentation from: {}'.format(frag_path))

      with open(frag_path, 'rb') as f:
        fragments = pickle.load(f)
        frag_centers = fragments['frag_centers']
        frag_sizes = fragments['frag_sizes']

      # Check if the loaded fragmentation is valid.
      for obj_id in frag_centers.keys():
        if frag_centers[obj_id].shape[0] != self.num_frags\
              or frag_sizes[obj_id].shape[0] != self.num_frags:
          raise ValueError('The loaded fragmentation is not valid.')

    else:
      tf.logging.info(
        'Fragmentation does not exist (expected file: {}).'.format(frag_path))
      tf.logging.info('Calculating fragmentation...')

      model_type_frag_str = model_type_frag
      if model_type_frag_str is None:
        model_type_frag_str = 'original'
      tf.logging.info('Type of models: {}'.format(model_type_frag_str))

      # Load 3D object models for fragmentation.
      model_store_frag = ObjectModelStore(
        dataset_name=self.dataset_name,
        model_type=model_type_frag,
        num_frags=self.num_frags,
        prepare_for_projection=False)

      # Fragment the 3D object models.
      model_store_frag.fragment_models()
      frag_centers = model_store_frag.frag_centers
      frag_sizes = model_store_frag.frag_sizes

      # Save the fragmentation.
      tf.logging.info('Saving fragmentation to: {}'.format(frag_path))
      with open(frag_path, 'wb') as f:
        fragments = {'frag_centers': frag_centers, 'frag_sizes': frag_sizes}
        pickle.dump(fragments, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load 3D object models for rendering.
    self.model_store = ObjectModelStore(
      dataset_name=self.dataset_name,
      model_type=model_type,
      num_frags=self.num_frags,
      frag_centers=frag_centers,
      frag_sizes=frag_sizes,
      prepare_for_projection=self.prepare_for_projection)
    self.model_store.load_models()

    if self.dataset_name == 'lmo':
      # For LM-O, create prediction channels for all 15 LM objects, so the
      # object ID corresponds to the channel ID and no mapping is necessary.
      # TODO: Implement the ID mapping.
      self.num_objs = 15
    elif self.dataset_name == 'hbs':
      # For HBS, create prediction channels for all 33 HB objects, so the
      # object ID corresponds to the channel ID and no mapping is necessary.
      # TODO: Implement the ID mapping.
      self.num_objs = 33
    else:
      self.num_objs = len(self.model_store.dp_model['obj_ids'])

    # Initialize a renderer.
    self.renderer = None
    if return_gt_orig or return_gt_maps:
      tf.logging.info('Initializing renderer for data generation...')
      width, height = self.crop_size
      output_width = int(width / self.output_stride)
      output_height = int(height / self.output_stride)
      self.renderer = bop_renderer.Renderer()
      self.renderer.init(output_width, output_height)
      for obj_id in self.model_store.dp_model['obj_ids']:
        path = self.model_store.dp_model['model_tpath'].format(obj_id=obj_id)
        self.renderer.add_object(obj_id, path)
      tf.logging.info('Renderer initialized.')

    # Initialize generator of ground-truth fields with fragment labels and 3D
    # fragment coordinates.
    self.frag_field_generator = None
    if return_gt_maps:
      self.frag_field_generator = datagen_utils.FragmentFieldGenerator(
        frag_centers=self.model_store.frag_centers,
        frag_sizes=self.model_store.frag_sizes,
        renderer=self.renderer,
        knn_frags=self.gt_knn_frags)

  @staticmethod
  def _decode_image(content, channels):
    """Decodes an image. Currently only supports jpeg and png.

    Args:
      content: Encoded image content.
      channels: Number of image channels.
    Returns:
      Tensor with decoded image.
    """
    return tf.cond(
      tf.image.is_jpeg(content),
      lambda: tf.image.decode_jpeg(content, channels),
      lambda: tf.image.decode_png(content, channels))

  @staticmethod
  def _decode_png_instance_masks(masks_png, width, height):
    """Decodes PNG object instance masks and stacks them into dense tensor.

    Args:
      masks_png: Encoded masks.
      width: Width of the masks.
      height: Height of the masks.
    Returns:
      A 3-D boolean tensor of shape [num_instances, height, width].
    """
    def decode_png_mask(image_buffer):
      image = tf.squeeze(
        tf.image.decode_image(image_buffer, channels=1), axis=2)
      return tf.greater(image, 0)

    if isinstance(masks_png, tf.SparseTensor):
      masks_png = tf.sparse_tensor_to_dense(masks_png, default_value='')
    return tf.cond(
      tf.greater(tf.size(masks_png), 0),
      lambda: tf.map_fn(decode_png_mask, masks_png, dtype=tf.bool),
      lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width])), tf.bool))

  @staticmethod
  def _parse_example_proto(example_proto):
    """Parses the following features from an example."""
    features = {
      'image/scene_id':
        tf.FixedLenFeature((), tf.int64, default_value=-1),
      'image/im_id':
        tf.FixedLenFeature((), tf.int64, default_value=-1),
      'image/path':
        tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
      'image/height':
        tf.FixedLenFeature((), tf.int64, default_value=-1),
      'image/width':
        tf.FixedLenFeature((), tf.int64, default_value=-1),
      'image/channels':
        tf.FixedLenFeature((), tf.int64, default_value=-1),
      'image/camera/fx':
        tf.FixedLenFeature((), tf.float32, default_value=-1.0),
      'image/camera/fy':
        tf.FixedLenFeature((), tf.float32, default_value=-1.0),
      'image/camera/cx':
        tf.FixedLenFeature((), tf.float32, default_value=-1.0),
      'image/camera/cy':
        tf.FixedLenFeature((), tf.float32, default_value=-1.0),

      # Ground-truth annotations of object instances.
      'image/object/id': tf.VarLenFeature(tf.int64),
      'image/object/visibility': tf.VarLenFeature(tf.float32),
      'image/object/pose/q1': tf.VarLenFeature(tf.float32),
      'image/object/pose/q2': tf.VarLenFeature(tf.float32),
      'image/object/pose/q3': tf.VarLenFeature(tf.float32),
      'image/object/pose/q4': tf.VarLenFeature(tf.float32),
      'image/object/pose/t1': tf.VarLenFeature(tf.float32),
      'image/object/pose/t2': tf.VarLenFeature(tf.float32),
      'image/object/pose/t3': tf.VarLenFeature(tf.float32),
      'image/object/mask': tf.VarLenFeature(tf.string),
    }
    return tf.parse_single_example(example_proto, features)

  def _parse_and_preprocess(self, example_proto):
    """Function to parse the example proto.

    Args:
      example_proto: Proto in the format of tf.Example.
    Returns:
      A dictionary with the parsed and processed example.
    """
    feat = self._parse_example_proto(example_proto)

    # Input image.
    im = tf.cast(self._decode_image(
      feat['image/encoded'], channels=3), tf.float32)

    # Size of the input image saved in the TFRecord file.
    im_h_orig = tf.cast(feat['image/height'], tf.int32)
    im_w_orig = tf.cast(feat['image/width'], tf.int32)

    # New image size before cropping (ensuring the maximum required height).
    im_h_new = tf.minimum(self.max_height_before_crop, im_h_orig)
    im_scale = tf.cast(im_h_new, tf.float32) / tf.cast(im_h_orig, tf.float32)
    im_w_new = tf.cast(tf.cast(im_w_orig, tf.float32) * im_scale, tf.int32)

    # Crop size.
    crop_w = self.crop_size[0]
    crop_h = self.crop_size[1]

    # Create a random bounding box for cropping.
    max_offset_h = tf.reshape(im_h_new - crop_h, [])
    max_offset_w = tf.reshape(im_w_new - crop_w, [])
    offset_h = tf.random_uniform([], maxval=max_offset_h + 1, dtype=tf.int32)
    offset_w = tf.random_uniform([], maxval=max_offset_w + 1, dtype=tf.int32)

    # Resize and crop the input image.
    im = misc.resize_image_tf(im, (im_w_new, im_h_new))
    im = misc.crop_image(im, offset_h, offset_w, crop_h, crop_w)
    im.set_shape([crop_h, crop_w, 3])

    # Intrinsic parameters.
    fx = feat['image/camera/fx'] * im_scale
    fy = feat['image/camera/fy'] * im_scale
    cx = feat['image/camera/cx'] * im_scale - tf.cast(offset_w, tf.float32)
    cy = feat['image/camera/cy'] * im_scale - tf.cast(offset_h, tf.float32)
    K = tf.convert_to_tensor(
      [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], name='K')

    sample = {
      common.SCENE_ID: feat['image/scene_id'],
      common.IM_ID: feat['image/im_id'],
      common.IMAGE_PATH: feat['image/path'],
      common.IMAGE: im,
      common.K: K
    }

    # Parameters for GT tensors.
    output_w = int(crop_w / self.output_stride)
    output_h = int(crop_h / self.output_stride)

    if self.output_stride == 1:
      output_K = sample[common.K]
    else:
      output_K = tf.convert_to_tensor(
        [[fx / self.output_stride, 0.0, cx / self.output_stride],
         [0.0, fy / self.output_stride, cy / self.output_stride],
         [0.0, 0.0, 1.0]], name='gt_K')

    gt_obj_ids = None
    gt_obj_quats = None
    gt_obj_trans = None
    gt_obj_masks = None
    if self.return_gt_orig or self.return_gt_maps:

      # Decode object annotations (for evaluation).
      # ------------------------------------------------------------------------
      # Shape: [num_gts, 1]
      gt_obj_ids = tf.sparse_tensor_to_dense(feat['image/object/id'])
      gt_obj_visib_fracts = tf.sparse_tensor_to_dense(
        feat['image/object/visibility'])

      # Shape: [num_gts, 4]
      gt_obj_quats = tf.stack([
        tf.sparse_tensor_to_dense(feat['image/object/pose/q1']),
        tf.sparse_tensor_to_dense(feat['image/object/pose/q2']),
        tf.sparse_tensor_to_dense(feat['image/object/pose/q3']),
        tf.sparse_tensor_to_dense(feat['image/object/pose/q4'])
      ], axis=1)

      # Shape: [num_gts, 3]
      gt_obj_trans = tf.stack([
        tf.sparse_tensor_to_dense(feat['image/object/pose/t1']),
        tf.sparse_tensor_to_dense(feat['image/object/pose/t2']),
        tf.sparse_tensor_to_dense(feat['image/object/pose/t3'])
      ], axis=1)

      # Object instance masks.
      # ------------------------------------------------------------------------
      # Shape: [num_gts, height, width]
      gt_obj_masks = self._decode_png_instance_masks(
        feat['image/object/mask'], im_w_orig, im_h_orig)

      # Resize the masks to the scaled input size.
      gt_obj_masks = tf.cast(tf.expand_dims(gt_obj_masks, axis=3), tf.uint8)
      gt_obj_masks = tf.image.resize_nearest_neighbor(
        images=gt_obj_masks,
        size=[im_h_new, im_w_new],
        align_corners=True)

      # Crop the masks.
      crop_image_partial = partial(
        misc.crop_image, offset_height=offset_h,
        offset_width=offset_w, crop_height=crop_h, crop_width=crop_w)
      gt_obj_masks = tf.map_fn(crop_image_partial, gt_obj_masks, tf.uint8)

      # Resize the masks to the output size.
      gt_obj_masks = tf.image.resize_nearest_neighbor(
        images=gt_obj_masks,
        size=[output_h, output_w],
        align_corners=True)
      gt_obj_masks = tf.squeeze(tf.cast(gt_obj_masks, tf.bool), axis=3)

      # Keep only object ID's present in the current datasets (e.g. not all LM
      # objects are included in the LM-O dataset).
      # ------------------------------------------------------------------------
      kept_gt_ids = datagen_utils.filter_gt_ids_tf(
          gt_obj_ids, self.model_store.dp_model['obj_ids'])

      gt_obj_ids = tf.gather(gt_obj_ids, kept_gt_ids)
      gt_obj_visib_fracts = tf.gather(gt_obj_visib_fracts, kept_gt_ids)
      gt_obj_quats = tf.gather(gt_obj_quats, kept_gt_ids)
      gt_obj_trans = tf.gather(gt_obj_trans, kept_gt_ids)
      gt_obj_masks = tf.gather(gt_obj_masks, kept_gt_ids)

      # Keep only annotated object instances which are sufficiently visible.
      # Note: The visibility is calculated in the original image, not in the
      # cropped one.
      # ------------------------------------------------------------------------
      if self.min_visib_fract is not None:
        kept_gt_ids = datagen_utils.filter_visib_tf(
            gt_obj_visib_fracts, self.min_visib_fract)

        gt_obj_ids = tf.gather(gt_obj_ids, kept_gt_ids)
        gt_obj_visib_fracts = tf.gather(gt_obj_visib_fracts, kept_gt_ids)
        gt_obj_quats = tf.gather(gt_obj_quats, kept_gt_ids)
        gt_obj_trans = tf.gather(gt_obj_trans, kept_gt_ids)
        gt_obj_masks = tf.gather(gt_obj_masks, kept_gt_ids)

      # Make sure the object masks are exclusive (this is assumed when
      # constructing the ground-truth fields).
      # ------------------------------------------------------------------------
      gt_obj_masks = datagen_utils.make_masks_exclusive_tf(
        gt_obj_masks=gt_obj_masks,
        gt_obj_ids=gt_obj_ids,
        gt_obj_quats=gt_obj_quats,
        gt_obj_trans=gt_obj_trans,
        K=output_K,
        renderer=self.renderer)
      gt_obj_masks.set_shape([None, output_h, output_w])

      # Tensors with the original GT annotations.
      # ------------------------------------------------------------------------
      if self.return_gt_orig:
        sample[common.GT_OBJ_IDS] = gt_obj_ids
        sample[common.GT_OBJ_VISIB_FRACT] = gt_obj_visib_fracts
        sample[common.GT_OBJ_MASKS] = gt_obj_masks
        sample[common.GT_OBJ_QUATS] = gt_obj_quats
        sample[common.GT_OBJ_TRANS] = gt_obj_trans

    if self.return_gt_maps:

      # Object label map.
      # ------------------------------------------------------------------------
      num_gts = tf.size(gt_obj_ids)

      # Get the segmentation mask by merging the object instance masks.
      # It is assumed that the object instance masks are exclusive (see above).
      gt_obj_ids_map = tf.tile(
        tf.reshape(gt_obj_ids, [num_gts, 1, 1]),
        [1, output_h, output_w])
      sample[common.GT_OBJ_LABEL] = tf.reduce_sum(tf.multiply(
        tf.cast(gt_obj_masks, tf.int32),
        tf.cast(gt_obj_ids_map, tf.int32)), axis=0)

      # Set "ignore" label to the black bg in the real training T-LESS images.
      # ------------------------------------------------------------------------
      if self.dataset_name == 'tless':
        active_path_patterns = ['tless/train_primesense']
        sample[common.GT_OBJ_LABEL] = tf.py_func(
          datagen_utils.set_background_to_ignore_label_py,
          [sample[common.IMAGE_PATH], active_path_patterns,
           sample[common.GT_OBJ_LABEL], self.ignore_obj_label], tf.int32)
        sample[common.GT_OBJ_LABEL].set_shape([output_h, output_w])

      # Fragment label map.
      # ------------------------------------------------------------------------
      (sample[common.GT_FRAG_LABEL], sample[common.GT_FRAG_LOC],
       sample[common.GT_FRAG_WEIGHT]) = \
          self.frag_field_generator.construct_frag_fields_tf(
              width=output_w,
              height=output_h,
              K=output_K,
              gt_obj_ids=gt_obj_ids,
              gt_obj_quats=gt_obj_quats,
              gt_obj_trans=gt_obj_trans,
              gt_obj_masks=gt_obj_masks)

    # Data augmentation.
    # --------------------------------------------------------------------------
    if self.is_training and self.data_augmentations is not None:

      # Scale the RGB image from [0, 255] to [0, 1].
      sample[common.IMAGE] /= 255.0

      for aug_name, aug_params in self.data_augmentations.items():

        if aug_name == 'random_adjust_brightness':
          sample[common.IMAGE] = augment.random_adjust_brightness(
            sample[common.IMAGE], aug_params['min_delta'],
            aug_params['max_delta'])

        elif aug_name == 'random_adjust_contrast':
          sample[common.IMAGE] = augment.random_adjust_contrast(
            sample[common.IMAGE], aug_params['min_delta'],
            aug_params['max_delta'])

        elif aug_name == 'random_adjust_saturation':
          sample[common.IMAGE] = augment.random_adjust_saturation(
            sample[common.IMAGE], aug_params['min_delta'],
            aug_params['max_delta'])

        elif aug_name == 'random_adjust_hue':
          sample[common.IMAGE] = augment.random_adjust_hue(
            sample[common.IMAGE], aug_params['max_delta'])

        elif aug_name == 'random_blur':
          sample[common.IMAGE] = augment.random_blur(
            sample[common.IMAGE], aug_params['max_sigma'])

        elif aug_name == 'random_gaussian_noise':
          sample[common.IMAGE] = augment.random_gaussian_noise(
            sample[common.IMAGE], aug_params['max_sigma'])

        elif aug_name == 'jpeg_artifacts':
          sample[common.IMAGE] = augment.jpeg_artifacts(
            sample[common.IMAGE], aug_params['min_quality'])

      # Scale the RGB image back from [0, 1] to [0, 255].
      sample[common.IMAGE] *= 255.0

    return sample

  def get_one_shot_iterator(self):
    """Gets an iterator that iterates across the dataset once.

    Returns:
      An iterator of type tf.data.Iterator.
    """
    # Keep num_readers to 1 or None (i.e. sequential reading), because the
    # renderer is maybe not thread safe.
    # TODO: Parallelize dataset reading.
    num_readers = None

    files = self._get_all_tfrecords()
    dataset = tf.data.Dataset.from_tensor_slices(files)

    dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(
        x, num_parallel_reads=num_readers).map(
        self._parse_and_preprocess, num_parallel_calls=num_readers),
      cycle_length=len(files), block_length=1, num_parallel_calls=num_readers)

    if self.should_shuffle:
      dataset = dataset.shuffle(buffer_size=self.buffer_size)

    if self.should_repeat:
      # Repeat forever for training.
      dataset = dataset.repeat()
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(self.batch_size)
    return dataset.make_one_shot_iterator()

  def _get_all_tfrecords(self):
    """Gets all tfrecord files to read data from.

    Returns:
      A list of input files.
    """
    input_files = []
    for tfrecord_name in self.tfrecord_names:
      file_pattern = os.path.join(
        config_epos.TF_DATA_PATH, '{}.tfrecord'.format(tfrecord_name))
      input_files += tf.gfile.Glob(file_pattern)

    tf.logging.info('Input files: {}'.format(input_files))
    if len(input_files) == 0:
      raise ValueError('No input files.')

    return input_files
