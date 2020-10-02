# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

r"""Converts a dataset in the BOP format to the TFRecord format.

Example 1: Create a TFRecord with T-LESS PBR training images:
python create_tfrecord.py
  --dataset=tless
  --split=train
  --split_type=pbr
  --examples_filename=tless_train-pbr_examples.txt
  --add_gt=True
  --shuffle=True
  --rgb_format=jpg

Example 2: Create a TFRecord with BOP'19 subset of T-LESS test images:
python create_tfrecord.py
  --dataset=tless
  --split=test
  --split_type=primesense
  --examples_filename=tless_test_targets-bop19_examples.txt
  --add_gt=True
  --shuffle=True
  --rgb_format=jpg

The txt file with examples can be created with script create_example_list.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Do not use GPU in this script.

import io
import time
import random
from functools import partial
import numpy as np
from PIL import Image
import tensorflow as tf
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import transform
from epos_lib import common
from epos_lib import config
from epos_lib import tfrecord


# Flags (other common flags are defined in epos_lib/common.py.
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'split', None,
  'Name of the dataset split.')
flags.DEFINE_string(
  'split_type', None,
  'Type of the dataset split.')
flags.DEFINE_string(
  'examples_filename', None,
  'Path to a file with examples to use (created with create_example_list.py).')
flags.DEFINE_string(
  'output_dir', os.path.join(config.TF_DATA_PATH),
  'Folder where to save the TFRecord file.')
flags.DEFINE_boolean(
  'add_gt', True,
  'Whether to add ground-truth annotations.')
flags.DEFINE_boolean(
  'shuffle', True,
  'Whether to shuffle the examples.')
# Possible RGB formats: 'png', 'jpg', None (None = as in the original images).
flags.DEFINE_string(
  'rgb_format', 'jpg',
  'Format of RGB images.')
# ------------------------------------------------------------------------------


def is_pt_in_im(pt, im_size):
  return 0 <= pt[0] < im_size[0] and 0 <= pt[1] < im_size[1]


def encode_image(im, format):
  with io.BytesIO() as output:
    if format.lower() in ['jpg', 'jpeg']:
      Image.fromarray(im).save(output, format='JPEG', subsampling=0, quality=95)
    else:
      Image.fromarray(im).save(output, format=format.upper())
    im_encoded = output.getvalue()
  return im_encoded


def create_tf_example(
      example, dp_split, scene_camera, scene_gt=None, scene_gt_info=None):

  scene_id = example['scene_id']
  im_id = example['im_id']
  width = dp_split['im_size'][0]
  height = dp_split['im_size'][1]
  K = scene_camera[scene_id][im_id]['cam_K']

  gts = None
  gts_info = None
  mask_visib_fpaths = None
  if FLAGS.add_gt:
    gts = scene_gt[scene_id][im_id]
    gts_info = scene_gt_info[scene_id][im_id]

    # Collect paths to object masks.
    mask_visib_fpaths = []
    for gt_id in range(len(gts)):
      mask_visib_fpaths.append(dp_split['mask_visib_tpath'].format(
        scene_id=scene_id, im_id=im_id, gt_id=gt_id))

  # RGB image.
  im_path = None
  rgb_encoded = None
  if 'rgb' in dp_split['im_modalities']:

    # Absolute path to the RGB image.
    im_path = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)

    # Determine the format of the RGB image.
    rgb_format_in = im_path.split('.')[-1]
    if rgb_format_in in ['jpg', 'jpeg']:
      rgb_format_in = 'jpg'

    # Load the RGB image.
    if rgb_format_in == FLAGS.rgb_format:
      with tf.gfile.GFile(im_path, 'rb') as fid:
        rgb_encoded = fid.read()
    else:
      rgb = inout.load_im(im_path)
      rgb_encoded = encode_image(rgb, FLAGS.rgb_format)

  # Grayscale image.
  elif 'gray' in dp_split['im_modalities']:

    # Absolute path to the grayscale image.
    im_path = dp_split['gray_tpath'].format(scene_id=scene_id, im_id=im_id)

    # Load the grayscale image and duplicate the channel.
    gray = inout.load_im(im_path)
    rgb = np.dstack([gray, gray, gray])
    rgb_encoded = encode_image(rgb, FLAGS.rgb_format)

  # Path of the image relative to BOP_PATH.
  im_path_rel = im_path.split(config.BOP_PATH)[1]
  im_path_rel_encoded = im_path_rel.encode('utf8')

  # Collect ground-truth information about the annotated object instances.
  pose_q1, pose_q2, pose_q3, pose_q4 = [], [], [], []
  pose_t1, pose_t2, pose_t3, t4 = [], [], [], []
  obj_ids = []
  obj_ids_txt = []
  obj_visibilities = []
  masks_visib_encoded = []
  if FLAGS.add_gt:
    for gt_id, gt in enumerate(gts):

      # Orientation of the object instance.
      R = np.eye(4)
      R[:3, :3] = gt['cam_R_m2c']
      q = transform.quaternion_from_matrix(R)
      pose_q1.append(q[0])
      pose_q2.append(q[1])
      pose_q3.append(q[2])
      pose_q4.append(q[3])

      # Translation of the object instance.
      t = gt['cam_t_m2c'].flatten()
      pose_t1.append(t[0])
      pose_t2.append(t[1])
      pose_t3.append(t[2])

      obj_ids_txt.append(str(gt['obj_id']).encode('utf8'))
      obj_ids.append(int(gt['obj_id']))
      obj_visibilities.append(float(gts_info[gt_id]['visib_fract']))

      # Mask of the visible part of the object instance.
      with tf.gfile.GFile(mask_visib_fpaths[gt_id], 'rb') as fid:
        mask_visib_encoded_png = fid.read()
        masks_visib_encoded.append(mask_visib_encoded_png)

  # Intrinsic camera parameters.
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

  # TF Example.
  feature = {
    'image/scene_id': tfrecord.int64_list_feature(scene_id),
    'image/im_id': tfrecord.int64_list_feature(im_id),
    'image/path': tfrecord.bytes_list_feature(im_path_rel_encoded),
    'image/encoded': tfrecord.bytes_list_feature(rgb_encoded),
    'image/width': tfrecord.int64_list_feature(width),
    'image/height': tfrecord.int64_list_feature(height),
    'image/channels': tfrecord.int64_list_feature(3),
    'image/camera/fx': tfrecord.float_list_feature([fx]),
    'image/camera/fy': tfrecord.float_list_feature([fy]),
    'image/camera/cx': tfrecord.float_list_feature([cx]),
    'image/camera/cy': tfrecord.float_list_feature([cy]),
    'image/object/id': tfrecord.int64_list_feature(obj_ids),
    'image/object/visibility': tfrecord.float_list_feature(obj_visibilities),
    'image/object/pose/q1': tfrecord.float_list_feature(pose_q1),
    'image/object/pose/q2': tfrecord.float_list_feature(pose_q2),
    'image/object/pose/q3': tfrecord.float_list_feature(pose_q3),
    'image/object/pose/q4': tfrecord.float_list_feature(pose_q4),
    'image/object/pose/t1': tfrecord.float_list_feature(pose_t1),
    'image/object/pose/t2': tfrecord.float_list_feature(pose_t2),
    'image/object/pose/t3': tfrecord.float_list_feature(pose_t3),
    'image/object/mask': tfrecord.bytes_list_feature(masks_visib_encoded),
  }
  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

  res = tf_example.SerializeToString()
  return res, example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load the list examples.
  examples_path = os.path.join(
    config.TF_DATA_PATH, 'example_lists', FLAGS.examples_filename)
  tf.logging.info('Loading a list of examples from: {}'.format(examples_path))
  examples_list = tfrecord.load_example_list(examples_path)

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    config.BOP_PATH, FLAGS.dataset, FLAGS.split, FLAGS.split_type)

  # Pre-load camera parameters and ground-truth annotations.
  scene_gt = {}
  scene_gt_info = {}
  scene_camera = {}
  scene_ids = set([e['scene_id'] for e in examples_list])
  for scene_id in scene_ids:

    scene_camera[scene_id] = inout.load_scene_camera(
      dp_split['scene_camera_tpath'].format(scene_id=scene_id))

    if FLAGS.add_gt:
      scene_gt[scene_id] = inout.load_scene_gt(
        dp_split['scene_gt_tpath'].format(scene_id=scene_id))
      scene_gt_info[scene_id] = inout.load_json(
        dp_split['scene_gt_info_tpath'].format(scene_id=scene_id),
        keys_to_int=True)

  # Check the name of the file with examples.
  examples_end = '_examples.txt'
  if not FLAGS.examples_filename.endswith(examples_end):
    raise ValueError(
      'Name of the file with examples must end with {}.'.format(examples_end))

  # Prepare writer of the TFRecord file.
  output_name = FLAGS.examples_filename.split(examples_end)[0]
  output_path = os.path.join(FLAGS.output_dir, output_name + '.tfrecord')
  writer = tf.python_io.TFRecordWriter(output_path)
  tf.logging.info('File to be created: {}'.format(output_path))

  # Optionally shuffle the examples.
  if FLAGS.shuffle:
    random.shuffle(examples_list)

  # Write the examples to the TFRecord file.
  w_start_t = time.time()

  create_tf_example_partial = partial(
    create_tf_example,
    dp_split=dp_split,
    scene_camera=scene_camera,
    scene_gt=scene_gt,
    scene_gt_info=scene_gt_info)

  for example_id, example in enumerate(examples_list):
    if example_id % 50 == 0:
      tf.logging.info('Processing example {}/{}'.format(
        example_id + 1, len(examples_list)))

    tf_example, _ = create_tf_example_partial(example)
    writer.write(tf_example)

  # Close the writer.
  writer.close()

  w_total_t = time.time() - w_start_t
  tf.logging.info('Writing took {} s.'.format(w_total_t))


if __name__ == '__main__':
  tf.app.run()
