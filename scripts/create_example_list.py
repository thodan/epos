# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

r"""Creates a list of examples from a BOP dataset.

Each line of the text file contains a scene/chunk ID and an image ID.

Example 1: Create a list of all T-LESS PBR training images:
python create_example_list.py
  --dataset=tless
  --split=train
  --split_type=pbr

Example 2: Create a list of all T-LESS test examples:
python create_example_list.py
  --dataset=tless
  --split=test

Example 3: Create a list of T-LESS test examples used in the BOP Challenge 2019:
python create_example_list.py
  --dataset=tless
  --split=test
  --targets_filename=test_targets_bop19.json
"""

import os
import tensorflow as tf
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from epos_lib import common
from epos_lib import config
from epos_lib import tfrecord


# Flags (other common flags are defined in epos_lib/common.py.
# ------------------------------------------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'split', 'test',
  'Name of the dataset split.')
flags.DEFINE_string(
  'split_type', None,
  'Type of the dataset split.')
flags.DEFINE_list(
  'scene_ids', None,
  'Comma-separated list of scene IDs. If not provided, all scenes in the '
  'specified split are considered.')
flags.DEFINE_string(
  'targets_filename', None,
  'Name of a JSON file with a list of targets, saved in the dataset folder.'
  '(github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).')
flags.DEFINE_string(
  'output_dir', os.path.join(config.TF_DATA_PATH, 'example_lists'),
  'Folder where to save the TFRecord file.')
# ------------------------------------------------------------------------------


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.scene_ids is not None and FLAGS.targets_filename is not None:
    raise ValueError(
      'Only up to one of scene_ids and targets_filename can be specified.')

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    config.BOP_PATH, FLAGS.dataset, FLAGS.split, FLAGS.split_type)

  output_suffix = None

  if FLAGS.targets_filename:
    output_suffix = 'targets'
    test_targets = inout.load_json(
      os.path.join(config.BOP_PATH, FLAGS.dataset, FLAGS.targets_filename))
    example_list = []
    for trg in test_targets:
      example = {'scene_id': trg['scene_id'], 'im_id': trg['im_id']}
      if example not in example_list:
        example_list.append(example)

  else:
    if FLAGS.scene_ids is None:
      FLAGS.scene_ids = dataset_params.get_present_scene_ids(dp_split)
    else:
      FLAGS.scene_ids = list(map(int, FLAGS.scene_ids))
      output_suffix = 'scenes-' + '-'.join(
        map(lambda x: '{:01d}'.format(x), FLAGS.scene_ids))

    tf.logging.info('Collecting examples...')
    example_list = []
    for scene_id in FLAGS.scene_ids:
      scene_gt_fpath = dp_split['scene_gt_tpath'].format(scene_id=scene_id)
      im_ids = inout.load_scene_gt(scene_gt_fpath).keys()
      for im_id in sorted(im_ids):
        example_list.append({'scene_id': scene_id, 'im_id': im_id})

  tf.logging.info('Collected {} examples.'.format(len(example_list)))
  assert(len(example_list) > 0)

  split_name = FLAGS.split
  if FLAGS.split_type is not None:
    split_name += '-' + FLAGS.split_type

  if output_suffix is not None:
    output_suffix = '_' + output_suffix
  else:
    output_suffix = ''

  output_fname = '{}_{}{}_examples.txt'.format(
    FLAGS.dataset, split_name, output_suffix)
  output_fpath = os.path.join(FLAGS.output_dir, output_fname)

  tf.logging.info('Saving the list to: {}'.format(output_fpath))
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  tfrecord.save_example_list(output_fpath, example_list)


if __name__ == '__main__':
  tf.app.run()
