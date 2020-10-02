# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

import os
import numpy as np
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc
from bop_toolkit_lib import inout
from epos_lib import config
from epos_lib import fragment


def fragmentation_fps_test():

  output_dir = 'fragmentation_test_output'
  misc.ensure_dir(output_dir)

  datasets = ['hb', 'ycbv', 'tless', 'lmo', 'icbin', 'itodd', 'tudl']

  for dataset in datasets:

    model_type = None
    if dataset == 'tless':
      model_type = 'reconst'
    elif dataset == 'itodd':
      model_type = 'dense'

    dp_model = dataset_params.get_model_params(
      config.BOP_PATH, dataset, model_type)

    for obj_id in dp_model['obj_ids']:
      print('Fragmenting object {} from dataset {}...'.format(obj_id, dataset))

      model_fpath = dp_model['model_tpath'].format(obj_id=obj_id)
      model = inout.load_ply(model_fpath)

      # Fragmentation by the furthest point sampling.
      frag_centers, vertex_frag_ids = \
        fragment.fragmentation_fps(model['pts'], num_frags=256)

      # Fragment colors.
      frag_colors = frag_centers - frag_centers.min()
      frag_colors = (255.0 * frag_colors / frag_colors.max()).astype(np.uint8)

      # Color the model points by the fragment colors.
      pts_colors = np.zeros((model['pts'].shape[0], 3), np.uint8)
      for frag_id in range(len(frag_centers)):
        pts_colors[vertex_frag_ids == frag_id] = frag_colors[frag_id]

      inout.save_ply(os.path.join(
          output_dir, '{}_obj_{:02d}_fragments.ply'.format(dataset, obj_id)),
        {'pts': model['pts'], 'faces': model['faces'], 'colors': pts_colors})


if __name__ == '__main__':
  fragmentation_fps_test()
