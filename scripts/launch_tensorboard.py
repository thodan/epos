#!/usr/bin/env python

"""A script to launch TensorBoard.

Example:
python launch_tensorboard.py
  --models=tless-bop20-xc65-f64,ycbv-bop20-xc65-f64
  --port=8008
"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
  '--models', help='Comma-separated names of models.')
parser.add_argument(
  '--port', help='HTTP port to run TensorBoard on.', default=8008)
args = parser.parse_args()

model_list = args.models.split(',')
port = int(args.port)

logdir_str = ''
for model in model_list:
  if logdir_str != '':
    logdir_str += ','
  model_path = os.path.join(os.environ['TF_MODELS_PATH'], model)
  logdir_str += '{}:{}'.format(model, model_path)

tensorboard_cmd = [
  'tensorboard',
  '--logdir', logdir_str,
  '--port', str(port)
]

print('Running: {}'.format(' '.join(tensorboard_cmd)))
if subprocess.call(tensorboard_cmd) != 0:
  raise RuntimeError('TensorBoard failed.')
