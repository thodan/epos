# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""User-specific configuration (paths etc.)."""

import os


# Folder with TFRecord files.
TF_DATA_PATH = os.environ['TF_DATA_PATH']

# Folder with trained EPOS models (each model is in a subfolder).
TF_MODELS_PATH = os.environ['TF_MODELS_PATH']

# Folder with BOP datasets (https://bop.felk.cvut.cz/datasets/).
BOP_PATH = os.environ['BOP_PATH']
