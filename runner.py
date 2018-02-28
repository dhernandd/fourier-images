# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
# import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from code.fourier import Fourier

DTYPE = tf.float32

# DIRECTORIES, SAVE FILES, ETC
LOCAL_DATA_ROOT = "/Users/danielhernandez/work/fourier-images/data/mnist/"

# MODEL ATTRIBUTES
YDIM = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CHANNELS = 1
BATCH_SIZE = 5

# OPTIMIZATION
LEARNING_RATE = 2e-2
BETA1 = 0.5

flags = tf.app.flags

flags.DEFINE_string('local_data_root', LOCAL_DATA_ROOT, "The directory that stores all datasets")

flags.DEFINE_integer('yDim', YDIM, "The number of categories")
flags.DEFINE_integer('img_height', IMG_HEIGHT, "Image height")
flags.DEFINE_integer('img_width', IMG_WIDTH, "Image width")
flags.DEFINE_integer('num_channels', NUM_CHANNELS, "Number of channels")
flags.DEFINE_integer('batch_size', BATCH_SIZE, "Batch size")

flags.DEFINE_float('learning_rate', LEARNING_RATE, "The learning rate")
flags.DEFINE_float('beta1', BETA1, "The beta1 parameter of the Adam optimization")

params = tf.flags.FLAGS

def main(_):
    """
    """
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            fou = Fourier(params) # Tu es fou, toi?

            sess.run(tf.global_variables_initializer())
            fou.train(sess)
   
    
if __name__ == '__main__':
    tf.app.run()