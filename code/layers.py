# Copyright 2017 Daniel Hernandez Diaz, Columbia University
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
import tensorflow as tf

if __name__ == 'layers':
    from utils import variable_in_cpu  # @UnresolvedImport @UnusedImport
else:
    from .utils import variable_in_cpu  # @Reimport

DTYPE = tf.float32

class FullLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'sigmoid' : tf.nn.sigmoid}
        
    def __call__(self, Input, nodes, nl='softplus', scope=None,
                 initializer=tf.orthogonal_initializer(),
                 b_initializer=tf.zeros_initializer()):
        """
        """
        nonlinearity = self.nl_dict[nl]
        input_dim = Input.get_shape()[-1]
        
        if self.collections is not None:
            self.collections += [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
            
        with tf.variable_scope(scope or 'fullL'):
            weights = variable_in_cpu('weights', [input_dim, nodes], 
                                      initializer=initializer,
                                      collections=self.collections)
            biases = variable_in_cpu('biases', [nodes],
                                     initializer=b_initializer,
                                     collections=self.collections)
            full = nonlinearity(tf.matmul(Input, weights) + biases,
                                name='output_'+nl)
                    
        return full
    

class BatchNormalizationLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}
    
    def __call__(self, Input, momentum=0.9, eps=1e-5, scope=None, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        with tf.variable_scope(scope or 'bnL'):
            bn = nonlinearity(tf.contrib.layers.batch_norm(Input, decay=momentum, epsilon=eps,
                                                           scale=True,
                                                           variables_collections=self.collections) )
            return tf.identity(bn, name='batch_norm')
        
        
class ConvLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'tanh' : tf.nn.tanh, 'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}

    
    def __call__(self, Input, num_filters_out, kernel_height=5, kernel_width=5, strides_height=2,
                 strides_width=2, scope=None, stddev=0.02, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        
        if self.collections is not None:
            self.collections += [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]

        kernel_shape = [kernel_height, kernel_width, Input.get_shape()[-1], num_filters_out]
        with tf.variable_scope(scope or 'convL'):
            kernel = tf.get_variable(name='kernel', shape=kernel_shape,
                                     initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     collections=self.collections )
            conv = tf.nn.conv2d(Input, kernel, strides=[1, strides_height, strides_width, 1],
                                padding='SAME')
            bias = tf.get_variable('bias', [num_filters_out], initializer=tf.constant_initializer(0.0),
                                   collections=self.collections)
            
            conv = nonlinearity(tf.nn.bias_add(conv, bias))
            
        return tf.identity(conv, name='conv_'+nl)