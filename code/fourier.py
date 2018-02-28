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
import tensorflow as tf

from code.layers import ConvLayer, FullLayer, BatchNormalizationLayer
from code.datetools import addDateTime

DTYPE = tf.float32

def load_mnist(params):
    """
    """
    data_root = params.local_data_root
    data_dir = data_root
#     data_dir = os.path.join(data_root, 'mnist')
    
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    Xtrain = loaded[16:].reshape([60000, 28, 28, 1]).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    Ytrain = np.asarray(loaded[8:].reshape([60000]).astype(np.float))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    Xvalid = loaded[16:].reshape([10000,28,28,1]).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    Yvalid = np.asarray(loaded[8:].reshape([10000]).astype(np.float))
    
    return {'Xtrain' : Xtrain, 'Ytrain' : Ytrain, 'Xvalid' : Xvalid,
            'Yvalid' : Yvalid} 

class Fourier():
    """
    """
    def __init__(self, params):
        """
        """
        self.params = params
        self.yDim = yDim = params.yDim
        self.batch_size = params.batch_size
        
        self.img_height, self.img_width = params.img_height, params.img_width
        data_dims = [self.img_height, self.img_width, params.num_channels]
        
        self.data_dict = load_mnist(params)
        with tf.variable_scope('FOURIER', reuse=tf.AUTO_REUSE):
            self.Y = tf.placeholder(DTYPE, [None, yDim], name='Y')
            self.X = tf.placeholder(DTYPE, [None] + data_dims, name='X')
            
            self.output = self.get_prediction()
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,
                                                                                  logits=self.output) )
            
            self.loss_summ = tf.summary.scalar('Loss', self.loss)
            
            self.train_step = tf.get_variable("global_step", [], tf.int32,
                                              tf.zeros_initializer(),
                                              trainable=False)
            opt = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1)
            opt_varsgrads = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(opt_varsgrads, global_step=self.train_step,
                                                name='Train')

    def get_prediction(self, Input=None, mode='a_capella'):
        """
        """
        if Input is None: Input = self.X
        
        # TODO: put checks
        num_filters = 64
        convolutional_layer = ConvLayer()
        fully_connected_layer = FullLayer()
        batch_norm_layer = BatchNormalizationLayer()
        if mode == 'a_capella': 
            with tf.variable_scope("a_capella"):
                conv1 = convolutional_layer(Input, num_filters, scope='conv1', nl='relu',
                                            strides_height=2, strides_width=2)
                
                
                conv2 = convolutional_layer(conv1, num_filters//2, scope='conv2', nl='relu',
                                            strides_height=2, strides_width=2)
                conv2 = tf.reshape(conv2, [-1, (num_filters//2)*(self.img_height//4)*(self.img_width//4)])
                bn2 = batch_norm_layer(conv2, scope='bn2', nl='lkyrelu')
                
                full3 = fully_connected_layer(bn2, 512, scope='full3', nl='relu')
                bn3 = batch_norm_layer(full3, scope='bn3', nl='lkyrelu')
                
                output = fully_connected_layer(bn3, self.yDim, scope='output', nl='linear')
                
        if mode == 'fourier1':
            with tf.variable_scope("fourier1"):
                init_id = tf.identity(Input)
                init_fou = tf.fft2d(tf.cast(tf.squeeze(Input), tf.complex64))
                init_fou = tf.cast(tf.expand_dims(tf.abs(init_fou), axis=3), DTYPE)
                init = tf.concat([init_id, init_fou], axis=3)
                
                conv1 = convolutional_layer(init, 64, scope='conv1', nl='relu')
                
                conv2 = convolutional_layer(conv1, 64, scope='conv2', nl='relu')
                conv2 = tf.reshape(conv2, [self.batch_size, -1])
                
                output = fully_connected_layer(conv2, self.yDim, scope='output', nl='relu')

        return output

    @staticmethod
    def to_one_hot(data, yDim):
        """
        """
        b_size = data.shape[0]
        one_hot = np.zeros([b_size, yDim])
        for i in range(b_size):
            one_hot[i][int(data[i])] = 1.0
        return one_hot
    
    @staticmethod
    def nptensorfloat_to_one_hot(data):
        """
        """
        result = np.zeros_like(data)
        idx_max = np.argmax(data, axis=1)
        for i, idx in enumerate(idx_max):
            result[i,idx] = 1.0
        return result
            

    def train(self, sess, num_epochs=20):
        """
        Remember to initialize all variables outside this method
        """
        batch_size = self.batch_size
        # Placeholder for some more summaries that may be of interest.
        merged_summaries = tf.summary.merge([self.loss_summ])
        self.writer = tf.summary.FileWriter(addDateTime('./logs/log'))
        
        Ytrain = self.to_one_hot(self.data_dict['Ytrain'], self.yDim)
        Xtrain = self.data_dict['Xtrain']
        Yvalid = self.to_one_hot(self.data_dict['Yvalid'], self.yDim)
        Xvalid = self.data_dict['Xvalid']
        for ep in range(num_epochs):
            num_batches = len(Xtrain) // batch_size
            for idx in range(0, num_batches):
                batch_images = Xtrain[idx*batch_size:(idx+1)*batch_size]
                batch_labels = Ytrain[idx*batch_size:(idx+1)*batch_size]
                
                _, loss, summaries = sess.run([self.train_op, self.loss, merged_summaries], 
                                          feed_dict={'FOURIER/X:0' : batch_images,
                                                     'FOURIER/Y:0' : batch_labels})
                if (idx*10)//num_batches > ((idx-1)*10)//num_batches:
                    print((100*idx)//num_batches,'% processed...')
                    
            self.writer.add_summary(summaries, ep)
            print('Ep, cost:', ep, loss)
            ypred = sess.run(tf.nn.softmax(self.output), feed_dict={'FOURIER/X:0' : Xtrain[:100]})
            ypred = self.nptensorfloat_to_one_hot(ypred)
            accuracy = np.sum(ypred*Ytrain[:100])/100
            print('Accuracy', accuracy)
            
            if ep % 50 == 0:
                Ypred_valid = sess.run(self.output, feed_dict={'FOURIER/X:0' : Xvalid[:100]})
                Ypred_valid = self.nptensorfloat_to_one_hot(Ypred_valid)
                accuracy = np.sum(Ypred_valid*Yvalid[:100])/100
                print('Accuracy', accuracy)
            
            
            # TODO: Plot something, implement validation, etc. 

         
            

    


        