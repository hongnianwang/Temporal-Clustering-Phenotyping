#!/usr/bin/env python3
"""
Attention-model class model file.

Updated: 12 March 2021
Created by: Henrique Aguiar, Institute of Biomedical Engineering, Department of Engineering Science, 
University of Oxford

if you have any queries, please reach me at henrique.aguiar@ndcn.ox.ac.uk
                
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import  MeanAbsoluteError


class Encoder(layers.Layer):

    """Encoder Layer for multivariate time-series.
    
    Inputs: num_nodes   - number of intermediate nodes in each layer (default = 20)
            num_layers  - number of intermediate layers (default = 0, no intermediate layers)
            z_dim       - dimension of each (sample, time-step) output encoding. (default = 1)
            masking     - whether to mask inputs (default = True)

            Note that the layer inputs should be of the form [N, time_steps, x_dim]

    Returns: Returns an Encoder Layer which converts inputs of shape [N, time_steps, x_dim] to shape [N, time_steps, y_dim]
    """

    def __init__(self, num_nodes = 20, num_layers = 0, z_dim = 1, masking = True):
        super().__init__()
        self.num_nodes  = num_nodes
        self.num_layers = num_layers
        self.z_dim      = z_dim
        self.masking    = masking

        for int_ in range(self.num_layers):
            # Each intermediate layer should output [ _, time_steps, num_nodes] tensors
            setattr(self, 'int_layer_' + str(int_), layers.LSTM(units = self.num_nodes,
                                                                activation = 'tanh',
                                                                return_sequences = True))
        # Final Layer projects to output of shape [_, time_steps, z_dim]
        self.proj       = layers.LSTM(units = self.z_dim,
                                      activation = 'tanh',
                                      return_sequences = True)


    def call(self, inputs):

        if self.masking:
            x = layers.Masking(inputs)
        else:
            x = inputs

        for int_ in range(self.num_layers):
            # Output for each intermediate layer
            layer_ = getattr(self, 'int_layer_' + str(int_))
            x = layer_(x, training = True, initial_state = None)
        x = self.proj(x, training = True)

        return x

class Decoder(Encoder):
    """Decoder Layer for multivariate time-series.

     Inputs: num_nodes   - number of intermediate nodes in each layer (default = 20)
             num_layers  - number of intermediate layers (default = 0, no intermediate layers)
             og_dim      - dimension of original multivariate time-series. (default = 10)
             masking     - whether to mask inputs (default = False)

            Note that the layer inputs should be of the form [N, time_steps, x_dim]

     Returns: Returns an Encoder Layer which converts inputs of shape [N, time_steps, x_dim] to shape [N, time_steps, y_dim]
     """

    def __init__(self, num_nodes=20, num_layers=0, og_dim = 8, masking=False):
        super().__init__()
        self.num_nodes  = num_nodes
        self.num_layers = num_layers
        self.og_dim     = og_dim
        self.masking    = masking

        for int_ in range(self.num_layers):
            # Each intermediate layer should output [ _, time_steps, num_nodes] tensors
            setattr(self, 'int_layer_' + str(int_), layers.LSTM(units=self.num_nodes,
                                                                activation='tanh',
                                                                return_sequences=True))
        # Final Layer projects to output of shape [_, time_steps, z_dim]
        self.proj = layers.LSTM(units=self.og_dim,
                                activation='tanh',
                                return_sequences=True)

    def call(self, inputs):

        if self.masking:
            x = layers.Masking(inputs)
        else:
            x = inputs

        for int_ in range(self.num_layers):
            # Output for each intermediate layer
            layer_ = getattr(self, 'int_layer_' + str(int_))
            x = layer_(x, training=True, initial_state=None)
        x = self.proj(x, training=True)

        return x


class LSTM_AE(tf.keras.Model):

    """
    Combines the Encoder-Decoder layers into an end-to-end model for training
    """

    def __init__(self, original_dim, z_dim = 1, num_nodes = 20, num_layers = 0):
        super().__init__()
        self.original_dim = original_dim
        self.Encoder      = Encoder(num_nodes = num_nodes,
                                    num_layers= num_layers,
                                    z_dim     = z_dim)
        self.Decoder      = Decoder(num_nodes = num_nodes,
                                    num_layers= num_layers,
                                    og_dim    = self.original_dim)

    def call(self, inputs):
        embedding    = self.Encoder(inputs)
        reconstruct  = self.Decoder(embedding)

        # Add Loss function
        reconstruction_loss = MeanAbsoluteError(name = 'reconstruction_loss')
        self.add_loss(reconstruction_loss)

        return reconstruct



    
    
    
    
    
    
    
    
    
    
    
    
    
    

