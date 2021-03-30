#!/usr/bin/env python3

"""
Model script for ACTPC implementation in tensorflow-gpu.

Code for the model can be found:

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk

"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM


class static_Encoder(layers.Layer):
    def __init__(self, hidden_layers = 0, hidden_nodes = 20, state_fn = "tanh", recurrent_fn = "sigmoid",intermediate_dim = 32, dropout = 0.7, recurrent_dropout = 0.7,
                 masking = True, name = "static encoder", training = True):
        super().__init__()
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.state_fn         = state_fn
        self.recurrent_fn     = recurrent_fn
        self.intermediate_dim = intermediate_dim
        self.dropout          = dropout
        self.recurrent_dropout= recurrent_dropout
        self.masking          = masking
        self.name             = name
        self.training         = training
        self.output_layer     = LSTM(units = self.intermediate_dim, activation = self.state_fn,  recurrent_activation = self.recurrent_fn,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_sequences=False, return_state = False, stateful=False)
        # Add intermediate layers
        for layer_id_ in range(hidden_layers):
            layer_            = LSTM(units = self.hidden_nodes, return_sequences = True, )


    def __call__(self, inputs):

        embedding             = self.output_layer(    , mask =  , training = , initial_state = )

s
class ACTPC(tf.Keras.Model):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """
    def __init__(self):
        super().__init__()
        self.static_Encoder    = static_Encoder()
        self.static_Predictor  = static_Predictor()
        self.selector          = Selector()

    def call(self, inputs, training = True):
        embedding     = self.static_Predictor(inputs)
        cluster_probs = self.selector(embedding)

        # Sample cluster
        cluster_samp  = tf.random.categorical(logits = cluster_probs, num_samples = 1, seed = 1717, name = 'cluster sampling')
        cluster_emb   = tf.gather_nd(params = cluster_embeddings, indices = tf.expand_dims(cluster_samp, -1))

        # Feed cluster as input to predictor
        output_       = self.static_Predictor(cluster_emb)
























