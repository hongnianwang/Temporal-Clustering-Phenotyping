#!/usr/bin/env python3

"""
Model script for ACTPC implementation in tensorflow-gpu.

Code for the model can be found:

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk

"""

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Masking, Dense
import tensorflow.keras.layers as layers
    from tensorflow.math import log, squared_difference


# Define custom loss functions
def predictive_clustering_loss(y_true, y_pred, y_type = 'categorical', name = 'pred_clus_L'):

    """Compute prediction clustering loss between predicted output and true output.
     Inputs have shape (batch_size, num_classes). There are a variety of different settings:

    - Binary:  Computes Binary Cross Entropy. Class/Event occurence is matched with a dimension.
                y_true with entries in [0,1], and y_pred with value between (0,1)
    - Categorical: Computes Cross Entropy Loss. Class assigned by highest value dimension.
                y_true is a one-hot encoding, and y_pred is a probabilistic vector.
    - Continuous: Computes L2 loss. Similar to the Binary case, but class attributes are continuous.
                y_true and y_pred both with real-value entries.

    Returns: Loss value between sample true y and predicted y based on y_type of shape (batch_size)
    """
    if y_type == 'binary':
        # Compute Binary Cross Entropy. y_pred output of sigmoid function to avoid taking log of infty.
        batch_loss = - tf.reduce_mean(tf.reduce_sum(y_true * log(y_pred) + (1-y) * log(y_pred),
                                                     axis = -1), name = name)

    elif y_type == 'categorical':
        # Compute Categorical Cross Entropy. y_pred output of softmax function to model probability vector.
        batch_loss = -tf.reduce_mean(tf.reduce_sum(y_true * log(y_pred), axis = -1), name = name)

    elif y_type == 'continuous':
        # Compute L2 Loss. y_pred not given final output function.
        batch_loss = tf.reduce_mean(tf.reduce_sum((y_true - y_pred) ** 2, axis = -1), name = name)

    else:
        raise Exception('y_type not well-define. Only possible values are {`binary`', `'categorical`',
                                                                                       `'continuous`')
    return batch_loss

def cluster_probability_entropy_loss(y_prob, name = 'clus_entr_L'):

    """
    Compute Entropy loss on Cluster Probability assignments.
    Inputs have shape (batch_size, num_classes), and y_prob is a probabilistic vector.

    :param y_prob: a probabilistic vector
    :return: Entropy loss, defined as - sum pi*log(pi) with minimum obtained by one-hot probability vectors.
    """
    # assert tf.reduce_all(tf.reduce_sum(y_prob, axis = -1) , axis = None)
    batch_loss = tf.reduce_mean(-tf.reduce_sum(y_prob * log(y_prob), axis = -1), name = name)

    return batch_loss

def embedding_separation_loss(y_embeddings, name = 'emb_sep_L'):

    """
    Compute Embedding separation Loss on embedding vectors.

    y_embeddings: shape (num_clusters, latent_dim) - for each cluster i, y_embeddings[i] is the
    corresponding embedding of cluster in latent dimension.

    return: A Embedding separation loss (how separate are the clusters in the latent space). Only L1
    loss in latent space considered so far.
    """
    embedding_column = tf.expand_dims(y_embeddings, axis = 1)  # shape K, 1, latent_dim
    embedding_row    = tf.expand_dims(y_embeddings, axis = 0)  # shape 1, K, latent_dim

    # Compute L1 distance
    pairwise_loss    = tf.reduce_sum(squared_difference(embedding_column, embedding_row),
                                     axis = -1)  # shape K, K
    loss             = tf.reduce_sum(pairwise_loss, axis = None, name = 'emb_sep_L')

    return loss




class static_Encoder(layers.Layer):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """

    def __init__(self, intermediate_dim = 32, hidden_layers = 0, hidden_nodes = 20, state_fn = "tanh", recurrent_fn = "sigmoid", dropout = 0.7, recurrent_dropout = 0.7,
                 mask_value   = 0.0, name = "static encoder", training = True):
        super().__init__(name = name)
        self.intermediate_dim = intermediate_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.state_fn         = state_fn
        self.recurrent_fn     = recurrent_fn
        self.dropout          = dropout
        self.recurrent_dropout= recurrent_dropout
        self.mask_value       = mask_value
        self.name             = name
        self.training         = training
        self.output_layer     = LSTM(units = self.intermediate_dim, activation = self.state_fn, recurrent_activation = self.recurrent_fn,return_sequences=False,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful=False)
        # Add intermediate layers
        for layer_id_ in range(self.hidden_layers):
            layer_            = LSTM(units = self.hidden_nodes, return_sequences = True, activation = self.state_fn, recurrent_activation = self.recurrent_fn,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful = False)
            setattr(self, 'layer_' + str(layer_id_), layer_)


    def __call__(self, inputs):
        x = inputs
        if masking is not None:
            try:
                x = Masking(mask_value = self.mask_value)(inputs)

            except Exception:
                print('Masking did not work!')

        # Iterate through hidden layers
        for layer_id_ in range(self.hidden_layers):
            layer_   = getattr(self, 'layer_' + str(layer_id_))
            x        = layer_(x, trainable = self.training)

        latent_rep   = self.output_layer(x, trainable = self.training)

        return latent_rep



class multilayer_perceptron(layers.Layer):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """

    def __init__(self, output_dim, name, hidden_layers = 2, hidden_nodes = 30, activation_fn = 'sigmoid', output_fn = 'softmax', training = True):
        super().__init__(name = name)
        self.name             = name
        self.output_dim       = output_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.activation_fn    = activation_fn
        self.output_fn        = output_fn
        self.training         = training
        self.output_layer     = Dense(units = self.output_dim, activation = self.output_fn, trainable = self.training, name = self.name)

        for layer_id_ in range(self.hidden_layers):
            layer_            = Dense(units = self.hidden_nodes, activation = self.activation_fn, trainable = self.training, name = self.name)
            setattr(self, 'layer_' + str(layer_id_), layer_)

    def __call__(self, inputs):
        x = inputs
        for layer_id_ in range(self.hidden_layers):
            x = getattr(self, 'layer_' + str(layer_id))(x)

        y_pred                = self.output_layer(x)

        return y_pred





class ACTPC(tf.Keras.Model):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """
    def __init__(self, K, output_dim, name, y_type):
        super().__init__(name  = name)
        self.K                 = self.num_clusters
        self.output_dim        = output_dim
        self.y_type            = y_type
        self.static_Encoder    = static_Encoder(intermediate_dim = 32, hidden_layers = 1, hidden_nodes = 10, name = 'encoder', training = True)
        self.static_Predictor  = multilayer_perceptron(output_dim = output_dim, name = 'selector', hidden_layers = 2, output_fn = 'softmax', training = True)
        self.selector          = multilayer_perceptron(output_dim = self.K, name = 'selector', hidden_layers = 2, output_fn = 'softmax', training = True)


    def call(self, x_inputs, y_inputs):
        embedding     = self.static_Encoder(inputs)
        cluster_probs = self.selector(embedding)

        # Sample cluster
        cluster_samp  = tf.random.categorical(logits = cluster_probs, num_samples = 1, seed = 1717, name = 'cluster sampling')
        cluster_emb   = tf.gather_nd(params = cluster_embeddings, indices = tf.expand_dims(cluster_samp, -1))

        # Feed cluster as input to predictor
        y_pred        = self.static_Predictor(cluster_emb)
        self.add_loss(predictive_clustering_loss(
                y_true = y_inputs,
                y_pred = y_pred,
                y_type = self.y_type,
                name   = 'pred_clus_L'
        ))
        self.add_loss(cluster_probability_entropy_loss(
                y_prob = cluster_probs,
                name   = 'clus_entr_L'
        ))

        return y_pred

























