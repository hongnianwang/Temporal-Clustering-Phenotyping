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
from utils import embedding_separation_loss, cluster_probability_entropy_loss, predictive_clustering_loss
import time

from sklearn.cluster import KMeans

class Encoder(layers.Layer):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """

    def __init__(self, intermediate_dim=32, hidden_layers=0, hidden_nodes=20, state_fn="tanh", recurrent_fn="sigmoid",
                 dropout=0.7, recurrent_dropout=0.7, mask_value=0.0, name="static encoder", training=True):
        super().__init__(name =name)
        self.intermediate_dim =intermediate_dim
        self.hidden_layers    =hidden_layers
        self.hidden_nodes     =hidden_nodes
        self.state_fn         =state_fn
        self.recurrent_fn     =recurrent_fn
        self.dropout          =dropout
        self.recurrent_dropout= recurrent_dropout
        self.mask_value       = mask_value
        self.training         = training
        self.output_layer     = LSTM(units = self.intermediate_dim, activation = self.state_fn,
                                     recurrent_activation = self.recurrent_fn, return_sequences=False,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful=False)
        # Add intermediate layers
        for layer_id_ in range(self.hidden_layers):
            layer_            = LSTM(units = self.hidden_nodes, return_sequences = True, activation = self.state_fn, recurrent_activation = self.recurrent_fn,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful = False)
            setattr(self, 'layer_' + str(layer_id_), layer_)


    def __call__(self, inputs):
        x = inputs
        if self.mask_value is not None:
            try:
                x = Masking(mask_value = self.mask_value)(inputs)

            except Exception:
                print('Masking did not work!')

        # Iterate through hidden layers
        for layer_id_ in range(self.hidden_layers):
            layer_   = getattr(self, 'layer_' + str(layer_id_))
            x        = layer_(x, trainable = self.training)

        latent_rep   = self.output_layer(x, training = self.training)

        return latent_rep



class MLP(layers.Layer):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """

    def __init__(self, output_dim, name, hidden_layers = 2, hidden_nodes = 30, activation_fn = 'sigmoid', output_fn = 'softmax', training = True):
        super().__init__(name = name)
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
            x = getattr(self, 'layer_' + str(layer_id_))(x)

        y_pred                = self.output_layer(x)

        return y_pred


class ACTPC(tf.keras.Model):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """
    def __init__(self, K, output_dim, alpha, init_epochs_ac, name , y_type, embeddings = None):
        super().__init__(name  = name)
        self.K                 = self.num_clusters
        self.output_dim        = output_dim
        self.alpha             = alpha
        self.y_type            = y_type
        self.init_epochs_1     = init_epochs_ac
        self.embeddings        = embeddings
        self.Encoder           = Encoder(intermediate_dim = 32, hidden_layers = 1, hidden_nodes = 10, name = 'encoder', training = True)
        self.Predictor         = MLP(output_dim = output_dim, name = 'selector', hidden_layers = 2, output_fn = 'softmax', training = True)
        self.Selector          = MLP(output_dim = self.K, name = 'selector', hidden_layers = 2, output_fn = 'softmax', training = True)


    def call(self, inputs):
        x_inputs, y_inputs     = inputs

        embedding     = self.static_Encoder(x_inputs)
        cluster_probs = self.selector(embedding)

        # Sample cluster
        cluster_samp  = tf.random.categorical(logits = cluster_probs, num_samples = 1, seed = 1717, name = 'cluster sampling')
        cluster_emb   = tf.gather_nd(params = self.embeddings, indices = tf.expand_dims(cluster_samp, -1))

        # Feed cluster as input to predictor
        y_pred        = self.static_Predictor(cluster_emb)

        return y_pred, cluster_probs

    def init_train(self, init_epochs_1 = 10, batch_size = 128 X, y = None):

        """
        Initialise AC, embeddings and selector
        """
        epochs  = init_epochs_1
        for epoch in range(epochs):

            print("\nStart of epoch %d" % (epoch, ))
            start_time = time.time()

            # Iterate through batches
            for step, (x_train_batch, y_train_batch) in enumerate(X, y):
                with tf.GradientTape() as tape:
                    # Compute "Auto Encoder" predicted y
                    latent_projs = self.Encoder(x_train_batch, training = True)
                    y_pred       = self.Predictor(latent_projs, training = True)

                    # Compute loss function
                    loss_value   = predictive_clustering_loss(
                        y_true = y_train_batch,
                        y_pred = y_pred,
                        y_type = 'categorical',
                        name   = 'pred_clus_loss'
                    )
                # Compute gradients and update weights
                train_weights = [var for var in self.trainable_weights if 'encoder ' in var or 'predictor' in var]
                gradient = tape.gradient(loss_value, train_weights)
                optimizer.apply_gradients(zip(gradient, train_weights))

                # Log Results every 10 batches
                if step % 10 == 0:
                    print("Training Actor-Critic Initialisation Loss for (one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("%d samples have been seen so far" % ((step + 1) * batch_size))


























