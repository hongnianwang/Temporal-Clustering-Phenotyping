#!/usr/bin/env python3

"""
Model script for ACTPC implementation in tensorflow-gpu.

Code for the model can be found:

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Masking, Dense
import tensorflow.keras.layers as layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from utils import embedding_separation_loss, cluster_probability_entropy_loss, predictive_clustering_loss, actor_predictive_clustering_loss
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
                 dropout=0.7, recurrent_dropout=0.7, mask_value=0.0, name='encoder'):
        super().__init__(name =name)
        self.intermediate_dim =intermediate_dim
        self.hidden_layers    =hidden_layers
        self.hidden_nodes     =hidden_nodes
        self.state_fn         =state_fn
        self.recurrent_fn     =recurrent_fn
        self.dropout          =dropout
        self.recurrent_dropout= recurrent_dropout
        self.mask_value       = mask_value

    def build(self, input_shape = None):
        self.output_layer     = LSTM(units = self.intermediate_dim, activation = self.state_fn,
                                     recurrent_activation = self.recurrent_fn, return_sequences=False,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful=False,
                                     name = self.name)
        # Add intermediate layers
        for layer_id_ in range(self.hidden_layers):
            layer_            = LSTM(units = self.hidden_nodes, return_sequences = True, activation = self.state_fn, recurrent_activation = self.recurrent_fn,
                                     dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_state = False, stateful = False,
                                     name = self.name)
            setattr(self, 'layer_' + str(layer_id_), layer_)

    def call(self, inputs, training = True):
        x = inputs
        if self.mask_value is not None:
            try:
                x = Masking(mask_value = self.mask_value)(inputs)

            except Exception:
                print('Masking did not work!')

        # Iterate through hidden layers
        for layer_id_ in range(self.hidden_layers):
            layer_   = getattr(self, 'layer_' + str(layer_id_))
            x        = layer_(x)

        latent_rep   = self.output_layer(x, training = training)

        return latent_rep



class MLP(layers.Layer):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """

    def __init__(self, output_dim, name, hidden_layers = 2, hidden_nodes = 30, activation_fn = 'sigmoid', output_fn = 'softmax'):
        super().__init__(name = name)
        self.output_dim       = output_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.activation_fn    = activation_fn
        self.output_fn        = output_fn

    def build(self, input_shape = None):
        self.output_layer     = Dense(units = self.output_dim, activation = self.output_fn, name = self.name)

        for layer_id_ in range(self.hidden_layers):
            layer_            = Dense(units = self.hidden_nodes, activation = self.activation_fn, name = self.name)
            setattr(self, 'layer_' + str(layer_id_), layer_)

    def call(self, inputs, training = True):
        x = inputs
        for layer_id_ in range(self.hidden_layers):
            x = getattr(self, 'layer_' + str(layer_id_))(x, training = training)

        y_pred                = self.output_layer(x)

        return y_pred



class attention(layers.Layer):

    def __init__(self, name = 'attention'):
        super().__init__(name = name)

    def build(selfself, input_shape = None):
        pass

    def call(self, inputs, training = True):
        pass




L1        = metrics.Mean(name="pred_clus_loss")
L1_actor  = metrics.Mean(name="actor_pred_clus_loss")
L2        = metrics.Mean(name="clus_entr_loss")
L3        = metrics.Mean(name="emb_sep_loss")
AUC       = metrics.AUC(name="AUROC")

class ACTPC(tf.keras.Model):

    """
    Model class for AC-TPC model. Input parameters are:

    -

    -

    output:
    """
    def __init__(self, num_clusters, latent_dim, output_dim, beta, alpha, init_epochs_ac = 1, init_epochs_sel = 1,
                 name = 'ACTPC' , y_type = 'categorical', embeddings = None, training = True,
                 optimizer = optimizers.Adam, seed = 2323):
        super().__init__(name  = name)
        self.K                 = num_clusters
        self.latent_dim        = latent_dim
        self.output_dim        = output_dim
        self.beta              = beta
        self.alpha             = alpha
        self.y_type            = y_type
        self.init_epochs_1     = init_epochs_ac
        self.init_epochs_2     = init_epochs_sel
        self.embeddings        = embeddings
        self.optimizer         = optimizer
        self.seed              = seed
        self.Encoder           = Encoder(intermediate_dim = self.latent_dim,
                                         hidden_layers = 1, hidden_nodes = 10, name = 'encoder')
        self.Predictor         = MLP(output_dim = self.output_dim,
                                     hidden_layers = 2, output_fn = 'softmax', name = 'predictor')
        self.Selector          = MLP(output_dim = self.K,
                                     hidden_layers = 2, output_fn = 'softmax', name = 'selector')
        self.embeddings        = tf.Variable(initial_value = tf.zeros(shape = [self.K, self.latent_dim], dtype = 'float32'),
                                             trainable = True, name = 'embeddings' )

    def call(self, inputs, training = True):
        x_inputs      = inputs

        latent_projs  = self.Encoder(x_inputs, training = training)
        cluster_probs = self.Selector(latent_projs, training = training)

        # Sample cluster
        cluster_samp  = tf.squeeze(tf.random.categorical(logits = cluster_probs, num_samples = 1, seed = 1717, name = 'cluster_sampling'))
        cluster_emb   = tf.gather_nd(params = self.embeddings, indices = tf.expand_dims(cluster_samp, -1))

        # Feed cluster as input to predictor
        y_pred        = self.Predictor(cluster_emb, training = training)

        return y_pred


    def compile(self, optimizer, **kwargs):
        super(ACTPC, self).compile(**kwargs)
        self.optimizer = optimizers.Adam()



    def initialise(self, X, y, batch_size = 2000, optimizer = optimizers.Adam, verbose = 0):

        """
        Initialise AC, embeddings and selector
        """
        X, y = X.astype('float32'), y.astype('float32')
        print('\nInitialising Actor-Critic networks.')
        epochs  = self.init_epochs_1
        data    = tf.data.Dataset.from_tensor_slices((X, y))

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch, ))
            start_time = time.time()

            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size = 5000).batch(batch_size)

            # Iterate through batches
            for step, (x_train_batch, y_train_batch) in enumerate(data_shuffle):
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
                train_weights = [var for var in self.trainable_weights if 'encoder' in var.name or 'predictor' in var.name]
                gradient = tape.gradient(loss_value, train_weights)
                optimizer().apply_gradients(grads_and_vars=zip(gradient, train_weights))

                # Log Results every 10 batches
                if step % 2 == 0:
                    print("Training Actor-Critic Initialisation Loss for (one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("%d samples have been seen so far" % ((step + 1) * batch_size))

        print('AC-initialisation complete')

        # Initialise embeddings
        latent_projs = self.Encoder(X, training = False)
        init_km = KMeans(n_clusters=self.K, init='k-means++', verbose=verbose,
                         random_state=self.seed)
        init_km.fit(latent_projs)
        print('\nCluster initialisation complete')
        # Obtain cluster centres and sample cluster assignments
        centroids = init_km.cluster_centers_
        cluster_assig = init_km.predict(latent_projs)


        # Check shapes
        assert centroids.shape == (self.K, self.latent_dim)
        assert cluster_assig.shape == (X.shape[0],)

        self.embeddings.assign(tf.convert_to_tensor(centroids, name = 'embedding_reps', dtype = 'float32'), name = 'embeddings_init')

        # Initialise Selector
        print('\nInitialising Selector Networks')
        epochs = self.init_epochs_2

        # Input cluster_true values in useful format
        num_clusters = self.K
        cluster_true = np.eye(num_clusters)[cluster_assig]   # Returns a numpy array of shape (X.shape, num_clusters) with one-hot encoding for each assignment
        data = tf.data.Dataset.from_tensor_slices((X, cluster_true.astype('float32')))

        for epoch in range(epochs):

            print("\nStart of epoch %d" % (epoch, ))
            start_time = time.time()

            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size=5000).batch(batch_size)

            # Iterate through batches
            for step, (x_train_batch, y_train_batch) in enumerate(data_shuffle):
                with tf.GradientTape() as tape:
                    # Compute "Auto Encoder" predicted y
                    latent_projs = self.Encoder(x_train_batch, training = False)
                    cluster_pred = self.Selector(latent_projs, training = True)

                    # Compute loss function (Cross-Entropy with cluster assignment as true value)
                    loss_value   = predictive_clustering_loss(
                        y_true = y_train_batch,
                        y_pred = cluster_pred,
                        y_type = 'categorical',
                        name   = 'pred_clus_loss'
                    )
                # Compute gradients and update weights
                train_weights = [var for var in self.trainable_weights if 'selector' in var.name]
                gradient = tape.gradient(loss_value, train_weights)
                optimizer().apply_gradients(zip(gradient, train_weights))

                # Log Results every 10 batches
                if step % 2 == 0:
                    print("Training Selector Initialisation Loss for (one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("%d samples have been seen so far" % ((step + 1) * batch_size))

                if step > 20:
                    break

        print('Selector initialisation complete. Ready for main training')



    # Main training step iterative update
    def train_step(self, inputs, training = True):

        X, y = inputs

        critic_vars   = [var for var in self.trainable_weights if 'predictor' in var.name]

        latent_projs  = self.Encoder(X, training=False)
        cluster_probs = self.Selector(latent_projs, training=False)

        # Sample cluster
        cluster_samp  = tf.squeeze(tf.random.categorical(logits=cluster_probs, num_samples=1, seed=1717, name='cluster_sampling'))
        cluster_emb   = tf.gather_nd(params=self.embeddings, indices=tf.expand_dims(cluster_samp, -1))

        "Update Critic"
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(critic_vars + [cluster_emb])

            # Forward pass
            y_pred = self.Predictor(cluster_emb, training = True)

            # Compute loss
            loss_critic = predictive_clustering_loss(y_true = y, y_pred = y_pred,
                y_type = self.y_type, name   = 'pred_clus_loss'
            )

        critic_grad = tape.gradient(target = loss_critic, sources = critic_vars)
        self.optimizer.apply_gradients(zip(critic_grad, critic_vars))
        print('Critic update completed.')


        "Update Actor - probabilistic assignments remain the same as before"
        actor_vars     = [var for var in self.trainable_weights if 'encoder' in var.name or 'selector' in var.name]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actor_vars)

            # Compute cluster probabilities, sampled cluster and probability of sampled cluster
            cluster_probs   = self.Selector(self.Encoder(X, training = True), training = True)
            cluster_samp    = tf.squeeze(tf.random.categorical(logits=cluster_probs, num_samples=1, seed=1717, name='cluster_sampling'))
            pi_clus_assign  = tf.gather_nd(params=cluster_probs , indices=tf.stack((tf.range(X.shape[0], dtype = 'int64'), cluster_samp), axis = 1))
            
            # Compute predicted y
            cluster_emb     = tf.gather_nd(params=self.embeddings, indices=tf.expand_dims(cluster_samp, -1))
            y_pred          = self.Predictor(cluster_emb, training = False)

            # Compute L1 loss weighted by cluster confidence
            weighted_loss_actor_1 = actor_predictive_clustering_loss(
                y_true = y, y_pred = y_pred, cluster_assignment_probs = pi_clus_assign,
                y_type = self.y_type, name = 'actor_pred_clus_loss'
            )

            # Compute Cluster Entropy loss
            entr_loss  = cluster_probability_entropy_loss(
                y_prob = cluster_probs,
                name   = 'clust_entropy_loss'
            )

            loss_actor = weighted_loss_actor_1 + self.alpha * entr_loss

        # Compute gradients and update
        actor_grad = tape.gradient(target=loss_actor, sources = actor_vars)
        self.optimizer.apply_gradients(zip(actor_grad, actor_vars))
        print('Actor update complete.')


        "Update embeddings"
        embedding_vars = self.embeddings
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(embedding_vars)

            # Compute cluster_assignments and pred_y
            cluster_probs = self.Selector(self.Encoder(X, training=False), training=False)
            cluster_samp  = tf.squeeze(tf.random.categorical(logits=cluster_probs, num_samples=1, seed=1717, name='cluster_sampling'))
            mask_emb      = tf.one_hot(cluster_samp, depth = self.K)
            cluster_emb   = tf.linalg.matmul(
                a = mask_emb, b = embedding_vars
            )
            y_pred     = self.Predictor(cluster_emb, training=False)

            # Compute embedding separation loss and predictive clustering loss
            emb_loss_1 = predictive_clustering_loss(
                y_true = y,
                y_pred = y_pred,
                y_type = self.y_type,
                name   = 'emb_pred_clus_loss'
            )
            emb_loss_2 = embedding_separation_loss(
                y_embeddings = embedding_vars,
                name   = 'emb_sep_loss'
            )

            loss_emb   = emb_loss_1 + self.beta * emb_loss_2

        # Compute gradients and update
        emb_grad  =  tape.gradient(target=loss_emb, sources=embedding_vars)
        self.optimizer.apply_gradients(zip([emb_grad], [embedding_vars]))
        self.embeddings = embedding_vars
        print('Embeddings update complete.')


        # Update Loss functions
        L1.update_state(loss_critic)
        L1_actor.update_state(weighted_loss_actor_1)
        L2.update_state(entr_loss)
        L3.update_state(emb_loss_2)

        return {'pred_clus': L1.result(), 'actor_loss': L1_actor.result(), 'clus_entr': L2.result(), 'emb_sep': L3.result()}


    @property
    def metrics(self):

        return [L1, L1_actor, L2, L3]


    def compute_latent_reps(self, inputs, training = False):
        """
        Compute latent representation vectors given inputs (either a batch or array), with training properties specified.
        """
        latent_projs     = self.Encoder(inputs, training = training)

        return latent_projs



    def compute_clusters_and_probs(self, inputs, one_hot = False, training = False):
        """
        Compute cluster assignments and cluster probability assignments given inputs and training regimem.
        Cluster assignments in one hot encoding format specified by "one-hot" parameter
        """
        latent_projs    = self.compute_latent_reps(inputs, training = training)

        return self.compute_clusters_and_probs_from_latent(latent_projs, one_hot = one_hot, training = training)



    def compute_clusters_and_probs_from_latent(self, latent_projs, one_hot = False, training = False):
        """
        Compute cluster assignments and cluster probability assignments through latent projections
        and training regimen. Cluster assignments in one hot encoding format specified by "one-hot"
        """
        cluster_probs  = self.Selector(latent_projs, training = training)

        # Assign cluster with highest probability
        cluster_assign = tf.math.argmax(cluster_probs, axis = -1)

        return cluster_probs, cluster_assign



    def compute_y(self, inputs, training = False):
        """
        Compute predicted y_vectors given inputs (either a batch or array) with training properties specified (
        useful for training or prediction).
        """
        y_pred = self(inputs, training = training)

        return y_pred


















