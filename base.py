#!/usr/bin/env python3

"""
Model script for ACTPC implementation in tensorflow-gpu.

Code for the model can be found:

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Masking, Dense, TimeDistributed
import tensorflow.keras.layers as layers
from tensorflow.keras import metrics
import utils_base as aux
import time

from sklearn.cluster import KMeans




class Encoder(layers.Layer):
    """
    Class for an Encoder layer architecture. Input tensors expected to be of shape (bs x times x xdim), and
    output tensors expected of shape (bs x times x latent dim).

    Encoder Layer stacks LSTMs layers using the whole output sequences, so that the overall output is of size
    (bs x time_steps x latent dim), with [i, j, :] representing the latent vector from time-subsequence corresponding
    to patient i and from times t=0, to t=j.

    - intermediate_dim    : dimensionality of latent space for each sub-sequence. (default = 32)
    - hidden_layers       : Number of "hidden" LSTM layers. Total number of stacked layers is hidden_layers + 1
                            (default = 1)
    - hidden_nodes        : For "hidden" LSTM layers, the dimensionality of the intermediate tensors and cell state.
                            (default = '20')
    - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
    - recurrent_activation: The activation function to use on forget/input/output gates. (default = 'sigmoid')
    - dropout             : dropout rate to be used on cell state/output computation. (default = 0.6)
    - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0)
    - mask_value          : mask_value to feed to masking layer - time stamps imputed with this mask_value across
                            xdims will be ignored (imputation has taken place) (default = 0.0)
    - name                : Name on which to save layer (default = 'encoder')

    output: returns sequence of LSTM layers assigned to "name". This object implements a call attribute which converts
    inputs of size (bs x T x xdims) to (bs x T x intermediate_dim).
    """
    def __init__(self, intermediate_dim=32, hidden_layers=1, hidden_nodes=20, state_fn="tanh", recurrent_fn="sigmoid",
                 dropout=0.7, recurrent_dropout=0.0, mask_value=0.0, name='encoder'):
        super().__init__(name =name)
        self.intermediate_dim = intermediate_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.state_fn         = state_fn
        self.recurrent_fn     = recurrent_fn
        self.dropout          = dropout
        self.recurrent_dropout= recurrent_dropout
        self.mask_value       = mask_value

    def build(self, input_shape = None):
        # Add intermediate layers,
        for layer_id_ in range(self.hidden_layers):
            layer_            = LSTM(units = self.hidden_nodes,
                                     return_sequences = True,                   # True to generate seq for next layer
                                     activation = self.state_fn,
                                     recurrent_activation = self.recurrent_fn,
                                     dropout = self.dropout,
                                     recurrent_dropout = self.recurrent_dropout,
                                     return_state = False,
                                     name = self.name)
            setattr(self, 'layer_' + str(layer_id_), layer_)

        # Add output layer
        self.output_layer     = LSTM(units = self.intermediate_dim,
                                     activation = self.state_fn,
                                     recurrent_activation = self.recurrent_fn,
                                     return_sequences = True,                   # True to generate latent for each sub-
                                     dropout = self.dropout,                    # sequence
                                     recurrent_dropout = self.recurrent_dropout,
                                     return_state = False,
                                     name = self.name)
    # attribute to compute output
    def call(self, inputs, training = True):

        x = inputs

        # Compute mask
        if self.mask_value is not None:
            try:
                # Mask layer ignores time stamps with all entries given by mask_value. Mask propagated through LSTMs
                # by default
                x = Masking(mask_value = self.mask_value)(inputs)

            except Exception as e:
                print(e)
                raise ValueError('Masking did not work!')

        # Iterate through hidden layers
        for layer_id_ in range(self.hidden_layers):
            layer_   = getattr(self, 'layer_' + str(layer_id_))
            x        = layer_(x, training = training)

        latent_rep  = self.output_layer(x, training = training)   # shape (bs, T, intermediate_dim)

        return latent_rep



class MLP(layers.Layer):
    """
    Class for a MultiLayer Perceptron layer architecture on a temporal sequence. Input tensors expected to be
    of shape (bs x T x xdim), corresponding to latent representation for each temporal subsequence truncated at index i,
    and output tensors expected of shape (bs x T x output_dim).

    MLP Layer stacks standard Perceptron layers as standard, using output for next input. The resulting output is
    of shape (bs x T x output) and returns, for sample (i) at index (j), a representation of dimension output_dim.

    - output_dim          : dimensionality of output space for each sub-sequence. (default = 32)
    - name                : Name on which to save layer
    - hidden_layers       : Number of "hidden" feedforward layers. Total number of layers is hidden_layers + 1.
                            (default = 2)
    - hidden_nodes        : For "hidden" feedforward layers, the dimensionality of the output space. (default = 20)
    - activation_fn       : The activation function to use. (default = 'sigmoid')
    - output_fn           : The activation function on the output of the MLP, e.g. softmax for probabilistic assignments
                            (default = 'softmax').
    - dropout             : dropout rate to be used on layer computation (default = 0.6).

    output: returns sequence of feedforward stacked layers This object implements a call attribute which converts
    inputs of size (bs x T x xdims) to (bs x T x output_dim).    """

    def __init__(self, output_dim, name, hidden_layers = 2, hidden_nodes = 30,
                 activation_fn = 'sigmoid', output_fn = 'softmax', dropout = 0.6):
        super().__init__(name = name)
        self.output_dim       = output_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.activation_fn    = activation_fn
        self.output_fn        = output_fn
        self.dropout          = dropout


    def build(self, input_shape = None):
        # Iterate through hidden layers with TimeDistributed.
        for layer_id_ in range(self.hidden_layers):
            layer_            = TimeDistributed(Dense(units = self.hidden_nodes,
                                      activation = self.activation_fn,
                                      name = self.name))
            setattr(self, 'layer_' + str(layer_id_), layer_)

        self.output_layer     = TimeDistributed(Dense(units = self.output_dim,
                                      activation = self.output_fn,
                                      name = self.name))

    # attribute to call on inputs
    def call(self, inputs, training = True):

        x = inputs

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):
            layer_ = getattr(self, 'layer_' + str(layer_id_))
            x      = layer_(x, training = training)

        y_pred                = self.output_layer(x, training = training)

        return y_pred





L1        = metrics.Mean(name="Critic_loss")
L1_actor  = metrics.Mean(name="Actor_loss")
L2        = metrics.Mean(name="Emb_loss")
L3        = metrics.Mean(name="L2_loss")
L4        = metrics.Mean(name="L3_loss")
AUC       = metrics.AUC(name="AUROC")

class ACTPC(tf.keras.Model):
    """
    Model Class for ACTPC architecture. Input tensors expected to be of shape (bs x times x xdim), and
    output tensors expected of shape (bs x times x num_classes).

    Given input tensors, Encoder network computes latent representation for each truncated subsequence.
    Selector Network computes cluster probability assignments given latent representation, and cluster is sampled
    through a Categorical distribution. Finally, Predictor network uses selected cluster embedding to predict
    cluster outcome.

    params:
        (General)
    - num_clusters        : number of clusters (default = 6)
    - output_dim          : dimensionality of target predicted output (default = 4).
    - y_type              : type of output prediction:
                                - "binary" for binary prediction on each output dimension
                                - "categorical" for multi-class prediction.
                                - "continuous" for continuous prediction on each output dimension.
                            (default = 'categorical')
    - latent_dim          : dimensionality of latent space (default = 32)
    - beta                : L3 weighting in clustering embedding separation. (default = 0.01)
    - alpha               : L2 weighting in cluster entropy. (default = 0.01)
    - seed                : Seed to run analysis on (default = 4347)

        (Embedding Params)
    - embeddings          : Embedding centroids to use on training. If None, KMeans will be used to initialise
                            (default = None).

        (Encoder Params)
    - num_encoder_layers  : Number of "hidden" Encoder layers. (default = 1)
    - num_encoder_nodes   : For hidden Encoder layers, the dimensionality of the cell state. (default = '20')
    - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
    - recurrent_activation: The activation function to use on forget/input/output gates. (default = 'sigmoid')
    - encoder_dropout     : dropout rate to be used on cell state/output computation. (default = 0.6)
    - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0)
    - mask_value          : mask_value to feed to masking layer - time stamps imputed with this mask_value across
                            xdims will be ignored (imputation has taken place) (default = 0.0)
    - encoder_name        : Name on which to save layer (default = 'encoder')

        (Selector Params)
    - selector_name       : Name on which to save Selector Layer. (default = 'selector')
    - num_selector_layers : Number of "hidden" feedforward layers on Selector. (default = 2)
    - num_selector_nodes  : For "hidden" feedforward layers, the dimensionality of the output space. (default = 20)
    - selector_fn         : The activation function to use on Selector. (default = 'sigmoid')
    - selector_output_fn  : The activation function on the output of Selector. (default = 'softmax')
    - selector_dropout    : dropout rate to be used on Selector computation (default = 0.6).

        (Predictor Params)
    - predictor_name      : Name on which to save Predictor Layer. (default = 'predictor')
    - num_predictor_layers: Number of "hidden" feedforward layers on Predictor. (default = 2)
    - num_predictor_nodes : For "hidden" feedforward layers, the dimensionality of the output space. (default = 20)
    - predictor_fn        : The activation function to use on Predictor. (default = 'sigmoid')
    - predictor_output_fn : The activation function on the output of Predictor. Should agree with y_type prediction.
    - predictor_dropout   : dropout rate to be used on Predictor computation (default = 0.6).

    output: This object implements a call attribute which converts inputs of size (bs x T x xdims) to (bs x T x ydims).
    """
    def __init__(self, num_clusters = 6, output_dim = 4, y_type = "categorical",
                 latent_dim = 32, beta = 0.01, alpha = 0.01, seed = 4347, embeddings = None,
                 num_encoder_layers = 1, num_encoder_nodes = 20, state_fn = 'tanh', recurrent_activation = 'sigmoid',
                 encoder_dropout = 0.6, recurrent_dropout = 0.6, mask_value = 0.0, encoder_name = 'encoder',
                 selector_name = 'selector', num_selector_layers = 2, num_selector_nodes = 20, selector_fn = 'sigmoid',
                 selector_output_fn = 'softmax', selector_dropout = 0.6,
                 predictor_name = 'predictor', num_predictor_layers = 2, num_predictor_nodes = 20,
                 predictor_fn = 'sigmoid', predictor_dropout = 0.6):
        super().__init__()

        # General params
        self.K                   = num_clusters
        self.output_dim          = output_dim
        self.y_type              = y_type
        self.latent_dim          = latent_dim
        self.beta                = beta
        self.alpha               = alpha
        self.seed                = seed
        self.embeddings          = embeddings

        # Encoder params
        self.num_encoder_layers  = num_encoder_layers
        self.num_encoder_nodes   = num_encoder_nodes
        self.state_fn            = state_fn
        self.recurrent_activation= recurrent_activation
        self.encoder_dropout     = encoder_dropout
        self.recurrent_dropout   = recurrent_dropout
        self.mask_value          = mask_value

        # Selector Params
        self.num_selector_layers = num_selector_layers
        self.num_selector_nodes  = num_selector_nodes
        self.selector_fn         = selector_fn
        self.selector_output_fn  = selector_output_fn
        self.selector_dropout    = selector_dropout

        # Predictor Params
        self.num_predictor_layers= num_predictor_layers
        self.num_predictor_nodes = num_predictor_nodes
        self.predictor_fn        = predictor_fn
        self.predictor_dropout   = predictor_dropout

        if self.y_type == 'binary':
            self.predictor_output_fn = 'sigmoid'

        elif self.y_type == 'categorical':
            self.predictor_output_fn = 'softmax'

        elif self.y_type == 'categorical':
            self.predictor_output_fn = 'ReLU'

        # Name saving
        self.encoder_name        = encoder_name
        self.selector_name       = selector_name
        self.predictor_name      = predictor_name


    def build(self, input_shape = None):

        # Initialise Layers and Embeddings.
        self.Encoder           = Encoder(intermediate_dim = self.latent_dim,
                                         hidden_layers = self.num_encoder_layers,
                                         hidden_nodes = self.num_encoder_nodes,
                                         state_fn     = self.state_fn,
                                         recurrent_fn = self.recurrent_activation,
                                         dropout      = self.encoder_dropout,
                                         recurrent_dropout = self.recurrent_dropout,
                                         mask_value =  self.mask_value,
                                         name = self.encoder_name)

        self.Predictor         = MLP(output_dim = self.output_dim,
                                     name = self.predictor_name,
                                     hidden_layers = self.num_predictor_layers,
                                     hidden_nodes = self.num_predictor_nodes,
                                     activation_fn = self.predictor_fn,
                                     output_fn = self.predictor_output_fn,
                                     dropout = self.predictor_dropout)

        self.Selector          = MLP(output_dim = self.K,
                                     name = self.selector_name,
                                     hidden_layers = self.num_selector_layers,
                                     hidden_nodes = self.num_selector_nodes,
                                     activation_fn = self.selector_fn,
                                     output_fn = self.selector_output_fn,
                                     dropout = self.selector_dropout)


        # Initialise embeddings as given (if None will be passed onto intiialisation)
        if self.embeddings == None:
            self.custom_emb_init = False
            self.embeddings        = tf.Variable(initial_value = tf.zeros(shape = [self.K, self.latent_dim], dtype = 'float32'),
                                                 trainable = True, name = 'embeddings' )
        else:
            try:
                self.custom_emb_init = True
                self.embeddings = tf.Variable(initial_value = self.embeddings, trainable = True, name = 'embeddings')
            except Exception as e:
                print(e)
                raise ValueError('Embeddings could not be converted to variables.')

        super().build(input_shape)             # Build networks as inherited from Keras model


    def call(self, inputs, training = True):

        # assign x to updatable tensor
        x = inputs

        # Compute cluster probabilistic assignments
        latent_projs  = self.Encoder(x, training = training)
        cluster_probs = self.Selector(latent_projs, training = training)
        cluster_unroll= tf.reshape(cluster_probs, shape = [-1, self.K])

        # Sample cluster embeddings from probabilistic assignment
        cluster_samp  = tf.squeeze(tf.random.categorical(logits = cluster_unroll, num_samples = 1,
                                                         seed = self.seed))
        cluster_emb_unroll   = tf.gather_nd(params = self.embeddings, indices = tf.expand_dims(cluster_samp, -1))
        new_shape     = cluster_probs.get_shape().as_list()[:-1] + [self.latent_dim]
        cluster_emb   = tf.reshape(cluster_emb_unroll, shape = new_shape)

        # Output vector given sampled cluster embedding
        y_pred        = self.Predictor(cluster_emb, training = training)    # Shape bs x T x self.output_dim

        return y_pred



    def network_initialise(self, X, y, optimizer = 'adam', batch_size = 64, init_epochs_ac = 5, init_epochs_pred = 5, **kwargs):
        """
        Initialisation Method for AC-TPC. In order, Encoder-Predictor are trained using latent projections as input to
        the predictor. Secondly, embeddings are computed with KMeans (if not given to the model), and, lastly, selector
        weights are updated.

        Params:
        - X, y: Input Numpy arrays / Tensor arrays of shape (nx, T, xdims), (ny, T, ydims).
        - batch_size: batch size for minibatch sampling. (default = 64)
        - init_epochs_ac: Initialisation epochs for Encoder-Predictor initialisation. (default = 5)
        - init_epohcs_pred: Initialisation epochs for Selector initialisation. (default = 5)
        - kwargs: named arguments to be fed into other functions.

        Returns: Updates Layer Weights and embeddings according to initialisation method.
        """
        data    = tf.data.Dataset.from_tensor_slices((X,y))
        self.optimizer = self._get_optimizer(optimizer)

        print('\n--------------------------------------------')
        print('Initialising Actor-Critic networks.')
        epochs  = init_epochs_ac

        # Iterate through training manually
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch))
            start_time = time.time()

            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size = 5000).batch(batch_size)

            # Iterate through batches and compute gradients.
            for step, (x_train_batch, y_train_batch) in enumerate(data_shuffle):
                with tf.GradientTape(watch_accessed_variables = False) as tape:

                    ac_vars = [var for var in self.weights if 'encoder' in var.name or 'predictor' in var.name]
                    tape.watch(ac_vars)

                    # Compute "Auto Encoder" predicted y
                    latent_projs = self.Encoder(x_train_batch, training = True)
                    y_pred       = self.Predictor(latent_projs, training = True)

                    # Compute loss function
                    loss_value   = aux.predictive_clustering_loss(
                        y_true = y_train_batch,
                        y_pred = y_pred,
                        y_type = self.y_type,
                        name   = 'ac init loss'
                    )
                # Compute gradients and update weights
                gradient = tape.gradient(loss_value, ac_vars)
                self.optimizer.apply_gradients(grads_and_vars=zip(gradient, ac_vars))

                # Log Results every 10 batches
                if step % 10 == 0:
                    print("AC Initialisation Loss for (one batch) at step %d: %.4f" % (step, float(loss_value)))

        print('Time taken for AC init: {:.4f}'.format(time.time() - start_time))
        print('AC-initialisation complete! ')
        print('------------------------------------')




        # Initialise embeddings through closest in cluster space
        print('\n------------------------------------')
        print('Initialising Embeddings')
        start_time = time.time()

        if self.custom_emb_init == False:
            # Compute predicted ys
            latent_projs = self.Encoder(X, training = False)
            y_pred       = self.Predictor(latent_projs, training = False)    # shape (nx, T, ydims)

            y_npy        = y_pred.numpy()
            y_unroll     = y_npy.reshape(-1, self.output_dim)                # shape (nx x T, ydims)

            # Compute KMeans on unrolled predicted output
            init_km = KMeans(n_clusters = self.K, init='k-means++', random_state = self.seed, **kwargs)
            init_km.fit(y_unroll)

            # Obtain cluster centres and sample cluster assignments
            centers_      = init_km.cluster_centers_                         # shape (K, ydims)
            cluster_assign= init_km.predict(y_unroll).reshape(y_npy.shape[:-1])  # shape (nx, T)

            emb_vecs_     = np.zeros(shape = (self.K, self.latent_dim), dtype = 'float32')
            for k in range(self.K):
                centroid_    = centers_[k, :]
                distances_to_centroid_ = np.sum(np.square(np.subtract(y_npy, centroid_)), axis = -1) # shape (nx, T)
                closest_id_ = np.unravel_index(np.argmin(distances_to_centroid_), distances_to_centroid_.shape)

                # assign cluster embedding the corresponding latent projection)
                emb_vecs_[k, :] = latent_projs.numpy()[closest_id_[0], closest_id_[1], :]

            self.embeddings.assign(tf.convert_to_tensor(emb_vecs_, name = 'embeddings', dtype = 'float32'), name = 'embeddings_init')

        else:
            print('Embeddings fed into initialisation. Custom cluster initialisation skipped.')
            # cluster_assign = #Some formula with current embeddings

        # Print Time
        print('Time taken: {:.4f}'.format(time.time() - start_time))
        print('Cluster Embedding complete')


        # Initialise Selector
        print('\n------------------------------------')
        print('Initialising Selector Networks')
        epochs = init_epochs_pred

        # Input cluster_true values in useful format
        cluster_assign = tf.convert_to_tensor(value = cluster_assign, dtype = 'int32', name = 'cluster_assign')
        clus_init_one_hot = tf.one_hot(cluster_assign, depth = self.K, axis = -1)

        # Save X, clus jointly to load correct estimated clusters for each batch
        data = tf.data.Dataset.from_tensor_slices((X, tf.cast(clus_init_one_hot, dtype = 'float32')))

        # Iterate through epochs manually
        for epoch in range(epochs):

            print("\nStart of epoch %d" % (epoch, ))
            start_time = time.time()

            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size=5000).batch(batch_size)

            # Iterate through batches
            for step, (x_train_batch, clus_batch) in enumerate(data_shuffle):
                with tf.GradientTape(watch_accessed_variables = False) as tape:

                    sel_vars = [var for var in self.weights if 'selector' in var.name]
                    tape.watch(sel_vars)

                    # Compute Cluster Assignment probabilities
                    latent_projs = self.Encoder(x_train_batch, training = False)
                    cluster_prob = self.Selector(latent_projs, training = True)

                    # Compute loss function (Cross-Entropy with cluster assignment as true value)
                    loss_value   = aux.selector_init_loss(
                        y_prob  = cluster_prob,
                        clusters= clus_batch,
                        name    = 'pred_clus_loss'
                    )
                # Compute gradients and update weights
                gradient = tape.gradient(loss_value, sel_vars)
                self.optimizer.apply_gradients(zip(gradient, sel_vars))

                # Log Results every 10 batches
                if step % 10 == 0:
                    print("Selector Init Loss at step %d: %.4f" % (step, float(loss_value)))


        print('Time taken: {:.4f}'.format(time.time() - start_time))
        print('Selector initialisation complete!')
        print('----------------------------------')




    # Main training step iterative update
    def train_step(self, inputs, training = True):
        X, y = inputs

        "Update Critic"
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            critic_vars = [var for var in self.trainable_weights if 'predictor' in var.name]
            # tape.watch(critic_vars + [cluster_emb])
            tape.watch(critic_vars)

            # Forward pass
            y_pred = self.call(X, training = True)

            # Compute loss
            loss_critic = aux.predictive_clustering_loss(y_true = y, y_pred = y_pred,
                y_type = self.y_type, name   = 'pred_clus_loss'
            )

        critic_grad = tape.gradient(target = loss_critic, sources = critic_vars)
        self.optimizer.apply_gradients(zip(critic_grad, critic_vars))


        "Update Actor - probabilistic assignments remain the same as before"
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Variables to consider
            actor_vars = [var for var in self.trainable_weights if 'encoder' in var.name or 'selector' in var.name]
            tape.watch(actor_vars)

            # Compute Forward pass variables
            latent_projs   = self.Encoder(X, training = True)
            cluster_probs  = self.Selector(latent_projs, training = True)

            # Unroll Assignments to Sample cluster and compute corresponding embedding
            cluster_unroll     = tf.reshape(cluster_probs, shape = [-1, self.K])
            cluster_samp       = tf.squeeze(tf.random.categorical(logits = cluster_unroll, num_samples = 1,
                                                                  seed = self.seed, dtype = 'int32'))
            cluster_emb_unroll = tf.gather_nd(params = self.embeddings, indices = tf.expand_dims(cluster_samp, -1))
            new_shape          = cluster_probs.get_shape().as_list()[:-1] + [self.latent_dim]
            cluster_emb        = tf.reshape(cluster_emb_unroll, shape = new_shape)

            # Compute output vector and probability for corresponding cluster given sampled cluster embedding
            idx_slices_        = tf.stack((tf.range(cluster_unroll.get_shape()[0]), cluster_samp), axis = 1)
            pi_clus_assign_unroll = tf.gather_nd(params=cluster_unroll, indices=idx_slices_)
            pi_clus_assign     = tf.reshape(pi_clus_assign_unroll, shape = cluster_probs.get_shape()[:-1])
            y_pred             =  self.Predictor(cluster_emb, training = training)

            # Compute L1 loss weighted by cluster confidence
            weighted_loss_actor_1 = aux.actor_predictive_clustering_loss(
                y_true = y, y_pred = y_pred, cluster_assignment_probs = pi_clus_assign,
                y_type = self.y_type, name = 'actor_pred_clus_loss'
            )

            # Compute Cluster Entropy loss
            entr_loss  = aux.cluster_probability_entropy_loss(
                y_prob = cluster_probs,
                name   = 'clust_entropy_loss'
            )

            loss_actor = weighted_loss_actor_1 + self.alpha * entr_loss

        # Compute gradients and update
        actor_grad = tape.gradient(target=loss_actor, sources = actor_vars)
        self.optimizer.apply_gradients(zip(actor_grad, actor_vars))


        "Update embeddings"
        embedding_vars = self.embeddings
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(embedding_vars)

            # Compute cluster_assignments and pred_y
            cluster_probs = self.Selector(self.Encoder(X, training=False), training=False)
            cluster_unroll     = tf.reshape(cluster_probs, shape = [-1, self.K])
            cluster_samp       = tf.squeeze(tf.random.categorical(logits = cluster_unroll, num_samples = 1,
                                                                  seed = self.seed))

            mask_emb           = tf.one_hot(cluster_samp, depth = self.K)
            cluster_emb_unroll = tf.linalg.matmul(
                a = mask_emb, b= embedding_vars
            )

            new_shape  = cluster_probs.get_shape().as_list()[:-1] + [self.latent_dim]
            cluster_emb= tf.reshape(cluster_emb_unroll, shape = new_shape)
            y_pred     = self.Predictor(cluster_emb, training=False)

            # Compute embedding separation loss and predictive clustering loss
            emb_loss_1 = aux.predictive_clustering_loss(
                y_true = y,
                y_pred = y_pred,
                y_type = self.y_type,
                name   = 'emb_pred_clus_loss'
            )
            emb_loss_2 = aux.embedding_separation_loss(
                y_embeddings = embedding_vars,
                name   = 'emb_sep_loss'
            )

            loss_emb   = emb_loss_1 + self.beta * emb_loss_2

        # Compute gradients and update
        emb_grad  =  tape.gradient(target=loss_emb, sources=embedding_vars)
        self.optimizer.apply_gradients(zip([emb_grad], [embedding_vars]))
        self.embeddings = embedding_vars


        # Update Loss functions
        L1.update_state(loss_critic)
        L1_actor.update_state(loss_actor)
        L2.update_state(loss_emb)
        L3.update_state(entr_loss)
        L4.update_state(emb_loss_2)

        return {'Critic_loss': L1.result(), 'actor_loss': L1_actor.result(),
                'Emb_loss': L2.result(), 'L2_loss': L3.result(), 'L3_loss': L4.result()}


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


















