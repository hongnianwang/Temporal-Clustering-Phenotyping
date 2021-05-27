#!/usr/bin/env python3

"""
Model script for ACTPC implementation in tensorflow-gpu.

Code for the model can be found:

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk

"""

# Loading libraries
import numpy as np
import tensorflow as tf

import time

from tensorflow.keras.layers import LSTM, Masking, Dense, TimeDistributed
from tensorflow.keras import layers
from tensorflow.keras import metrics

from sklearn.cluster import KMeans

# Auxiliary function
import utils_model as net_utils


# Loss tracking
L1_cri    = metrics.Mean(name = "LCri")
L1_act    = metrics.Mean(name = "LAct")     # L1 + alpha x entropy
L_emb     = metrics.Mean(name = "LEmb")     # L1 + beta x embedding sep
L2        = metrics.Mean(name = "L2")       # entropy sep
L3        = metrics.Mean(name = "L3")       # embedding sep


# Validation loss tracking
val_L1_cri= metrics.Mean(name = "val_LCri")
val_L1_act= metrics.Mean(name = 'val_LAct')
val_L_emb = metrics.Mean(name = 'val_LEmb')
val_L2    = metrics.Mean(name = 'val_L2')
val_L3    = metrics.Mean(name = 'val_L3')


# Metrics tracking
clus_KL_sep = metrics.Mean(name = 'clus_phen_sep')
cat_acc   = metrics.CategoricalAccuracy(name = 'Cat_acc')
auroc     = metrics.AUC(name = "auc")


# Define auxiliary classes 
class Encoder(layers.Layer):
    
    """
    Class for an Encoder layer architecture. Input tensors expected to be of shape (bs x times x xdim), and
    output tensors expected of shape (bs x times x latent dim).

    Encoder Layer stacks LSTMs layers using  whole output sequences - the overall output is of size (bs x time_steps x latent dim), with [i, j, :] representing the latent vector from time-subsequence corresponding to patient i and from times t=0, to t=j.
    
    Params:

    - intermediate_dim    : dimensionality of latent space for each sub-sequence. (default = 32)
    - hidden_layers       : Number of "hidden" LSTM layers. Total number of stacked layers is hidden_layers + 1 (default = 1)
    - hidden_nodes        : For "hidden" LSTM layers, the dimensionality of the intermediate tensors/state. (default = '20')
    - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
    - recurrent_activation: The activation function to use on forget/input/output gates. (default = 'sigmoid')
    - dropout             : dropout rate to be used on cell state/output computation. (default = 0.6)
    - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0)
    - mask_value          : mask_value to feed to masking layer (checks across last dimensions of x). (default = 0.0)
    - name                : Name on which to save component. (default = 'encoder')

    Call:
            inputs - tensor of shape (batch_size, max_length_time_steps, x_dim)
            outputs- tensor of shape (batch_size, max_length_time_steps, intermediate_dim).
            
            Output at index [i,j,:] represents the latent representation of patient i given subsequence from time 0 to time j.
    """
    def __init__(self, intermediate_dim=32, hidden_layers=1, hidden_nodes=20, state_fn="tanh", recurrent_fn="sigmoid",
                 dropout=0.7, recurrent_dropout=0, mask_value=0.0, name='encoder'):
        
        super().__init__(name =name)
        self.intermediate_dim = intermediate_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.state_fn         = state_fn
        self.recurrent_fn     = recurrent_fn
        self.dropout          = dropout
        self.recurrent_dropout= recurrent_dropout
        self.mask_value       = mask_value


    # Build for tensorflow model
    def build(self, input_shape = None):
        
        # Add intermediate layers,
        for layer_id_ in range(self.hidden_layers):
            
            # Each intermediate sequence returns a time sequence of representations.
            layer_            = LSTM(units = self.hidden_nodes, return_sequences = True, activation = self.state_fn,
                recurrent_activation = self.recurrent_fn, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout,
                return_state = False, name = self.name)
            
            # Set attribute for class
            setattr(self, 'layer_' + str(layer_id_), layer_)

        # Add output layer - output is the latent representation for each sub-sequence.
        self.output_layer     = LSTM(units = self.intermediate_dim, activation = self.state_fn,
                recurrent_activation = self.recurrent_fn, return_sequences = True, dropout = self.dropout,     
                recurrent_dropout = self.recurrent_dropout, return_state = False, name = self.name)
        
        
    # call attribute is used to evaluate the layer.
    def call(self, inputs, training = True):

        x = inputs

        # Apply Masking layer.
        if self.mask_value is not None:
            try:
                # Mask layer ignores time stamps with all entries given by mask_value - propagated by default.
                x = Masking(mask_value = self.mask_value)(inputs)

            except Exception as e:
                print(e)
                raise ValueError('Masking did not work!')

        # Iterate through hidden layers
        for layer_id_ in range(self.hidden_layers):
            layer_   = getattr(self, 'layer_' + str(layer_id_))
            x        = layer_(x, training = training)
            
        # Output layer.
        latent_rep  = self.output_layer(x, training = training)   # shape (bs, T, intermediate_dim)

        return latent_rep


    # Configuration file for layer
    def get_config(self):
        config = super().get_config()
        
        # Update configuration
        config.update({"dropout": self.dropout, 
                       "intermediate_dim": self.intermediate_dim})
        
        return config
    
    

class MLP(layers.Layer):
    
    """
    Class for a MultiLayer Perceptron layer architecture on a temporal sequence. MLP Layer stacks standard Perceptron, single layers.
    
    Params:

    - output_dim          : dimensionality of output space for each sub-sequence. (default = 32)
    - hidden_layers       : Number of "hidden" feedforward layers. Total number of layers is hidden_layers + 1. (default = 2)
    - hidden_nodes        : For "hidden" feedforward layers, the dimensionality of the output space. (default = 30)
    - activation_fn       : The activation function to use. (default = 'sigmoid')
    - output_fn           : The activation function on the output of the MLP. (default = 'softmax').
    - dropout             : dropout rate to be used on layer computation (default = 0.6).
    - name                : name on which to save layer. (defult = 'decoder')
    
    Call:
        - inputs: tensor of shape (batch_size, max_time_steps, xdim)
        - outputs: tensorf of shape (batch_size, max_time_steps, output_dim)
        
        MLP layer applied to each x_dim size vector at each time-step.  
    """
    def __init__(self, output_dim, hidden_layers = 2, hidden_nodes = 30, activation_fn = 'sigmoid', 
                 output_fn = 'softmax', dropout = 0.6, name = 'decoder'):
        
        super().__init__(name = name)
        self.output_dim       = output_dim
        self.hidden_layers    = hidden_layers
        self.hidden_nodes     = hidden_nodes
        self.activation_fn    = activation_fn
        self.output_fn        = output_fn
        self.dropout          = dropout


    # Build attribute for tensorflow model
    def build(self, input_shape = None):
        
        # Iterate through hidden layers with TimeDistributed layer.
        for layer_id_ in range(self.hidden_layers):
            layer_            = TimeDistributed(Dense(units = self.hidden_nodes, activation = self.activation_fn,
                                      name = self.name))
            setattr(self, 'layer_' + str(layer_id_), layer_)

        # Final layer
        self.output_layer     = TimeDistributed(Dense(units = self.output_dim, activation = self.output_fn,
                                      name = self.name))


    # attribute to call on inputs
    def call(self, inputs, training = True):

        x = inputs

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):
            layer_ = getattr(self, 'layer_' + str(layer_id_))
            x      = layer_(x, training = training)

        # Output y
        y          = self.output_layer(x, training = training)

        return y


    # Configuration for layer
    def get_config(self):
        config    = super().get_config()
        
        # Update
        config.update({"out_dim": self.output_dim,
                       "dropout": self.dropout})
        
        return config





class ACTPC(tf.keras.Model):
    
    """
    Model Class for ACTPC architecture. Model has 3 components: Encoder, Selector and Predictor. Separately, there are cluster 
    embeddings representing clusters.
    
    Encoder computes latent embeddings for each truncated sample subsequence (from the start to each timestep).
    Selector samples a cluster assignment from the latent embeddings.
    Predictor predicts phenotypes from sampled cluster.
    
    Model implements initialisation in order via: Encoder - Predictor, Selector Entropy minimisation and Embedding separation.

    Inputs - batch tensor of shape (batch size, max_time_steps, x_dim)
    Output - a tuple consisting of sampled cluster assignments, probability assignments and predicted output.

    Params:
        
        (General)
    - num_clusters        : number of clusters (default = 6)
    - output_dim          : dimensionality of target predicted output (default = 4).
    - y_type              : type of output prediction ("binary", "categorical" or "continuous"). (default = 'categorical')
    - latent_dim          : dimensionality of latent space (default = 32)
    - beta                : L3 weighting in clustering embedding separation. (default = 0.01)
    - alpha               : L2 weighting in cluster entropy. (default = 0.01)
    - seed                : Seed to run analysis on (default = 4347)

        (Embedding Params)
    - embeddings          : Embedding centroids to use on training. If None, KMeans will be used to initialise. (default = None).

        (Encoder Params)
    - num_encoder_layers  : Number of "hidden" Encoder layers. (default = 1)
    - num_encoder_nodes   : For hidden Encoder layers, the dimensionality of the cell state. (default = 20)
    - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
    - recurrent_activation: The activation function to use on forget/input/output gates. (default = 'sigmoid')
    - encoder_dropout     : dropout rate to be used on cell state/output computation. (default = 0.6)
    - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0)
    - mask_value          : mask_value to feed to masking layer (default = 0.0)
    - encoder_name        : Name on which to save layer (default = 'encoder')

        (Selector Params)
    - num_selector_layers : Number of "hidden" feedforward layers on Selector. (default = 2)
    - num_selector_nodes  : For "hidden" feedforward layers, the dimensionality of the output space. (default = 20)
    - selector_fn         : The activation function to use on Selector. (default = 'sigmoid')
    - selector_output_fn  : The activation function on the output of Selector. (default = 'softmax')
    - selector_dropout    : dropout rate to be used on Selector computation. (default = 0.6)
    - selector_name       : Name on which to save Selector Layer. (default = 'selector')

        (Predictor Params)
    - num_predictor_layers: Number of "hidden" feedforward layers on Predictor. (default = 2)
    - num_predictor_nodes : For "hidden" feedforward layers, the dimensionality of the output space. (default = 20)
    - predictor_fn        : The activation function to use on Predictor. (default = 'sigmoid')
    - predictor_output_fn : The activation function on the output of Predictor. (default = depends on y_type)
    - predictor_dropout   : dropout rate to be used on Predictor computation (default = 0.6).
    - predictor_name      : Name on which to save Predictor Layer. (default = 'predictor')
    """ 
    def __init__(self, num_clusters = 12, output_dim = 4, y_type = "categorical",
         latent_dim = 32, beta = 0.01, alpha = 0.01, seed = 4347, embeddings = None, num_encoder_layers = 2, num_encoder_nodes = 32, state_fn = 'tanh', recurrent_activation = 'sigmoid', encoder_dropout = 0.6, recurrent_dropout = 0, mask_value = 0.0,  encoder_name = 'encoder', selector_name = 'selector', num_selector_layers = 2, num_selector_nodes = 32, selector_fn = 'sigmoid', selector_output_fn = 'softmax', selector_dropout = 0.6, predictor_name = 'predictor', num_predictor_layers = 2, num_predictor_nodes = 32, predictor_fn = 'sigmoid', predictor_dropout = 0.6):
        
        super().__init__()

        # General params
        self.K                   = num_clusters
        self.output_dim          = output_dim
        self.y_type              = y_type
        self.latent_dim          = latent_dim
        self.beta                = beta
        self.alpha               = alpha
        self.seed                = seed
        
        # Embedding params
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

    
    # Build method to initialise model
    def build(self, embeddings = None , input_shape = None):

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


        # Check if embeddings have been given or need to be initialise
        if embeddings is None:
            
            self.custom_embedding_init = False
            self.embeddings        = tf.Variable(initial_value = tf.zeros(shape = [self.K, self.latent_dim], dtype = 'float32'),
                                                  trainable = True, name = 'embeddings' )
        else:
            try:
                self.custom_embedding_init = True
                self.embeddings    = tf.Variable(initial_value = embeddings, trainable = True, name = 'embeddings')
                
            except Exception as e:
                print(e)
                raise ValueError('Embeddings could not be converted to variables.')
                

        super().build(input_shape)             # Build networks as inherited from Keras model


    # Call method - defines model computation logic
    def call(self, inputs, training = True):

        # assign x to updatable tensor
        x = inputs

        # Compute cluster probabilistic assignments
        latent_projs  = self.Encoder(x, training = training)
        cluster_probs = self.Selector(latent_projs, training = training)

        # Sample cluster embedding and corresponding probability given probability assignments
        cluster_sel , cluster_prob = self.sample_cluster_pis(cluster_probs)

        # Output vector given sampled cluster embedding
        y_pred       = self.Predictor(cluster_sel, training = training)    # Shape bs x T x self.output_dim

        return y_pred


    # Define initialisation procedure for weights
    def init_params(self, X, y, optimizer = 'adam', batch_size = 64, init_epochs_ac = 5, init_epochs_pred = 5, **kwargs):
        
        """
        Initialisation Method for AC-TPC:
            1 - X - Encoder - Predictor - y initialise
            2 - initialise Selector based on output of Selector (maximise entropy)
            3 - initialise embeddings based on K-Means on the latent space, if embeddings have not been given.

        Params:
        - X, y: Input Numpy arrays / Tensor arrays of shape (nx, T, xdims), (ny, T, ydims).
        - batch_size: batch size for minibatch sampling. (default = 64)
        - init_epochs_ac: Initialisation epochs for Encoder-Predictor initialisation. (default = 5)
        - init_epohcs_pred: Initialisation epochs for Selector initialisation. (default = 5)
        - kwargs: named arguments to be fed into other functions.

        Returns: Updates Layer Weights and embeddings according to initialisation method.
        """
        print('\n--------------------------------------------')
        print('Initialising Actor-Critic networks')
              
        # Load data 
        data           = tf.data.Dataset.from_tensor_slices((X,y))
        epochs  = init_epochs_ac
        
        # Initialise optimizer
        if type(optimizer) == str:
            self.optimizer = self._get_optimizer(optimizer)
        else:
            self.optimizer = optimizer

        # Initialise epoich loss tracker
        avg = 0
        
        # Iterate through training manually
        for epoch in range(epochs):
            
            if epoch % 10 == 0:
                print("\nStart of epoch %d" % (epoch, ))
            
            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size = 5000).batch(batch_size)
            
            # Initialise tracker
            epoch_loss = 0

            # Iterate through batches and compute gradients.
            for step, (x_train_batch, y_train_batch) in enumerate(data_shuffle):
                with tf.GradientTape(watch_accessed_variables = False) as tape:
                        
                    # variables to watch and compute gradients for
                    ac_vars = [var for var in self.weights if 'encoder' in var.name or 'predictor' in var.name]
                    tape.watch(ac_vars)

                    # Compute "Auto Encoder" predicted y
                    latent_projs = self.Encoder(x_train_batch, training = True)
                    y_pred       = self.Predictor(latent_projs, training = True)

                    # Compute loss function
                    loss_value   = net_utils.predictive_clustering_loss(
                        y_true = y_train_batch,
                        y_pred = y_pred,
                        y_type = self.y_type,
                        name   = 'ac init loss'
                    )
                    
                # Compute gradients and update weights
                gradient = tape.gradient(loss_value, ac_vars)
                self.optimizer.apply_gradients(grads_and_vars=zip(gradient, ac_vars))
                
                epoch_loss += loss_value.numpy()
                
            # Update loss tracker
            avg += epoch_loss/step
                
            # Log Results every 10 epochs
            if epoch % 10 == 0:
                print("AC Initialisation Loss Average at epoch %d: %.4f" % (epoch, float(avg/10)))
                avg = 0


        print('AC-initialisation complete! ')


        # Initialise embeddings through closest in cluster space
        print('\n------------------------------------')
        print('Initialising Embeddings')
        start_time = time.time()

        if self.custom_embedding_init == False:
            
            # Compute predicted ys
            latent_projs = self.Encoder(X, training = False)
            y_pred       = self.Predictor(latent_projs, training = False)    

            # Collapse the time axis
            y_npy        = y_pred.numpy()
            y_unroll     = y_npy.reshape(-1, self.output_dim)                # shape (nx x T, ydims)

            # Compute KMeans on unrolled predicted output
            init_km = KMeans(n_clusters = self.K, init='k-means++', random_state = self.seed, **kwargs)
            init_km.fit(y_unroll)

            # Obtain cluster centres and sample cluster assignments
            centers_      = init_km.cluster_centers_                             # shape (K, ydims)
            cluster_assign= init_km.predict(y_unroll).reshape(y_npy.shape[:-1])  # shape (nx, T)

            # Compute embedding vectors as closest latent to cluster centroid
            emb_vecs_     = np.zeros(shape = (self.K, self.latent_dim), dtype = 'float32')
            for k in range(self.K):
                
                # Compute relevant points (assigned to cluster, only)
                ys_in_cluster    = y_npy[cluster_assign == k, :]
                centroid = centers_[k, :]
                
                # Compute closest point
                distances_to_centroid_ = np.sum(np.square(np.subtract(ys_in_cluster, centroid)), axis = -1) # shape (nx, T)
                closest_id_ = np.argmin(distances_to_centroid_)
                
                # latent
                latent_closest_id_ = latent_projs.numpy().reshape(-1, self.latent_dim)[closest_id_, :]
                
                # assign cluster embedding the corresponding latent projection)
                emb_vecs_[k, :] = latent_closest_id_

                # # Consider only those assigned to cluster k
                # latents_in_cluster = latent_projs.numpy()[cluster_assign == k, :]
                
                # # Compute mean
                # mean_latents = np.mean(np.mean(latents_in_cluster, axis = 1), axis = 0)

                # # assign cluster embedding the corresponding latent projection)
                # emb_vecs_[k, :] = mean_latents

            # self.embeddings.assign(tf.convert_to_tensor(np.arctanh(emb_vecs_), name = 'embeddings', dtype = 'float32'), name = 'embeddings_init')
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
        
        # Initialise loss tracker
        avg_loss = 0
        
        # Iterate through epochs manually
        for epoch in range(epochs):

            if epoch % 10 ==0:
                print("\nStart of epoch %d" % (epoch, ))
                
                
            start_time = time.time()

            # Shuffle data each epoch and generate batch enumerator
            data_shuffle = data.shuffle(buffer_size=5000).batch(batch_size)
            
            # loss for epoch
            epoch_loss = 0

            # Iterate through batches
            for step, (x_train_batch, clus_batch) in enumerate(data_shuffle):
                with tf.GradientTape(watch_accessed_variables = False) as tape:

                    sel_vars = [var for var in self.weights if 'selector' in var.name]
                    tape.watch(sel_vars)

                    # Compute Cluster Assignment probabilities
                    latent_projs = self.Encoder(x_train_batch, training = False)
                    cluster_prob = self.Selector(latent_projs, training = True)

                    # Compute loss function (Cross-Entropy with cluster assignment as true value)
                    loss_value   = net_utils.selector_init_loss(
                        y_prob  = cluster_prob,
                        clusters= clus_batch,
                        name    = 'pred_clus_loss'
                    )
                # Compute gradients and update weights
                gradient = tape.gradient(loss_value, sel_vars)
                self.optimizer.apply_gradients(zip(gradient, sel_vars))
                
                epoch_loss += loss_value.numpy()

            avg += epoch_loss /step
                
            # Log Results every 10 epochs
            if epoch % 10 == 0:
                print("Selector Init Loss Avg at Epoch %d: %.4f" % (epoch, float(avg/10)))
                avg = 0


        print('Time taken: {:.4f}'.format(time.time() - start_time))
        print('Selector initialisation complete!')
        print('----------------------------------')



    # Main training step iterative update
    def train_step(self, inputs, training = True):
        X, y = inputs


        # Update Critic
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            
            # Select vars
            critic_vars = [var for var in self.trainable_weights if 'predictor' in var.name]
            tape.watch(critic_vars)

            # Compute steps
            latent_projs  = self.Encoder(X, training = False)
            cluster_probs = self.Selector(latent_projs, training = False)
            
            # Flatten and sample cluster
            pis_unroll_  = tf.reshape(cluster_probs, shape = [-1, self.K])
            cluster_samp = tf.squeeze((tf.random.categorical(logits = pis_unroll_, num_samples = 1,
                                                            seed = self.seed)))
            
            # Convert to one_hot encoding
            mask_emb = tf.one_hot(cluster_samp, depth = self.K)
    
            # Matrix multiplication returns unrolled data with corresponding embedding value
            cluster_emb_unroll = tf.linalg.matmul(
                a = mask_emb, b = self.embeddings)
            
            # Return to temporal shape
            new_shape   = [-1, cluster_probs.get_shape()[1]] + [self.latent_dim]
            cluster_emb = tf.reshape(cluster_emb_unroll, shape = new_shape) 

    
            # Output vector given sampled cluster embedding
            y_pred       = self.Predictor(cluster_emb, training = training)    
        
            # Compute loss
            loss_critic = net_utils.predictive_clustering_loss(y_true = y, y_pred = y_pred,
                y_type = self.y_type, name   = 'pred_clus_loss')

            
        # Compute gradient
        critic_grad = tape.gradient(target = loss_critic, sources = critic_vars)
        self.optimizer.apply_gradients(zip(critic_grad, critic_vars))


        # Update Actor
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            
            # Select vars
            actor_vars = [var for var in self.trainable_weights if 'encoder' in var.name or 'selector' in var.name]
            tape.watch(actor_vars)

            # Compute Forward pass variables
            latent_projs   = self.Encoder(X, training = True)
            cluster_probs  = self.Selector(latent_projs, training = True)

            # Unroll Assignments to Sample cluster and compute corresponding embedding
            cluster_emb, cluster_pis = self.sample_cluster_pis(cluster_probs)

            # Compute output vector and probability for corresponding cluster given sampled cluster embedding
            y_pred         =  self.Predictor(cluster_emb, training = False)

            # Compute L1 loss weighted by cluster confidence
            weighted_loss_actor_1 = net_utils.actor_predictive_clustering_loss(
                y_true = y, y_pred = y_pred, cluster_assignment_probs = cluster_pis,
                y_type = self.y_type, name = 'actor_pred_clus_loss')

            # Compute Cluster Entropy loss
            entr_loss  = net_utils.cluster_probability_entropy_loss(
                y_prob = cluster_probs,
                name   = 'clust_entropy_loss')

            loss_actor = weighted_loss_actor_1 + self.alpha * entr_loss
            
        # Compute gradients and update
        actor_grad = tape.gradient(target=loss_actor, sources = actor_vars)
        self.optimizer.apply_gradients(zip(actor_grad, actor_vars))


        #Update embeddings
        embedding_vars = self.embeddings
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            
            # Watch embedding variables
            tape.watch(embedding_vars)

            # Compute cluster_assignments and pred_y
            cluster_probs      = self.Selector(self.Encoder(X, training=False), training=False)
            
            # Flatten and sample cluster
            pis_unroll_  = tf.reshape(cluster_probs, shape = [-1, self.K])
            cluster_samp = tf.squeeze((tf.random.categorical(logits = pis_unroll_, num_samples = 1,
                                                            seed = self.seed)))
            
            # Convert to one_hot encoding
            mask_emb = tf.one_hot(cluster_samp, depth = self.K)
    
            # Matrix multiplication returns unrolled data with corresponding embedding value
            cluster_emb_unroll = tf.linalg.matmul(
                a = mask_emb, b = embedding_vars)
            
            # Return to temporal shape
            new_shape   = [-1, cluster_probs.get_shape()[1]] + [self.latent_dim]
            cluster_emb = tf.reshape(cluster_emb_unroll, shape = new_shape) 
        
            # Predict y
            y_pred     = self.Predictor(cluster_emb, training=False)
            
            # Compute cluster phenotypes
            y_clus     = tf.squeeze(self.Predictor(
                        tf.expand_dims(embedding_vars, axis = 0), 
                        training = False))

            # Compute embedding separation loss and predictive clustering loss
            emb_loss_1 = net_utils.predictive_clustering_loss(
                y_true = y, y_pred = y_pred,
                y_type = self.y_type, name   = 'emb_pred_clus_loss')
            
            # emb_loss_2 = net_utils.euclidean_separation_loss(
            #     y_clusters = tf.squeeze(y_clus), name   = 'emb_sep_loss')
            emb_loss_2 = net_utils.euclidean_separation_loss(
                y_clusters = embedding_vars, name   = 'emb_sep_loss')
            
            emb_loss_3 = net_utils.KL_separation_loss(
                y_clusters = y_clus, name = "KL_emb_loss")

            loss_emb   = emb_loss_1 + self.beta * emb_loss_2
            # loss_emb   = emb_loss_1 + self.beta * emb_loss_3

        # Compute gradients and update
        emb_grad  =  tape.gradient(target = loss_emb, sources=embedding_vars)
        self.optimizer.apply_gradients(zip([emb_grad], [embedding_vars]))
        self.embeddings = embedding_vars


        # Update Loss functions
        L1_cri.update_state(loss_critic)
        L1_act.update_state(loss_actor)
        L_emb.update_state(loss_emb)
        L2.update_state(entr_loss)
        L3.update_state(emb_loss_2)

        return {'L1': L1_cri.result(), 'L2': L1_act.result(), 'L3': L_emb.result(), "Ent": L2.result(), "Sep": L3.result()}
                # 'entropy': L2.result(), 'emb_sep': L3.result()}

    
    # Evaluation step for Validation or Evaluation
    def test_step(self, inputs):
        
        X, y = inputs

        # Compute Forward pass variables
        latent_projs = self.Encoder(X, training = False)
        cluster_probs = self.Selector(latent_projs, training = False)
        
        # Sample cluster and compute predicted phenotypes
        cluster_emb, cluster_pis = self.sample_cluster_pis(cluster_probs)
        y_pred = self.Predictor(cluster_emb, training = False)

        
        # Compute output of cluster embeddings
        y_clusters = self.compute_y_clusters()
        
        # Compute unrolled y_preds
        # y_unroll_ = tf.reshape(tf.repeat(tf.expand_dims(y, axis = 1), repeats = y_pred.get_shape()[1],
        #                        axis = 1), [-1, self.output_dim])
        y_unroll_        = tf.reshape(y, [-1, self.output_dim])
        y_pred_unroll_   = tf.reshape(y_pred, [-1, self.output_dim])       
        

        # Compute Loss Functions
        pred_clus_loss   = net_utils.predictive_clustering_loss(y_true = y, y_pred = y_pred, 
                                                                y_type = self.y_type)
        entr_loss        = net_utils.cluster_probability_entropy_loss(y_prob = cluster_probs)
        clus_sep_loss    = net_utils.euclidean_separation_loss(y_clusters = self.embeddings)
        weig_pred_loss   = net_utils.actor_predictive_clustering_loss(y_true = y, y_pred = y_pred,
                                        cluster_assignment_probs = cluster_pis, y_type = self.y_type)
        KL_sep_loss      = net_utils.KL_separation_loss(y_clusters)

        critic_loss      = pred_clus_loss
        actor_loss       = weig_pred_loss + self.alpha * entr_loss
        Emb_loss         = pred_clus_loss + self.beta * clus_sep_loss
        

        # Update Validation losses
        val_L1_cri.update_state(critic_loss)
        val_L1_act.update_state(actor_loss)
        val_L_emb.update_state(Emb_loss)
        val_L2.update_state(entr_loss)
        val_L3.update_state(clus_sep_loss)

        # Update Relevant Metrics
        # clus_KL_sep.update_state(KL_sep_loss)
        # auroc.update_state(y_true = y_unroll_, y_pred = y_pred_unroll_)
        cat_acc.update_state(y_true = y_unroll_, y_pred = y_pred_unroll_)

        return {'L1': val_L1_cri.result(), 'L2': val_L1_act.result(),
                'L3': val_L_emb.result(), 'pis': val_L2.result(), 'Sep': val_L3.result(),
                'KL':  KL_sep_loss,  "ACC": cat_acc.result()} #"AUC": auroc.result(),



    def get_config(self):
        
        # Save variables
        save_dic = {'Clusters': self.K,
                    'y_dims': self.output_dim,
                    'y_type': self.y_type,
                    'latent_dim': self.latent_dim,
                    'beta': self.beta,
                    'alpha': self.alpha,
                    'seed': self.seed,
                    'custom_embedding_init': self.custom_embedding_init,
                    'num_encoder_layers': self.num_encoder_layers,
                    'num_encoder_nodes': self.num_encoder_nodes,
                    'state_fn': self.state_fn,
                    'recurrent_activation': self.recurrent_activation,
                    'encoder_dropout': self.encoder_dropout,
                    'recurrent_dropout': self.recurrent_dropout,
                    'mask_value': self.mask_value,
                    'num_selector_layers': self.num_selector_layers,
                    'num_selector_nodes': self.num_selector_nodes,
                    'selector_fn': self.selector_fn,
                    'selector_output_fn': self.selector_output_fn,
                    'selector_dropout': self.selector_dropout,
                    'num_predictor_layers': self.num_predictor_layers,
                    'num_predictor_nodes': self.num_predictor_nodes,
                    'predictor_fn': self.predictor_fn,
                    'predictor_output_fn': self.predictor_output_fn,
                    'predictor_dropout': self.predictor_dropout,
                    'encoder_name': self.encoder_name,
                    'selector_name': self.selector_name,
                    'predictor_name': self.predictor_name
                    }

        return save_dic


    def sample_cluster_pis(self, cluster_probs):
        
        """
        Auxiliary function to sample with differentiation.

        Inputs
                cluster_probs:  of shape (bs , T, num_clusters)
                embeddings: of shape (num_clusters, latent_dim)
                
        returns: A sampled embedding given the corresponding probabilistic assignment of shape (bs, T, latent_dim)
        Also return corresponding probability value of shape (bs, T)
        """
        # Unroll Cluster to 2D format
        pis_unroll_  = tf.reshape(cluster_probs, shape = [-1, self.K])
        
        # Sample given probabilistic assignment to each sample, time pair (i,j)
        cluster_samp = tf.squeeze((tf.random.categorical(logits = pis_unroll_, num_samples = 1,
                                                        seed = self.seed)))
        
        # Convert to one_hot encoding
        mask_emb = tf.one_hot(cluster_samp, depth = self.K)

        # Matrix multiplication returns unrolled data with corresponding embedding value
        cluster_emb_unroll = tf.linalg.matmul(
            a = mask_emb, b = self.embeddings)

        # Return to temporal shape
        old_shape   = [-1, cluster_probs.get_shape()[1]]
        new_shape   = old_shape + [self.latent_dim]
        cluster_emb = tf.reshape(cluster_emb_unroll, shape = new_shape)

        # Compute corresponding probability assignment for sampled cluster
        if not (None in cluster_probs.get_shape().as_list()):
            idx_slices_      = tf.stack(
                                (tf.range(pis_unroll_.get_shape()[0]),
                                 tf.cast(cluster_samp, dtype = 'int32')), axis = 1)   # 2D indices (k,pred_cluster)
            
        else:
            idx_slices_      = tf.expand_dims(tf.cast(cluster_samp, dtype ='int32'), axis = 1)

        # Convert to temporal format
        clus_pi_unroll_  = tf.gather_nd(params = pis_unroll_, indices = idx_slices_)   #
        clus_pi          = tf.reshape(clus_pi_unroll_, shape = [-1, cluster_probs.get_shape()[1]])


        return cluster_emb, clus_pi



    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


    # to reset metrics each time is called
    @property
    def metrics(self):
        "Return initialised metrics"
        return [L1_cri, L1_act, L_emb, L2, L3, val_L1_act, val_L1_cri, val_L_emb, val_L2, val_L3,
                clus_KL_sep, auroc, cat_acc]



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


    def compute_y_clusters(self, training = False):
        """
        Comute cluster phenotypes given embeddings.

        Parameters
        ----------
        training : str, optional
            Whether to train or not. The default is False.

        Returns
        -------
        Cluster phenotypes with shape (-1, num_clusters, latent_dim).
        """
        y_clusters = self.Predictor(tf.expand_dims(self.embeddings, axis = 0))
        
        return y_clusters
















