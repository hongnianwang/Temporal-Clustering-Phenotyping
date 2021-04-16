#!/usr/bin/env python3
"""
Auxiliary functions for base ACTPC.

At the moment, currently temporal loss implementations are computed.
"""

import tensorflow as tf
from tensorflow.math import log, squared_difference, multiply
from sklearn.metrics import roc_auc_score


# Define custom loss functions
def predictive_clustering_loss(y_true, y_pred, y_type = 'categorical', name = 'pred_clus_L'):
    """Compute prediction clustering loss between predicted output and true output.
     Inputs have shape (batch_size, T, num_classes) for y_pred and (batch_size, num_classes) for y_true.
    There are a variety of different settings:

    - Binary:  Computes Binary Cross Entropy. Class/Event occurence is matched with a dimension.
                y_true with entries in [0,1], and y_pred with value between (0,1)
    - Categorical: Computes Cross Entropy Loss. Class assigned by highest value dimension.
                y_true is a one-hot encoding, and y_pred is a probabilistic vector.
    - Continuous: Computes L2 loss. Similar to the Binary case, but class attributes are continuous.
                y_true and y_pred both with real-value entries.

    Returns: Loss value between sample true y and predicted y based on y_type of shape (batch_size)
    """
    y_true_temp_  = tf.repeat(tf.expand_dims(y_true, axis = 1),
                              repeats = y_pred.shape[1], axis = 1, name = 'true_y_time')

    if y_type == 'binary':
        # Compute Binary Cross Entropy. y_pred output of sigmoid function to avoid taking log of infty.
        batch_loss = - tf.reduce_mean(tf.reduce_sum(y_true_temp_ * log(y_pred) + (1 - y_true_temp_) * log(y_pred),
                                                    axis = -1), name = name)

    elif y_type == 'categorical':
        # Compute Categorical Cross Entropy. y_pred output of softmax function to model probability vector.
        batch_loss = -tf.reduce_mean(tf.reduce_sum(y_true_temp_ * log(y_pred), axis = -1), name = name)

    elif y_type == 'continuous':
        # Compute L2 Loss. y_pred not given final output function.
        batch_loss = tf.reduce_mean(tf.reduce_sum((y_true_temp_ - y_pred) ** 2, axis = -1), name = name)

    else:
        raise Exception("""y_type not well-defined. Only possible values are {'binary', 'categorical',
                                                                                       'continuous'}""")
    return batch_loss


def actor_predictive_clustering_loss(y_true, y_pred, cluster_assignment_probs, y_type = 'categorical',
                                     name = 'actor_pred_clus_L'):
    """
    Compute prediction clustering loss between predicted output and true output with probability weights
    from cluster assignments.
    Inputs have shape (batch_size, T, num_classes) for y_pred and (batch_size, num_classes) for y_true
    or (batch_size, num_cluster) for cluster_assignment_probs.

    There are a variety of different settings, all weighted sample-wise by assignment probability:

    - Binary:  Computes Binary Cross Entropy. Class/Event occurence is matched with a dimension.
                y_true with entries in [0,1], and y_pred with value between (0,1)
    - Categorical: Computes Cross Entropy Loss. Class assigned by highest value dimension.
                y_true is a one-hot encoding, and y_pred is a probabilistic vector.
    - Continuous: Computes L2 loss. Similar to the Binary case, but class attributes are continuous.
                y_true and y_pred both with real-value entries.

    Returns: Loss value between sample true y and predicted y based on y_type of shape (batch_size)
    """
    y_true_temp_ = tf.repeat(tf.expand_dims(y_true, axis = 1),
                             repeats = y_pred.shape[1], axis = 1, name = 'true_y_time')

    if y_type == 'binary':
        # Compute Binary Cross Entropy weighted by cluster assignment probabilities.
        sample_loss = multiply(tf.reduce_sum(y_true_temp_ * log(y_pred) + (1 - y_true_temp_) * log(y_pred), axis = -1),
                               cluster_assignment_probs)
        batch_loss = -tf.reduce_mean(sample_loss, name = name)

        return batch_loss

    elif y_type == 'categorical':
        # Compute Categorical Cross Entropy weighted by cluster assignment probabilities.
        sample_loss = multiply(tf.reduce_sum(y_true_temp_ * log(y_pred), axis = -1), cluster_assignment_probs)
        batch_loss = -tf.reduce_mean(sample_loss, name = name)

        return batch_loss

    elif y_type == 'continuous':
        # Compute L2 Loss weighted by cluster assigment probabilities.
        sample_loss = multiply(tf.reduce_sum((y_true - y_pred) ** 2, axis = -1), cluster_assignment_probs)
        batch_loss = tf.reduce_mean(sample_loss, name = name)

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
    embedding_row = tf.expand_dims(y_embeddings, axis = 0)  # shape 1, K, latent_dim

    # Compute L1 distance
    pairwise_loss = tf.reduce_sum(squared_difference(embedding_column, embedding_row),
                                  axis = -1)  # shape K, K
    loss = - tf.reduce_sum(pairwise_loss, axis = None, name = name)

    return loss


def selector_init_loss(y_prob, clusters, name = 'init_selec_loss'):
    """
    Selector Loss for Initialisation. Initialise Selector given estimated cluster assignments.

    Input shapes are (bs, T, num_clusters), and y_prob sums to 1 over the last axis (outputs probability vector).
    Clusters[i, j, :] is a one-hot encoding of estimated cluster for sample i truncated at time stamp j.

    returns: Categorical Cross-Entropy loss averaged over samples and time-steps. This is very similar to predictive
    clustering loss without cluster assignment constant across time
    """
    # Compute Categorical Cross Entropy. y_pred output of softmax function to model probability vector.
    batch_loss = -tf.reduce_mean(tf.reduce_sum(clusters * log(y_prob), axis = -1), name = name)

    return batch_loss



# class old_AUROC(tf.keras.metrics.Metric):
#
#     def __init__(self, name = 'Numpy AUROC computation', **kwargs):
#         super().__init__(**kwargs)
#
#     def update_state(self, y_true, y_pred, sample_weight = None):
#         # Compute using numpy
#         npy_true = y_true.numpy()
#         npy_pred = y_pred.numpy()
#
#         auroc    = roc_auc_score(
#             y_true = npy_true,
#             y_score= npy_pred,
#             average= None,
#             multi_class='OVO'
#         )




























