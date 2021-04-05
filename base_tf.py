import tensorflow as tf
import numpy as np
import tensorflow.math as math
import os, sys

class actpc_base(tf.Keras.Model):
    def __init__(self, *args, **kwargs):
        self.sess = sess
        self.batch_size = batch_size
        .....

    def call(self, inputs, training = True):
        x_inputs, y_inputs     = inputs

        embedding     = self.static_Encoder(x_inputs)
        cluster_probs = self.selector(embedding)

        # Sample cluster
        cluster_samp  = tf.random.categorical(logits = cluster_probs, num_samples = 1, seed = 1717, name = 'cluster sampling')
        cluster_emb   = tf.gather_nd(params = self.embeddings, indices = tf.expand_dims(cluster_samp, -1))

        # Feed cluster as input to predictor
        y_pred        = self.static_Predictor(cluster_emb)

        return y_pred, cluster_probs


for epochs in range(epochs):
    print("\nStart of epoch ")




    def train_step(self, inputs):
        # Unpack the data
        x_inputs, y_inputs = inputs

        with tf.GradientTape() as tape:
            # Compute the Forward Pass and loss
            y_pred, cluster_probs     = self(x_inputs, training = True)
            loss1_     = predictive_clustering_loss(
                y_true = y_inputs,
                y_pred = y_pred,
                y_type = self.y_type,
                name   = 'pred_clus_loss'
            )
            loss2_     = cluster_probability_entropy_loss(
                y_prob = cluster_probs,
                name   = 'clus_entr_L'
            )

        # Compute Gradients for Network Update
        network_vars   = [var for var in self.trainable_variables if 'emb' not in var]
        gradients      = tape.gradient(loss1_ + self.alpha * loss2_, network_vars)
        self.optimizer.apply_gradients(zip(gradients), network_vars)

        # Update embeddings
        emb_vars       = [var for var in self.trainable_variables if 'emb' in var]
        emb_gradients  = tape.gradient(loss_1, emb_vars)
        self.optimizer.apply_gradients(zip(emb_gradients), emb_vars)

        # Update metrics
        self.compiled_metrics.update_state(y_inputs, y_pred)

        # return metric dic
        return {metric.name: metric.result() for metric in self.metrics}




