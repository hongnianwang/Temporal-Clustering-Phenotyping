import os
import numpy as np
import tensorflow as tf
from data_loader import import_data
from utils import predictive_clustering_loss
from base import AC_initialiser

X, y = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)

init_model  = AC_initialiser(
    y_dim = y.shape[-1],
    num_clusters = 12,
    intermediate_dim = 32,
    seed = 2323,
    y_type = 'categorical'
)

init_model.compile(optimizer = 'adam', loss=predictive_clustering_loss)
init_model.fit(X, y, epochs = 10, batch_size = 64)


def compute_embeddings(self, inputs):
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    "Compute embedding vectors given latent projection"
    latent_projs = self.Encoder(inputs)

    # Initialise K-means
    init_km = KMeans(n_clusters=self.K, init='k-means++',
                     precompute_distance=True, verbose=5,
                     random_state=self.seed, n_jobs=-1)
    init_km.fit(inputs)

    # Obtain cluster centres and sample cluster assignments
    centroids = init_km.cluster_centers_
    cluster_assig = init_km.predict(self, inputs)

    # Check shapes
    assert centroids.shape == (self.K, self.int_dim)
    assert cluster_assig.shape == (inputs.shape[0],)

    return centroids, cluster_assig
