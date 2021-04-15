import tensorflow as tf
from data_loader import import_data
from utils import predictive_clustering_loss
from base import ACTPC

# Import data as tensorflow data Dataset
X_data, y_data = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)



init_model  = ACTPC(
    num_clusters = 12,
    latent_dim = 32,
    output_dim = y_data.shape[-1],
    beta  = 0.000001,
    alpha = 0.01)

init_model.build(input_shape = X_data.shape)
init_model.network_initialise(
    X = X_data, y = y_data,
    optimizer = 'adam',
    init_epochs_ac = 5, init_epochs_pred = 5,
    batch_size = 512
)

# init_model.compile(optimizer='adam')
# init_model.fit(X_data, y_data, epochs = 5, batch_size = 100)



y_data.shape