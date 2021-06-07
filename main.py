import tensorflow as tf
import numpy as np
import pandas as pd

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


# Load GPUs
# physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#print("Num GPUs Available: ", len(physical_devices))



from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


os.chdir('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/adding-attention/scripts/')
sys.path.append('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/adding-attention/scripts/models/main/')

from data_loader import load_from_csv
# Compute Data Loader load from sample

import utils_model
from utils_model import predictive_clustering_loss as L1
from utils import get_callbacks, compute_metric

from model import ACTPC, Enc_Pred, SEL_INIT


"""
Load and import data
"""

# Import data as float32 numpy arrays
X_data, y_data, ids, mask  = load_from_csv(
    folder_path = '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    X_name   = 'COPD_VLS_process', y_name = 'copd_outcomes', time_range  = (0, 72), feat_name    = 'vitals', norm = "min-max")
# X_data = np.ones(shape = (5000, 18, 4))
# y_data = np.eye(5)[np.repeat(a = np.array([0,1,2,3,4]), repeats = 5000//5).reshape(-1)]
# ids = np.ones(shape = (5000, 18, 2))
# mask = np.ones(shape = (5000, 18))
# mask[:, -4:] = 0

# Re-label and assign each subsequence to time-series.
X, y = X_data, np.repeat(np.expand_dims(y_data, axis = 1), repeats = X_data.shape[1], axis = 1)


# Split into train, validation, test data
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, train_size = 0.6, 
    random_state = 2323, shuffle=True, stratify = np.argmax(y, axis = -1)[:, 0])

X_train, X_val, y_train, y_val, id_train, id_val   = train_test_split(
    X_train, y_train, id_train, train_size = 0.8, 
    random_state = 2323, shuffle = True, stratify = np.argmax(y_train, axis = -1)[:, 0])


#%% Select arguments
name = "test02"
output_dim = 4
y_type = "categorical"
latent_dim = 32
num_clusters = 7
seed = 2323

# Network parameters
num_encoder_layers, num_encoder_nodes = 2, 128
state_fn, recurrent_fn = "tanh", "sigmoid"
encoder_dropout, recurrent_dropout = 0.6, 0.0
mask_value = 0.0

num_predictor_layers, num_predictor_nodes = 3, 128
predictor_fn = "relu"
predictor_dropout = 0.7

num_selector_layers, num_selector_nodes = 3, 128
selector_fn, selector_output_fn = "relu", "softmax"
selector_dropout, selector_name = 0.7, "selector"


# Training parameters
beta_init = 1
alpha_init = 1
lr_init = 0.001
bs_init = 64

init_epochs_ac = 1000
init_epochs_sel = 50

beta = 0.01
alpha = 0.01
lr   = 0.001

epochs = 1000
bs     = 64

interval = 5



#%% Run model initialise with Actor-Critic Model
import utils
from utils import get_callbacks, compute_metric

# Initialise GPU strategy
# strategy = tf.distribute.MirroredStrategy(devices = None,
#                                   cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())    
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Use all GPUs
# with strategy.scope():
    
# Initialise Actor-Critic model
init_ac = Enc_Pred(num_clusters = num_clusters, output_dim = output_dim, y_type = y_type, 
                    latent_dim = latent_dim, seed = seed,
                    num_encoder_layers= num_encoder_layers, num_encoder_nodes=num_encoder_nodes,
                    state_fn=state_fn, recurrent_activation=recurrent_fn,
                    encoder_dropout=encoder_dropout, recurrent_dropout=recurrent_dropout, mask_value=mask_value,
                    num_predictor_layers=num_predictor_layers, num_predictor_nodes=num_predictor_nodes,
                    predictor_fn=predictor_fn, predictor_dropout=predictor_dropout)

# Initialise model for main_training
opt_init = optimizers.Adam(learning_rate = lr_init)
    
# Compile Model 
init_ac.compile(optimizer = opt_init, loss = L1, metrics = "ce")


# Load callbacks
model_name = "{}/init-ac".format(name)
folder     = "../"
loss_names   = ["loss"]
metric_names = ["Cat-Acc", "AUC", "conf-mat"]
track = "loss"
callbacks = get_callbacks(model_name, folder, loss_names, metric_names, track, csv_log = True, early_stop = True, 
                  lr_scheduler = True, save_weights = True, tensorboard = True, patience = 30
    )

for metric in metric_names:
    callbacks.update({metric: compute_metric(metric)((X_val, y_val), interval = interval)})


# Weights
# weights = np.sum(y_val.reshape(-1, output_dim), axis = 0)
# norm_weights = 1 / (weights * np.sum(1/weights))

# weight_dic = {key:value for key, value in zip(range(4), norm_weights)}

# Fit
init_ac.fit(x = X_train, y = y_train, batch_size = bs_init, epochs = init_epochs_ac, verbose = 1,
               validation_data=(X_val, y_val),  callbacks = callbacks)



#%% Initialise Cluster Embeddings and Selector Network

# with strategy.scope():
    
# Compute cluster embeddings
emb_vecs_, cluster_assign, km = init_ac.initialise_embeddings(X_train)


# Initialise Selector Component
previous_params = dict((key, init_ac.model_config[key]) for key in init_ac.model_config.keys() if "predictor" not in key)
init_sel = SEL_INIT(alpha = alpha_init,
                 num_selector_layers = num_selector_layers, num_selector_nodes = num_selector_nodes,
                 selector_fn = selector_fn, selector_output_fn=selector_output_fn, 
                 selector_dropout=selector_dropout, selector_name=selector_name, **previous_params)
init_sel.build(X_train.shape)

# Compute embeddings
init_sel.embeddings.assign(tf.convert_to_tensor(emb_vecs_, name = 'embeddings', dtype = 'float32'), name = 'embeddings_init')
  
# Update weights for Encoder
init_sel.Encoder.set_weights(init_ac.Encoder.get_weights())

# Compile Model
init_sel.compile(optimizer = opt_init, loss = utils_model.selector_init_loss, metrics = "ce")

# Load Callbacks
model_name = "{}/init-sel".format(name)
folder     = "../"
loss_names   = ["loss"]
metric_names = []
track = "loss"
callbacks = get_callbacks(model_name, folder, loss_names, [], track, csv_log = True, early_stop = True, 
                  lr_scheduler = True, save_weights = True, tensorboard = True, patience = 30
    )

for metric in metric_names:
    callbacks.update({metric: compute_metric(metric)((X_val, y_val), interval = interval)})

# Fit
clusters_target_train = tf.one_hot(cluster_assign.astype(np.float32), depth = num_clusters)
clusters_target_val   = tf.one_hot(km.predict(y_val.reshape(-1, 4)).reshape(y_val.shape[:-1]).astype(np.float32),
                                   depth= num_clusters)
                                                                 
init_sel.fit(x = X_train, y = clusters_target_train, batch_size = bs_init, epochs = init_epochs_sel, 
             verbose = 1, validation_data = (X_val, clusters_target_val), callbacks = callbacks)

#####

# NEED TO RESTORE BEST WEIGHTS INSTEAD OF CURRENT

#%% Main Training
from model import ACTPC

# with strategy.scope():

# Load data
model = ACTPC(num_clusters=num_clusters, output_dim=output_dim, y_type=y_type,
              latent_dim=latent_dim, beta=beta, alpha=alpha, seed=seed, num_encoder_layers=num_encoder_layers,
              num_encoder_nodes=num_encoder_nodes, state_fn=state_fn, recurrent_activation=recurrent_fn,
              encoder_dropout=encoder_dropout, recurrent_dropout=recurrent_dropout, mask_value=mask_value,
              num_selector_layers=num_selector_layers, num_selector_nodes=num_selector_nodes, selector_fn=selector_fn,
              selector_output_fn=selector_output_fn, selector_dropout=selector_dropout,
              num_predictor_layers=num_predictor_layers, num_predictor_nodes=num_predictor_nodes, 
              predictor_fn=predictor_fn, predictor_dropout=predictor_dropout,
              encoder_name='encoder', predictor_name='predictor', selector_name='selector')

embs  = init_sel.embeddings
model.build(input_shape = X_data.shape, embeddings = embs)

# Update weights
model.Encoder.set_weights(init_sel.Encoder.get_weights())
model.Predictor.set_weights(init_ac.Predictor.get_weights())
model.Selector.set_weights(init_sel.Selector.get_weights())

# Load optimizer
opt = optimizers.Adam(learning_rate=lr)
model.compile(optimizer = opt, metrics="ce",run_eagerly = True)

# Load Callbacks
model_name = "init-sel/{}".format(name)
folder     = "../"
loss_names   = ["loss"]
metric_names = ["ROC", "ACC", "KL-sep", "conf-mat", "emb-sep"]
track = "L1"
callbacks = get_callbacks(model_name, folder, loss_names, metric_names, track, csv_log = True, early_stop = True, 
                  lr_scheduler = True, save_weights = True, tensorboard = True, patience = 30
    )

for metric in metric_names:
    callbacks.update({metric: compute_metric(metric)((X_val, y_val), interval = interval)})
    


# Train model
model.fit(x = X_train, y = y_train, batch_size=bs, epochs=epochs, verbose = 1,
          validation_data = (X_val, y_val), callbacks = callbacks)





