import tensorflow as tf
import numpy as np
import pandas as pd

# Load GPUs
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import os, sys
from hashlib import sha256

os.chdir('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/attention-project/scripts/')
sys.path.append('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/attention-project/scripts/models/custom/')
sys.path.append('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/attention-project/scripts/models/KMeans/')

from data_loader import load_from_csv

from utils_model import get_callbacks
from utils_model import KL_separation_loss as compute_KL
from model import ACTPC
from kmeans import init_kmeans_model
import utils


"""
Load and prepare data
"""

# Import data as float32 numpy arrays
X_data, y_data, ids, mask  = load_from_csv(
    folder_path = '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    X_name   = 'COPD_VLS_process', y_name = 'copd_outcomes', time_range  = (0, 72), feat_name    = 'vitals', norm = "min-max")

# Re-label and assign each subsequence to time-series.
X, y = X_data, np.repeat(np.expand_dims(y_data, axis = 1), repeats = X_data.shape[1], axis = 1)

# Split into train, validation, test data
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, train_size = 0.6, 
    random_state = 2323, shuffle=True, stratify = np.argmax(y, axis = -1)[:, 0])

X_train, X_val, y_train, y_val, id_train, id_val   = train_test_split(
    X_train, y_train, id_train, train_size = 0.8, 
    random_state = 2323, shuffle = True, stratify = np.argmax(y_train, axis = -1)[:, 0])


# Wrap data in Dataset objects.
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))


name = "custom"

#%% - KMeans

if name == "KMeans":
    algorithm = "TSKM"
    model = init_kmeans_model(algorithm = algorithm, K = 6,
                              metric = "dtw", verbose = 1, init = "k-means++", seed = 2323,
                              max_iter = 100, n_init = 5, n_jobs = -1)
    model.fit(X)
    
    # Predict
    y_pred = model.predict(X)
    
    # Compute assignment
    pats_  = ids[:, 0, 0].astype(int)
    assign = pd.DataFrame(index = pats_, data = y_pred, columns = ["cluster"])
    assign.index.name = "subject_id"
    assign.reset_index(drop = False, inplace = True)
    
    # Encode subject id
    assign["subject_id"] = assign.subject_id.apply(lambda x: sha256(str(x).encode()).hexdigest())
    assign["cluster"] = assign.cluster.apply(lambda x: sha256(str(x).encode()).hexdigest())
    
    assign.to_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/Alex_share/KMeans.csv")


#%% Run Custom model

if name == "custom":
    
    # Load model and initialise
    # Create model instance
    init_model  = ACTPC(num_clusters = 8, latent_dim = 2, output_dim = 4, beta  = 1, alpha = 0.01)
    init_model.build(input_shape = X_data.shape)
    
    # Initialise parameters according to initialisation procedure
    opt = optimizers.Adam(learning_rate = 0.001)
    init_model.init_params(X_train, y_train, optimizer = opt, init_epochs_ac = 3, 
            init_epochs_pred = 3, batch_size = 128)
    
    
    # 





