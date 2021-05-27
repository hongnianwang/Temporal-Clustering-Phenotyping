import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np

# Load GPUs
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



os.chdir('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/attention-project/scripts/')
sys.path.append('/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/attention-project/scripts/models/custom/')

from data_loader import load_from_csv

from utils_model import get_callbacks
from utils_model import KL_separation_loss as compute_KL
from model import ACTPC
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


#%% - Initialise model
from model import ACTPC
"""
Initialise model
"""
# Create model instance
init_model  = ACTPC(num_clusters = 8, latent_dim = 20, output_dim = 4,
                    beta  = 0.01, alpha = 0.01)
init_model.build(embeddings = None, input_shape = X_data.shape)

# Initialise parameters according to initialisation procedure
opt = optimizers.Adam(learning_rate = 0.01)
init_model.init_params(X_train, y_train, 
        optimizer = opt, 
        init_epochs_ac = 2000, 
        init_epochs_pred = 200, 
        batch_size = 32)

# Investigate cluster divergence and y divergence
print("Embeddings: ", np.linalg.norm(init_model.embeddings.numpy()[:, None, :] - init_model.embeddings.numpy()[None, :, :],
                                     axis = -1))
print("\n\nY distribution for clusters", init_model.compute_y_clusters())


#%% - Main Training
"""
Main training
"""

from model import ACTPC

# Define Model Callbacks
folder = '../'
loss_names = ['L1', 'L2', 'L3', 'entr', 'emb sep']
metric_names = ['KL_sep', 'AUC']

callbacks = get_callbacks('test14', folder = folder, loss_names = loss_names, metric_names = metric_names, 
                         track_metric = 'L1',  patience = 50)


# Initialise GPU strategy
strategy = tf.distribute.MirroredStrategy(devices = None,
                                  cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())     

# Use all GPUs
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    
    # Create new model instance and loads weights
    model = ACTPC(num_clusters = 8, latent_dim = 20, output_dim = 4, beta = 0.000001, alpha = 0.01)
    model.build(embeddings = init_model.embeddings.numpy(), input_shape = X_data.shape)
    model.set_weights(init_model.get_weights())

# Compile model
opt = optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, run_eagerly = True)


# Display Training
print("------------------------------------------------------")
print("Loss function descriptions: \nL1 - predictive clustering loss \nL2 - entropy loss")
print("L3 - embedding separation loss \nL1_cri - L1 \nL1_act - weighted_L1 + alpha*L2 (weights are pis)")
print("L_emb - L1 + beta*L3 /n")

model.fit(X_train, y_train, epochs = 100, batch_size = 64, 
          validation_data = (X_val, y_val), callbacks = callbacks.values())

# Investigate cluster divergence and y divergence
print("Embeddings: ", np.linalg.norm(model.embeddings.numpy()[:, None, :] - model.embeddings.numpy()[None, :, :],
                                     axis = -1))
print("\n\nY distribution for clusters", model.compute_y_clusters())

# Compare with outcome distributions
_, cluster_assign = init_model.compute_clusters_and_probs(X, one_hot = False, training = False)
traj_pred = cluster_assign.numpy().reshape(-1)
pat_pred  = tf.squeeze(utils.compute_mode(np.expand_dims(cluster_assign.numpy(), axis = -1)[:, 7:13, :]))

# Compare with trajectory assignment
traj_true = np.argmax(y, axis = -1).reshape(-1)

# compare with patients
pat_true  = np.argmax(y, axis = -1)[:, 0]

# Trajectory distribution
out_dist_per_traj = np.zeros(shape = (np.max(cluster_assign) + 1, np.max(traj_true) + 1))
out_dist_per_pat = np.zeros(shape = (np.max(cluster_assign) + 1, np.max(traj_true) + 1))

for k_ in range(np.max(cluster_assign) + 1):
    
    # k is current cluster
    traj_in_cluster = traj_pred == k_
    ids_in_cluster  = pat_pred == k_
    
    # Compute value count
    clus_, counts_ = np.unique(traj_true[traj_in_cluster], return_counts = True)
    out_dist_per_traj[k_, clus_] = counts_
    
    # Similar but for patients
    clus_, counts_ = np.unique(pat_true[ids_in_cluster], return_counts = True)
    out_dist_per_pat[k_, clus_] = counts_    


# Normalise per row
out_dist_per_traj = np.divide(out_dist_per_traj, np.sum(out_dist_per_traj, axis = 1).reshape(-1, 1))
out_dist_per_pat = np.divide(out_dist_per_pat, np.sum(out_dist_per_pat, axis = 1).reshape(-1, 1))

import seaborn as sns
import matplotlib.pylab as plt

# Plot all 3
fig, ax = plt.subplots(nrows = 2, ncols = 2)
test = sns.heatmap(out_dist_per_traj, linewidth = 0.5, ax = ax[0,0], annot = True)
ax[0,0].set_title("Trajectory outcome distribution")
test = sns.heatmap(out_dist_per_pat, linewidth = 0.5, ax = ax[0,1], annot = True)
ax[0,1].set_title("Patient outcome distribution")
test = sns.heatmap(tf.squeeze(model.compute_y_clusters()).numpy(), linewidth = 0.5, ax = ax[1,0], annot = True)
ax[1,0].set_title("Cluster predicted ys")


#%% Evaluate performance
import seaborn as sns
import matplotlib.pylab as plt

# Compute KL divergence of cluster separation
KL_div = compute_KL(model.compute_y_clusters()).numpy()
print("KL_divergence between clusters: {:.2f}".format(KL_div))

# Print embedding separation in latent space
fig, ax = plt.subplots()
cluster_sep = tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(model.embeddings, axis = 0), 
                                                  tf.expand_dims(model.embeddings, axis = 1)),
                                                  axis = -1).numpy()

ax = sns.heatmap(cluster_sep, linewidth=0.5)
plt.show()

# Print distance in phenotypes
fig, ax = plt.subplots()
phen_sep = tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(tf.squeeze(model.compute_y_clusters()), axis = 0), 
                                                  tf.expand_dims(tf.squeeze(model.compute_y_clusters()), axis = 1)),
                                                  axis = -1).numpy()

ax = sns.heatmap(phen_sep, linewidth=0.5)
plt.show()

# Print phenotypes
print(model.compute_y_clusters())

# Compute time mask
time_mask = mask[:, :, 0].reshape(-1)

# Compute and print AUROC for sequences and patients
y_output = model(X).numpy()
y_unroll = y_output.reshape(-1, 4)
y_true = y.reshape(-1, 4)

seq_auc = roc_auc_score(y_true = y_true[time_mask, :], y_score = y_unroll[time_mask, :],
                        multi_class = "ovr")
print("Sequential AUROC: {:.2f}".format(seq_auc))

# Compute mode
y_output_mode = utils.compute_mode(y_output[:, 6:-6, :])
y_true_mode   = utils.compute_mode(y)                      # Recovers original clusters
traj_auc = roc_auc_score(y_true = y_true_mode, y_score = y_output_mode,
                        multi_class = "ovr", average = None)
print("Trajectory AUROC: ", traj_auc)




