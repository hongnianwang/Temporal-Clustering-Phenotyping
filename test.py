import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import import_data
from base import ACTPC
import tensorflow.keras.callbacks as CLB
from sklearn.metrics import roc_auc_score

# Import data as tensorflow data Dataset
X_data, y_data = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size = 0.5, random_state = 2323, shuffle=True)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, train_size = 0.5, random_state = 2323, shuffle = True)

# Define Model Callbacks
savepath = 'tmp/checkpoints/'
csvpath  = 'tmp/results/'
metric_names = ['cri_loss', 'act_loss', 'emb_loss', 'L2', 'L3']
callbacks_dic = {}

for metric in metric_names:
    ckpt = CLB.ModelCheckpoint(filepath = savepath + '{}_val'.format(metric) + '/ckpt-{epoch}',
                               monitor = 'val_' + metric, save_freq = 'epoch')
    csv_ckpt = CLB.CSVLogger(filename = csvpath + metric, separator = ',', append = False)
    callbacks_dic[metric] = ckpt

callbacks_dic['save_weights'] = CLB.ModelCheckpoint(filepath = savepath + 'weights/ckpt-{epoch}',
            monitor = 'val_' + 'cri_loss', save_weights_only = True, mode = 'max', save_best_only = False, save_freq = 'epoch')

callbacks_dic['EarlyStopping'] = CLB.EarlyStopping(monitor = 'val_cri_loss',
                                                         min_delta = 0.001, patience = 10,
                                                         mode = "min", restore_best_weights = True)
callbacks_dic['ReduceLR'] = CLB.ReduceLROnPlateau(monitor = 'val_cri_loss',
                                                        factor = 0.2, patience = 10,
                                                        mode = 'min', cooldown = 5, min_lr = 0.001)

# Initialise GPU strategy
# strategy = tf.distribute.MirroredStrategy(devices = None,
#                                   cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())      # Use all GPUs
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# with strategy.scope():
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
init_epochs_ac = 3, init_epochs_pred = 3,
batch_size = 5096
)


init_model.compile(optimizer='adam')
init_model.fit(X_train, y_train, epochs = 250, batch_size = 100, validation_data = (X_val, y_val),
               callbacks = callbacks_dic.values())

pis_pred_test = init_model.Selector(init_model.Encoder(y_test, training = False), training = False)
pis_pred_data = init_model.Selector(init_model.Encoder(y_data, training = False), training = False)

# Compute AUROC
