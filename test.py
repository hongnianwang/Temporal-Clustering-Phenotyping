
from data_loader import import_data
from base import ACTPC
import tensorflow.keras.callbacks as CLB

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
    init_epochs_ac = 3, init_epochs_pred = 3,
    batch_size = 5096
)

# Define Model Callbacks
savepath = 'tmp/checkpoints/checkpoint'
csvpath  = 'tmp/results/metrics.csv'
metric_names = ['Critic_loss', 'Actor_loss', 'Emb_loss', 'L2_loss', 'L3_loss']
callbacks_dic = {}

for metric in metric_names:
    ckpt = CLB.ModelCheckpoint(filepath = savepath, monitor = 'val_' + metric)
    csv_ckpt = CLB.CSVLogger(filename = csvpath, separator = ',', append = True)
    callbacks_dic[metric] = ckpt

callbacks_dic['save_weights'] = CLB.ModelCheckpoint(filepath = savepath, monitor = 'val_' + 'Critic_loss',
                                  save_weights_only = True, mode = 'max', save_best_only = False, save_freq = 'epoch')

callbacks_dic['EarlyStopping'] = CLB.EarlyStopping(monitor = 'Critic_loss',
                                                         min_delta = 0.001, patience = 25   ,
                                                         mode = "min", restore_best_weights = True)
callbacks_dic['ReduceLR'] = CLB.ReduceLROnPlateau(monitor = 'Critic_loss',
                                                        factor = 0.2, patience = 25,
                                                        mode = 'min', cooldown = 5, min_lr = 0.001)


# save_model_L1 = Checkpoint(filepath = savepath, monitor =  )
init_model.compile(optimizer='adam')
init_model.fit(X_data, y_data, epochs = 250, batch_size = 100,
               callbacks = callbacks_dic.values())

