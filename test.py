import os
import numpy as np
import tensorflow as tf

from data_loader import import_data

save_folder = '/home/ball4537/PycharmProjects/PhD-I/data/sample/'
X, y        = import_data(save_folder , 'X')

model = M.LSTM_AE(original_dim = X.shape[2] ,
                  z_dim = 1,
                  num_nodes = 20,
                  num_layers = 0)

model.compile(optimizer = 'adam', metrics = tf.keras.metrics.AUC)
model.fit(X, y, epochs = 10, batch_size = 64)