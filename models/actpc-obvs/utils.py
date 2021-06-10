#!/usr/bin/env python3
"""
Auxiliary functions for base ACTPC.

At the moment, currently temporal loss implementations are computed.
"""
import numpy as np
from scipy.stats import mode
from sklearn.metrics import roc_auc_score as ROC

from datetime import date
import os

import tensorflow as tf
import tensorflow.keras.callbacks as cbck


def compute_mode(array):
    
    "Compute mode for numpy array, across axis 1 (time-wise)"
    assert len(array.shape) == 3
    
    # Compute mode 
    array_mode = mode(array, axis = 1, nan_policy = "omit")
    
    return array_mode[0].reshape(-1, array.shape[-1])



class categorical_accuracy(cbck.Callback):
    def __init__(self, validation_data = (), interval = 5):
        super().__init__()
        
        self.interval = interval
        self.X_val, self.y_val = validation_data
        
    def on_epoch_end(self, epoch, logs = {}):
        
        # Check interval size
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose = 0)
            
            # compute predicted class
            class_pred = np.argmax(y_pred, axis = -1)
            class_true = np.argmax(self.y_val, axis = -1)
            
            # compute categorical accuracy = number of correct assignments
            cat_acc = np.sum(class_true == class_pred) / class_pred.size
            
            print("End of Epoch {:d} - Cat Acc: {:.5f}".format(epoch, cat_acc))
            
        
class KL_separation(cbck.Callback):
    "Sum over items to compute separation loss"
    def __init__(self, validation_data = (), interval = 5):
        super().__init__()
        
        self.interval = interval
        self.X_val, _    = validation_data
    
    def on_epoch_end(self, epoch, logs = {}):
        
        # Check interval size
        if epoch % self.interval == 0:
            
            total_loss = 0
            epsilon = 1e-9
            X = np.squeeze(self.model.Predictor(tf.expand_dims(self.model.embeddings, axis = 0)).numpy() + epsilon)
            
            
            # Compute num_clusters
            num_clusters = X.shape[0]
            
            for i in range(num_clusters):
                for j in range(i+1, num_clusters):
                    total_loss += np.sum(X[i, :] * np.log(X[i, :] / X[j, :]))
                    total_loss += np.sum(X[j, :] * np.log(X[j, :] /X[i, :]))
            
            
            # normalise
            norm_loss = total_loss / (0.5 * num_clusters * (num_clusters + 1))
            print("End of Epoch {:d} - KL sep : {:.6f}".format(epoch, norm_loss))
            
        
class emb_separation(cbck.Callback):
    "Sum over items to compute separation loss"
    def __init__(self, validation_data = (), interval = 5):
        super().__init__()
        
        self.interval = interval
        self.X_val, _    = validation_data
    
    def on_epoch_end(self, epoch, logs = {}):
        
        # Check interval size
        if epoch % self.interval == 0:
            
            total_loss = 0
            epsilon = 1e-9
            X = self.model.embeddings.numpy()
            
            # Compute num_clusters
            num_clusters = X.shape[0]
            
            for i in range(num_clusters):
                for j in range(i+1, num_clusters):
                    total_loss += np.sum(np.square(X[i, :] - X[j, :]))
            
            # normalise
            norm_loss = total_loss / (0.5 * num_clusters * (num_clusters + 1))
            print("End of Epoch {:d} - emb sep : {:.6f}".format(epoch, norm_loss))


class Confusion_Matrix(cbck.Callback):
    "Print Confusion Matrix"
    def __init__(self, validation_data = (), interval = 5):
        super().__init__()
        
        self.interval = interval
        self.X_val, self.y_val    = validation_data
        
        self.num_classes = self.y_val.shape[-1]
        
    def on_epoch_end(self, epoch, logs = {}):
        
        # Check interval size
        if epoch % self.interval == 0:
            
            CM     = np.zeros(shape = (self.num_classes, self.num_classes))
            y_pred = np.argmax(self.model.predict(self.X_val), axis = -1).reshape(-1)
            y_true = np.argmax(self.y_val, axis = -1).reshape(-1)
            
            for true_class in range(self.num_classes):
                
                samples_in_class = (y_true[:] == true_class)
                for pred_class in range(self.num_classes):
                    
                    conf_matrix_value = np.sum(y_pred[samples_in_class] == pred_class)
                    CM[true_class, pred_class] = conf_matrix_value
                    
            print("End of Epoch {:d} - Confusion matrix: \n {}".format(epoch, CM.astype(int)))
            
        
class AUROC(cbck.Callback):
    # Compute AUROC over predictions
    def __init__(self, validation_data = (), interval = 5):
        super().__init__()
        
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.y_val   = self.y_val.reshape(-1, self.y_val.shape[-1])
        
    def on_epoch_end(self, epoch, logs = {}):
        
        # Check epoch number
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            roc_auc_score = ROC(y_true = self.y_val, y_score = y_pred,
                                average = None, multi_class = 'ovo')
            
            print("End of Epoch {:d} - ROC score: {}".format(epoch, roc_auc_score))
            


def compute_metric(metric_name):
    if "auc" in metric_name.lower() or "roc"  in metric_name.lower():

        return AUROC
    
    elif "con" in metric_name.lower():

        return Confusion_Matrix

    elif "kl" in metric_name.lower():
        
        return KL_separation
     
    elif "acc" in metric_name.lower():
        
        return categorical_accuracy
    
    elif "emb" in metric_name.lower():
        
        return emb_separation

    else:
        print("Error with string specified")


# Callbacks
def get_callbacks(model_name, folder, loss_names, metric_names, track, csv_log = True, early_stop = True, 
                  lr_scheduler = True, save_weights = True, tensorboard = True, **kwargs):
    
    """
    Generate callbacks dictionary.
    """
    if folder[-1] != '/':
        folder = folder + '/'
        
    # Generate loss and model save paths
    today = date.today().strftime("%Y-%m-%d")
    save_folder = folder + 'experiments/{}/{}/checkpoints/'.format(today, model_name)
    csv_folder  = folder + 'experiments/{}/{}/results/'.format(today, model_name)
    log_dir = folder + 'experiments/{}/{}/logs/fit/'.format(today, model_name)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        
    # Initialise Callbacks Dictionary    
    callbacks_dic = {}

    for loss in loss_names:

        ckpt     = cbck.ModelCheckpoint(filepath = save_folder + '{}'.format(loss) + '/ckpt-{epoch}',
                                   monitor = loss, save_freq = 'epoch')
        val_ckpt = cbck.ModelCheckpoint(filepath = save_folder + '{}_val'.format(loss) + '/ckpt-{epoch}',
                                   monitor = 'val_' + loss, save_freq = 'epoch')
        csv_ckpt = cbck.CSVLogger(filename = csv_folder + loss, separator = ',', append = True)
        
        # Save callbacks
        callbacks_dic[loss] = ckpt
        callbacks_dic[loss + "_val"] = val_ckpt
        callbacks_dic[loss + '_csv'] = csv_ckpt

    for metric in metric_names:
    
        ckpt     = cbck.ModelCheckpoint(filepath = save_folder + '{}'.format(metric) + '/ckpt-{epoch}',
                                   monitor = metric, save_freq = 'epoch')
        csv_ckpt = cbck.CSVLogger(filename = csv_folder + metric, separator = ',', append = True)
        
        # Save callback
        callbacks_dic[metric] = ckpt
        callbacks_dic[metric + '_csv'] = csv_ckpt        
      
    
    if csv_log == True:
        callbacks_dic['CSV_Logger'] = cbck.CSVLogger(filename = csv_folder + "loss-metrics.csv", separator = ',', append = True)
        
        
    if early_stop == True:
        callbacks_dic['EarlyStopping'] = cbck.EarlyStopping(monitor = 'val_' + track,
        mode = "min", restore_best_weights = True, min_delta = 0.00001, **kwargs)     
        
        
    if lr_scheduler == True:
        callbacks_dic['ReduceLR'] = cbck.ReduceLROnPlateau(monitor = 'val_' + track,
                    mode = 'min', cooldown = 15, min_lr = 0.00001, factor = 0.25,**kwargs)    
        
        
    if save_weights == True:
        callbacks_dic['save_weights'] = cbck.ModelCheckpoint(filepath = save_folder + 'weights/ckpt-{epoch}', 
             monitor = 'val_' + track, save_weights_only = True, mode = 'max', save_best_only = False, save_freq = 'epoch')
    
    
    if tensorboard == True:
        callbacks_dic['TensorBoard'] = cbck.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
    
    return callbacks_dic























