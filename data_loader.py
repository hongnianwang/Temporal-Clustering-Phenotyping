import numpy as np
import os
import tensorflow as tf

def import_data(folder_path, data_name):
    
    '''
    Function to import data. Given folder, selects the corresponding dataset.
    
    Input:
        - folder_path: a path corresponding to the overall folder path
        - data_name: One of "Vit, Vit-All, Vit-Bio, Vit-Ser, Vit-Sta, Vit-Sta-Bio, Vit-Sta-Ser"
        - categorical: Binary flag indicating whether outputs are categorical or binary.
        
    Output: tuple of numpy arrays:
        - data_x: Numpy array of shape (N, max_len_, x_dim) - N is the number of patients
                                                            - max_len_ is the max time-series length
                                                            - x_dim is the dimension of the features. 
                data_x is normalised with pre-filled "0"s where patients had admissions less than max_len_
                
        - data_y: Numpy array of shape (N, max_len, y_dim)
                If categorical format, y_dim is the number of categories and data_y[ , , :] is a one-hot encoding of the label.
    '''
    if not os.path.exists(folder_path):
        print('Wrong Folder_path specified when loading data')
        
    else:
        try:
            x_path_ = folder_path + '{}.npy'.format(data_name)
            y_path_ = folder_path + 'y.npy'
            
            X, y    = np.load(x_path_, allow_pickle = True), np.load(y_path_, allow_pickle = True)

            assert X.shape[0] == y.shape[0]      # X and y must agree on #patient and time dimensions
            
            if np.isnan(X).sum() > 0:              # Some nan values not filled exactly
                print('Not all values filled! ')
                X   = np.nan_to_num(X, copy = False, nan = 0.0)
                assert np.isnan(X) == 0 
        
        except:
            print('Wrong data name specified!')
            raise

        print('To-do: Need to check for standardisation!')

        # Convert to Tensor Dataset
        X = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X, dtype = 'float32'))
        y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y, dtype = 'float32'))

        return X, y