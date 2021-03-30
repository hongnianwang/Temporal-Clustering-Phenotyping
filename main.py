import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

import os, sys
import argparse

script_path_ = 'src/models/Attention-model/'
sys.path.append(script_path_)

from data_loader import import_data

parser = argparse.ArgumentParser()

parser.add_argument('--folder', type = str, default = '../data/processed-final/', help = 'folder path to obtain input arrays')
parser.add_argument('--data_name', type = str, default = 'Vit-All', help = 'Name of input data')
parser.add_argument('--seed', type = int, default = 1987, help = 'seed value')
# Think of adding network layers, etc... on a separate configuration file
    
fd_path_, data_name_ , seed_ = parser.parse_args().folder, parser.parse_args().data_name, parser.parse_args().seed

# Import and Load Data
X, y = import_data(fd_path_, data_name_)

# Train-Val-Test data splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = seed_)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = seed_)