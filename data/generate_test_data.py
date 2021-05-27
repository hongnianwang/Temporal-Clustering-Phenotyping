import numpy as np
import pandas as pd
from random import sample
import os

time_stpms = np.linspace(start = 0.0, stop = 50.0, num = 20)
random_    = np.geomspace(start = 1E-2, stop = 1E+2, num = 30)

data_ids_  = pd.DataFrame(data = 0, index = np.array(range(10000)).astype(str), columns = ['amp1', 'amp2', 'freq1', 'freq2', 'lin', 'log_', 'err'])
x          = np.empty(shape = (len(range(10000)), len(time_stpms), 4))
y          = np.zeros(shape = (10000, 5))

for i in range(10000):

    generators_   = sample(random_.tolist(), 6)
    amp1_, amp2_, freq1_, freq2_, lin_, log_ = generators_
    err_ = np.random.normal(0,1, size = 1)

    # Save config ids_
    data_ids_.iloc[i, :-1] = generators_
    data_ids_.iloc[i, -1]  = err_

    # Generate data matrix
    x[i, :, 0] = amp1_ * np.cos(freq1_ * time_stpms) + amp2_ * np.sin(freq2_ * time_stpms) + lin_ * time_stpms + log_ * np.log(time_stpms + 1) + np.random.normal(0.0, 0.1, len(time_stpms))
    x[i, :, 1] = freq1_* np.cos(amp1_  * time_stpms) + freq2_ * np.sin(amp2_ * time_stpms) + log_ * time_stpms + lin_ * np.log(time_stpms + 1) + np.random.normal(0.0, 0.1, len(time_stpms))
    x[i, :, 2] = amp2_ * np.cos(freq2_ * time_stpms) + amp1_ * np.sin(freq1_ * time_stpms) + lin_ * time_stpms + log_ * np.log(time_stpms + 1) + np.random.normal(0.0, 0.1, len(time_stpms))
    x[i, :, 3] =freq2_* np.cos(amp2_  * time_stpms) + freq1_ * np.sin(amp1_  * time_stpms) + log_ * time_stpms + lin_ * np.log(time_stpms + 1) + np.random.normal(0.0, 0.1, len(time_stpms))

    # Generate outputs
    mean_frac_, _ = np.modf(np.max(x[i, :]))
    if 0 <= mean_frac_ <= 0.1:
        y_ = [1,0,0,0,0]
    elif 0.1 < mean_frac_ <= 0.2:
        y_ = [0,1,0,0,0]
    elif 0.2 < mean_frac_ <= 0.25:
        y_ = [0,0,1,0,0]
    elif 0.25 < mean_frac_ <= 0.4:
        y_ = [0,0,0,1,0]
    elif 0.4 < mean_frac_ <= 1:
        y_ = [0,0,0,0,1]
    else:
        print('Error: ', mean_frac_)
        break

    y[i, :] = y_

'Save generated data'
save_folder = '/home/ball4537/PycharmProjects/PhD-I/data/sample/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

data_ids_.to_csv(save_folder + 'data_params.csv', index = True)
np.save(save_folder + 'X.npy', x, allow_pickle = True)
np.save(save_folder + 'y.npy', y, allow_pickle = True)


