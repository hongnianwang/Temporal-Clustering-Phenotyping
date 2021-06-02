import numpy as np
import pandas as pd
import tensorflow as tf
from tslearn.utils import to_time_series_dataset

import os, sys
import datetime as dt
from tqdm import tqdm


def check_folder_path_exists(folder_path):
    "Quick check for whether folder path exists"
    
    if not os.path.exists(folder_path):
        print("Folder path does not exist. Stopping execution.")
        sys.exit()
        
        
def compute_time_to_end(data, id_col, time_column, **kwargs):
    """
    Compute time to end of admission for each id in id_col as given by the time column
    """
    time_to_end = data.groupby(id_col).apply(lambda x: x[time_column].max() - x[time_column])
    time_to_end.reset_index(drop = True, inplace = True)
    
    # Convert to num based on kwargs
    output      = convert_datetime_to_num(time_to_end, **kwargs)
    
    return output


def convert_datetime_to_num(data, mode = "hours"):
    """
    Convert timedelta series to num according to mode.

    Parameters
    ----------
    data : pd.Series 
        Array-like with timedelta values.
    mode : string, possibilities include "days", "hours", "seconds", or None (keep as is).

    Returns data in numeric format as determined by mode.
    """    
    total_seconds = data.dt.total_seconds()
    
    if "h" in mode.lower()[0]:
        # Convert timedelta to hours
        seconds_per_hour = 3600
        
        output_ = total_seconds / seconds_per_hour
        
    elif "d" in mode.lower()[0]:
        # Convert timedelta to days
        seconds_per_day = 24 * 3600
        
        output_ = total_seconds / seconds_per_day
        
    elif "s" in mode.lower()[0]:
        # Use total seconds
        
        output_ = total_seconds
        
    elif mode == None:
        # Keep as is
        
        output_ = data
        
    return output_
        
    
    
    
def convert_to_3darray(data, id_col = "subject_id", time_col = "time_to_end"):
    """
    Convert 2D dataframe to 3d array, indexed by id_col, and time-steps given by time_col.
    
    returns: 3D array of shape Num_idx x max_time_steps x (features + 1),
    where last element of 3rd axis corresponds to time diff (i.e. time between observations)

            2D array of pat ids and times.
    """
    
    # Obtain relevant shape sizes
    max_time_length = data.groupby(id_col)[time_col].count().max()
    num_ids         = data[id_col].nunique()
    
    # Compute unique ids
    ids = data[id_col].unique()
    
    # determine features of data
    features = [col for col in data.columns if col not in [id_col, time_col]]
    
    # Initialise output array
    feats_array = np.empty(shape = (num_ids, max_time_length, data.shape[1] - 1))
    feats_array[:] = np.nan
    
    # Initialise subject, times arrays
    id_times_array = np.empty(shape = (num_ids, max_time_length, 2))
    id_times_array[:] = np.nan
    
    
    # Update id values
    id_times_array[:, :, 0] = np.repeat(ids.reshape(-1, 1), repeats = max_time_length, axis = -1)
    
    
    # Iterate through each id. Update time values and output array
    for id_ in tqdm(ids):
        
        # index of id_
        index = np.where(ids == id_)[0]
        
        # Subset patient data
        id_data = data[data[id_col] == id_]
        id_feats= id_data[features].values
        time_   = id_data[time_col].diff().values
        
        # compute id length
        assert id_feats.shape[0] == time_.size
        id_time_length = id_feats.shape[0]
        
        # Update values
        feats_array[index, :id_time_length, :-1]  = id_feats
        feats_array[index, :id_time_length, -1]   = time_
        id_times_array[index, :id_time_length, 1] = time_
        
    return feats_array.astype("float32"), id_times_array.astype("float32")
    


def feats_from_name(name):
    """
    Given name, obtain relevant set of features
    """
    vitals = ['HR', 'RR', 'SBP', 'DBP', 'SPO2', 'FIO2', 'TEMP', 'AVPU']
    serum  = ['HGB', 'WGC', 'EOS', 'BAS', 'EBR', 'NEU', 'LYM', 'NLR']
    biochem= ['ALB', 'CR', 'CRP', 'POT', 'SOD', 'UR']
    static = ['age', 'gender', 'cci', 'is_elec', 'is_surg']
    
    # Vitals are always considered.
    out_feats_ = set(vitals)
    
    if 'vit' in name.lower():
        # Add vitals
        out_feats_.update(vitals)
        
    
    if 'ser' in name.lower():
        # Add serum variables
        out_feats_.update(serum)
    
    
    if 'bio' in name.lower():
        # Add biochem variables
        out_feats_.update(biochem)
        
    
    if 'sta' in name.lower():
        # Add static variables
        out_feats_.update(static)
        
    
    if 'lab' in name.lower():
        # Add serum and biochem variables
        out_feats_.update(biochem)
        out_feats_.update(serum)
        
        
    if 'all' in name.lower():
        # Select all variables
        out_feats_ = str(feats_from_name('bio-ser-sta'))
        
    print("Subsetting to {} features".format(name))
    
    return list(out_feats_)
    


def normalise(data, mode = "min-max", ignore_time = True):
    """
    

    Parameters
    ----------
    data : numpy 3Darray of shape (num_ids, max_time_length, num_feats + 1), with missing values.
    mode : str, either "min-max", "Gaussian", or None. (default = "min-max")
        Type of normalisation/standardisation.
    ignore_time: bool.
        Whether to ignore time dimension (-1) and remove it. If True, ignore, if False keep and not norm.

    Returns 
    -------
    Normalised data as given by mode. Normalisation completed through first axis (ids)

    """
    # compute num ids
    num_ids = data.shape[0]
    time_steps = data.shape[1]
    
    # Flatten for ease of computation
    if ignore_time:
        data_flatten = data[:, :, :-1].reshape(num_ids, -1)
        
    else:
        data_flatten = data.reshape(num_ids, -1)
    
    
    if "gauss" in mode.lower() or "norm" in mode.lower():
        
        # Complete Z-normalisation
        mean_flatten = np.nanmean(data_flatten, axis = 0)
        std_flatten  = np.nanstd(data_flatten, axis = 0)
        
        # Normalise
        output       = np.divide((data_flatten - mean_flatten), std_flatten)
        
    elif "min" in mode.lower() and "max" in mode.lower():
        
        # Compute min-max normalisation. Ignoring NaN
        min_flatten  = np.nanmin(data_flatten, axis = 0).reshape(1, -1)
        max_flatten  = np.nanmax(data_flatten, axis = 0)
        range_flatten= max_flatten - min_flatten
        
        # Normalise
        output = np.divide((data_flatten - min_flatten), range_flatten)
        
    elif "min" == None:
        
        output = data_flatten
        
    
    # Reshape into original size
    output = output.reshape(num_ids, time_steps, -1)    
    
    # If keep time, add time dimension
    if not ignore_time:
        
        # Add original time index
        orig_t = np.expand_dims(data[:, :, -1], axis = -1)
        assert len(orig_t.shape) == 3
        
        output = np.concatenate((output, data[:, :, -1]))
        
    
    # Check expected shape
    assert output.shape == data.shape or output.shape[-1] == (data.shape[-1] - 1)
        
    print("%s normalisation completed." % mode)
    
    return output


def impute_regular_NaNs(data, id_col = "subject_id", time_col = "time_to_end", fill_limit = None):
    """
    Function to impute parameters.

    Parameters
    ----------
    data : Pandas DataFrame
        contains 2 columns with id and time_col
    id_col: str, column to use as id identifier, default = "subject_id
    time_col: str, column to use as time to end identified, default = "time_to_end"
    fill_limt: int, number of limits to forward and backward filling, default = None (no limit)

    Returns
    -------
    Pandas DataFrame with intermediate values imputed. In turn, the following are completed:
        a) Forward Filling (propagate forward)
        b) Backward Filling (propagate backwards)
        c) Median imputing
    """
    out_ = data.copy(deep = True)
    features = [col for col in data.columns if col not in [id_col, time_col]]
    
    if out_.isna().any().any():
        print("{:.2f} % missing values in data. Imputing.".format(
            100 * out_.isna().sum().sum() / (out_.shape[0]*out_.shape[1])))
        
        # Check id col and time col without missing values
        assert out_[[id_col, time_col]].isna().sum().sum() == 0
        
        # Step 1: impute Forward fill
        out_ = out_.groupby(id_col).apply(lambda x: x.ffill(limit = fill_limit))
        out_.reset_index(drop = True, inplace = True)
        print("{:.2f} % missing values in data after FFill.".format(
            100 * out_.isna().sum().sum() / (out_.shape[0]*out_.shape[1])))
        
        # Step 2: impute backwards fill
        out_ = out_.groupby(id_col).apply(lambda x: x.bfill(limit = fill_limit))
        out_.reset_index(drop = True, inplace = True)
        print("{:.2f} % missing values in data after BFill.".format(
            100 * out_.isna().sum().sum() / (out_.shape[0]*out_.shape[1])))
        
        # Step 3: impute median fill
        out_ = out_.groupby(id_col).fillna(out_.median(axis = 0))
        out_ = out_.reset_index(drop = False).drop("level_1", axis = 1)
        print("{:.2f} % missing values in data after median.".format(
            100 * out_.isna().sum().sum() / (out_.shape[0]*out_.shape[1])))
        
    assert not out_.isna().any().any()
    assert out_.shape == data.shape

    return out_



def impute_time_length_mismatch_NaNs(data, mask_value = 0.0):
    """
    Impute NaN values with corresponding mask. Useful for tensorflow to ignore time length mismatches.
    
    data: np.array 3D
    mask_value: value to replace missing values with
    
    returns:
        3D array of same shape as data and with all remaining missing values imputed with mask.value
    """
    mask = np.isnan(data)
    
    if np.sum(np.isnan(data)) > 0:
        print('Not all values filled. Converting to tf friendly input.')
        
        data   = np.nan_to_num(data, copy = False, nan = mask_value)
        
    # Check model has been converted well
    assert np.any(np.isnan(data)) == False
    
    return data, mask




def check_data_loaded_correctly(X, y):
    """
    Check conditions for data loading

    Parameters
    ----------
    X : 3D numpy array
        Feature data
    y : 3D numpy array
        Target phenotypes

    Returns
    -------
    None - if all conditions for loading checkout.

    """

    # Check normalisation was correct
    cond1 = np.all(np.abs(np.amin(X, axis = 0)) < 1e-8)
    cond2 = np.all(np.abs(np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0) - 1) < 1e-8)
    
    assert cond1 and cond2
    
    return None



def load_from_csv(folder_path, X_name, y_name, time_range = (0, 72),  mode = "hours", feat_name = 'vitals', norm = "min-max"):
    
    """
    Import Data given csv file from a pandas dataframe. We check for feature selection, time-period selection and normalisation.
    
    Input:
        - folder_path: a path corresponding to the overall folder path
        - data_name: name of vital data or tuple of vital name and outcome name
        - time_range: Number of maximum days to outcome to consider. 
                Alternatively, a string with lower and upper bounds to consider (default: (0, 3))
                Unit must agree with mode.
                
        - mode: Time unit. One of "hours", "seconds", "days" or None.
        - feat_name: Set of features to consider. Vitals always considered (default: 'vitals')
        - norm: Method of normalisation. Either "Gaussian", "min-max" or None for no normalisation.
        
    
    - Data is loaded according to feature set and dtime_range.
    - Data is normalised
    
    
    - Return triple of arrays corresponding to vital signs and target outcomes/phenotypes and ids
    """
    # Check existence
    check_folder_path_exists(folder_path)
        
    # add csv to file
    X_path = folder_path + X_name + ".csv"
    y_path = folder_path + y_name + ".csv"
    
    # Import data
    try:
        X = pd.read_csv(X_path, parse_dates = ['charttime', 'hadm_end_time', 'hadm_start_time', 'event_time'])
        y = pd.read_csv(y_path, index_col = 0)
        
        print('Data {}-{} loaded successfully.'.format(X_name, y_name))
            
    except:
        print('Wrong data name specified')
        raise ValueError
    
    
    # Compute time to end of admission
    X['time_to_end'] = compute_time_to_end(X, id_col = "subject_id", time_column = "charttime", mode = mode).values
    
    # Check sorting
    assert X.subject_id.is_monotonic
    assert X.groupby("subject_id").filter(lambda x: not x["time_to_end"].is_monotonic_decreasing).empty
    
        
    # Select time_range
    if type(time_range) == tuple:
        start_, stop_ = time_range
        
    else:
        # Time Range only the upper bound. Lower bound assumed to be 0
        stop_  = time_range
        start_ = 0    
    
    # Select admission between start_ and stop_
    X_truncated= X[X['time_to_end'].between(start_,  stop_, inclusive=  True)]
    print('Admissions min-max times - ({} - {}) {} before an outcome'.format(start_, stop_, mode))
    
    
    # Subset to features
    X_id_feats      = X_truncated[['subject_id', 'time_to_end'] + feats_from_name(feat_name)]
    id_times        = X_truncated[['subject_id', 'hadm_id', 'time_to_end']]
    
    # Impute NaNs 
    X_id_feats_imp  = impute_regular_NaNs(X_id_feats)
    
    # Convert DataFrame to time series numpy array of 3 Dimensions
    x_feat_npy, id_diff_npy = convert_to_3darray(X_id_feats_imp, "subject_id", "time_to_end")
    
    # Average across batch for each time and input dimension
    x_feat_norm = normalise(x_feat_npy, norm)


    # Fill Na for value outside corresponding time step
    x_input, mask    = impute_time_length_mismatch_NaNs(x_feat_norm, mask_value = 0.0)

     
    # Load and import phenotypes
    y_data           = y[['Healthy', 'Death', 'ICU', 'Card']]
    y_input          = y_data.to_numpy()       
    
    # Check data loading conditions
    check_data_loaded_correctly(x_input, y_input)
    
    # print shape informaiton
    print("Data has been loaded correctly, with shapes: \n ({} - {})".format(x_input.shape, y_input.shape))
    
    return x_input.astype('float32'), y_input.astype('float32'), id_diff_npy, mask
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    