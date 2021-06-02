import pandas as pd 
import numpy as np

import os
import datetime as dt
import tqdm

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter

# Load extra data
frailty = pd.read_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/clinical/frailty.csv", index_col = 0)
co2     = pd.read_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/clinical/co2.csv", index_col = 0)
icd10   = pd.read_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/clinical/icd10.csv", index_col = 0)
meds    = pd.read_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/clinical/medications.csv", index_col = 0)
t2rf    = pd.read_csv("/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/clinical/t2rf.csv", index_col = 0)


def _add_time_to_outcome(df):

    assert 'time_to_outcome' not in df.columns.tolist()
    
    # Compute time to outcome
    time_to_out_ = df.groupby("subject_id").apply(lambda x: x.charttime.max() - x.charttime)
    
    df_ = df.copy(deep = True)
    
    # Add to dataframe
    df_['time_to_outcome'] = time_to_out.values
    
    return df_

def y_label(feature):
    
    if feature == 'gender':
        label_ = 'Male'
        
    elif feature == 'elective_admission':
        label_ = 'Ele Adm'
        
    elif feature == 'surgical_admission':
        label_ = 'Sur Adm'
    
    elif feature == 'HR':
        label_ = 'HR (bpm)'
        
    elif feature == 'RR':
        label_ = 'RR (bpm)'
        
    elif feature == 'DBP':
        label_ = 'DBP (bpm)'
        
    elif feature == 'SBP':
        label_ = 'SBP (bpm)'
        
    elif feature == 'SPO2':
        label_ = 'SPO2 (%)'
    
    elif feature == 'TEMP':
        label_ = 'TEMP (ºC)'
        
    elif feature == 'AVPU':
        label_ = 'AVPU (N)'
        
    elif feature == 'FIO2':
        label_ = 'FIO2 (%)'
        
    elif feature == 'ALB':
        label_ = 'ALB (N)'
        
    elif feature == 'CR':
        label_ = 'CR (mg/L)'
        
    elif feature == 'HGB':
        label_ = 'HGB (N)'
        
    elif feature == 'POT':
        label_ = 'POT (mmol/L)'
    
    elif feature == 'SOD':
        label_ = 'SOD (mmol/L)'
        
    elif feature == 'UR':
        label_ = 'UR (mL)'
        
    elif feature == 'WBC':
        label_ = 'WBC (N)'
        
    elif feature == 'EOS':
        label_ = 'EOS (N/L)'
        
    elif feature == 'NEU':
        label_ = 'NEU (N/L)'
        
    elif feature == 'LYM':
        label_= 'LYM (N/L)'
        
    elif feature == 'BAS':
        label_ = 'BAS (N/L)'
        
    elif feature == 'CRP':
        label_ = 'CRP (mg/L)'
        
    elif feature == 'EBR':
        label_ = 'EBR' 
        
    elif feature == 'NLR':
        label_ = 'NLR'
    
    elif feature == 'O2 ratio':
        label_ = 'num'
        
    else:
        label_ = feature
        print(feature)
    
    return label_
    
def compute_moments(data, feature):
    
    "Compute moments of feature in data in descending according to time_to_outcome"
    if feature == 'O2 ratio':
        mean_ = data.groupby('time_to_outcome').apply(lambda x: (x['SPO2'] / x['FIO2']).mean())
        sterr_= data.groupby('time_to_outcome').apply(lambda x:
                                                      (x['SPO2'] / x['FIO2']).std() // np.sqrt(data.subject_id.nunique()))
            
    elif feature == 'is_elec':
        mean_, sterr_ = compute_moments(data, 'is_elec')

    elif feature == 'is_surg':
        mean_, sterr_ = compute_moments(data, 'is_surg')
        
    elif feature == 'gender' or feature == 'Male':
        mean_, sterr_ = compute_moments(data, 'gender')
        
    else:        
        mean_ = data.groupby('time_to_outcome').apply(lambda x: 
                                                x[feature].mean())
        sterr_= data.groupby('time_to_outcome').apply(lambda x:
                                                x[feature].std() // np.sqrt(data.subject_id.nunique()))
        
    mean_.sort_index(ascending = False, inplace = True)
    sterr_.sort_index(ascending = False, inplace = True)
        
    return mean_, sterr_

def plot_means(clusters, obvs_, features, by = None):
    
    """
    clusters: dataframe with subject_id as index, and contains one-hot encoding of the original outcomes.
    We assume it has a column "clus" which assigns a cluster assignment to each cluster.
    Should this not be the case, the assumption is that Clusters does not have this column, and 
    we are PLOTTING MEANS PER OUTCOME Group.
                                                                                                                                                                       
                                                                                                                                                                        
    obvs_: dataframe with subject_ids corresponding to subject_id of model_clus, and with temporal data for these patients.                                                                                                                                                                 
    
    feature: column of obvs_
    """
    plt.ioff()      # Don't display figures
    
    assert np.sum(clusters.index.values != obvs_.subject_id.unique()) == 0
    clusters_ = clusters.copy(deep = True)
    
    # Select number of clusters to iterate through
    if by is not None: # If clus (or whatever name feature us saved under exists), pick those clusters
        list_clusters_ = clusters_[by].unique().tolist()
        name_ = "Clusters"
        
    else:       # If clus does not exist, select outcomes columns (idxmax over the row)
        clusters_.loc[:, 'clus'] = clusters_.idxmax(axis = 1)
        list_clusters_ = clusters_.clus.unique().tolist()
        name_ = "Outcome Groups"
        
    
    if type(features) == str:        # We are only selecting 1 feature
        
        # Initialise figure
        fig, ax = plt.subplots(frameon = True)
        colors  = get_cmap(name = "tab10").colors
        
        if by is None:     # If no cluster assignment, we are using outcomes, with custom colors                 
            colors = colors[2], colors[7], colors[0], colors[3]
            
        # Iterate through each cluster
        for cluster in list_clusters_:
                                                                                                                                                                            
            # Subset to data within corresponding cluster/group
            patients_in_cluster_ = clusters_[clusters_.clus.eq(cluster)].index
            data_cluster_ = obvs_[obvs_.subject_id.isin(patients_in_cluster_)]
            
            assert data_cluster_.subject_id.nunique() == patients_in_cluster_.size
            
            
            # Compute Time Index and Moments (Mean and Standard Deviation)
            time_idx = data_cluster_.loc[:, 'time_to_outcome'].value_counts().sort_index(
                ascending = False).index.values // (3600 * 10**9)     # Time index descending in hours.
            
            clus_mean_, clus_sterr_ = compute_moments(data_cluster_, features)
            clus_mean_, clus_sterr_ = compute_moments(data_cluster_, features)
            color = colors[list_clusters_.index(cluster)]
            
            
            "Plot mean with standard deviation"
            ax.plot(time_idx[6:], clus_mean_[6:], color = color, linestyle = '-', label = '{} - N= {}'.format(cluster, patients_in_cluster_.size))
            ax.plot(time_idx[6:], clus_mean_[6:] + clus_sterr_[6:], alpha = 0.8, color = color, linestyle = '--')
            ax.plot(time_idx[6:], clus_mean_[6:] - clus_sterr_[6:], alpha = 0.8, color = color, linestyle = '--')
            
        ax.set_xlabel('Time to Outcome (hours)')
        ax.set_ylabel(y_label(features))
        
        ax.invert_xaxis()
        
        ax.set_title('Mean trajectories of {}'.format(features))
        plt.legend(title = name_)
    
    elif type(features) == list:      # We are selecting multiple features
    
        # Initialise figure
        N = len(features)
        nrows, ncols = int(np.ceil(N/2)), 2
        
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = 'col', sharey = False, frameon = True,
                               figsize = (11.69, 8.27))
        ax      = ax.reshape(-1)
        colors = get_cmap(name = 'tab10').colors
        
        if by is None:
            colors = colors[2], colors[7], colors[0], colors[3]
        
        for feature in features:      #Iterate through features
            
            ax_id_ = features.index(feature)
            labels_= []
            
            # Iterate through clusters
            for cluster in list_clusters_:
                
                # Subset to data within corresponding cluster/group
                patients_in_cluster_ = clusters_[clusters_.clus.eq(cluster)].index
                data_cluster_ = obvs_[obvs_.subject_id.isin(patients_in_cluster_)]
                
                if not '{} - N= {}'.format(cluster, patients_in_cluster_.size) in labels_:
                    
                    labels_ = labels_ + list('{} - N= {}'.format(cluster, patients_in_cluster_.size))
                    
                assert data_cluster_.subject_id.nunique() == patients_in_cluster_.size
                
                
                # Compute Time Index and Moments (Mean and Standard Deviation)
                time_idx = data_cluster_.loc[:, 'time_to_outcome'].value_counts().sort_index(
                    ascending = False).index.values // (3600 * 10**9)     # Time index descending in hours.
                
                clus_mean_, clus_sterr_ = compute_moments(data_cluster_, feature)
                clus_mean_, clus_sterr_ = compute_moments(data_cluster_, feature)
                color = colors[list_clusters_.index(cluster)]
                
                
                "Plot mean with standard deviation"
                ax[ax_id_].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax[ax_id_].plot(time_idx[6:], clus_mean_[6:], color = color, linestyle = '-', label = '{} - N= {}'.format(cluster, patients_in_cluster_.size))
                ax[ax_id_].plot(time_idx[6:], clus_mean_[6:] + clus_sterr_[6:], alpha = 0.8, color = color, linestyle = '--')
                ax[ax_id_].plot(time_idx[6:], clus_mean_[6:] - clus_sterr_[6:], alpha = 0.8, color = color, linestyle = '--')
                
            ax[ax_id_].set_ylabel(feature, labelpad=5)
            
        ax[0].invert_xaxis()
        ax[0].set_xlabel('Time to Outcome (hours)')
        ax[1].invert_xaxis()
        ax[1].set_xlabel('Time to Outcome (hours)')
        
        fig.tight_layout()

    return fig, ax
    
    
def _info_binary(series):
    
    # Check conditions for series
    assert np.all(np.isin(series.unique(),[0,1]))
    
    "Compute relevant information for a binary pandas series. Computes counts, proportion and number of binary scenarios."
    counts   = series.size
    has_feat = series.sum()
    prop     = has_feat / counts
    
    # Convert info to str
    str_     = "{} ({:.2f} %)".format(int(has_feat), prop * 100)
    
    return str_

def _info_cat(series):
    
    "Compute relevant information for categorical pandas series. Computes max, min, IRQ, median."
    quantiles = series.quantile(q = [0.25, 0.50, 0.75]).astype(int)
    
    # Convert info to str
    str_      = "{} ({} - {})".format(quantiles.loc[0.50], quantiles.loc[0.25], quantiles.loc[0.75])
    
    return str_
    
def _into_cont(series):
    
    "Compute relevant information for continuous data."
    description_ = series.describe()
    
    # Convert info to str
    str_      = "{:.1f} ({:.1f} - {:.1f})".format(description_.loc["50%"], description_.loc["25%"], description_.loc["75%"])
    
    return out_
    
def print_static_summary(df):
    
    "Compute static information for each feature in static dataset. "
    
    # Compute list of features
    unique_values = df.nunique()
    output_tbl    = pd.Series(data = np.nan, index = ["n"] + unique_values.index.tolist())
    
    # Update count
    output_tbl.loc["n"] = df.shape[0]
    
    for feat_ in unique_values.index:
        if unique_values.loc[feat_] > 2:
            feat_info_ = _info_cat(df[feat_])  
        else:
            feat_info_ = _info_binary(df[feat_])     
            
        # Update output dic
        output_tbl.loc[feat_] = feat_info_
    
    return output_tbl


def print_static_summary_by_outcome(df, outcomes):
    
    "Compute static information for each feature separated by static dataset."
    
    # Compute indices
    lst_outcomes_ = outcomes.columns.values
    lst_feats_    = df.columns.values.tolist()
    
    # Initialise output
    output_tbl    = pd.DataFrame(data = np.nan, index = ["n"] + lst_feats_, columns = lst_outcomes_)
    
    for outcome_ in lst_outcomes_:
        
        # Subset data to specific cohort
        pats_subset_ = outcomes[outcomes[outcome_] == 1].index
        cohort_sub_  = df.loc[pats_subset_, :]
        
        # Update output table
        output_tbl.loc[:, outcome_] = print_static_summary(cohort_sub_)

    return output_tbl

def compute_LOS_by_outcome(df, log_scale = False):
    
    "Compute Length of stay with a separate column tracking event type"
    
    # Compute Length of Stay
    hadm_los_ = df.groupby("subject_id").apply(lambda x: x['hadm_end_time'].iloc[0] - x['hadm_start_time'].iloc[0])
    
    # compute output_df
    output_tbl= df.groupby("subject_id").apply(lambda x: x["event_type"].iloc[0]).to_frame(name = "event_type")
    
    # add length of stay
    output_tbl['los (d)'] = hadm_los_.dt.total_seconds().divide(24*3600)
    
    # Conver to log scale if true
    if log_scale == True:
        output_tbl["los (d)"] = np.log(output_tbl.loc[:, "los (d)"].values)
    
    return output_tbl
    
def compute_LOV_by_outcomes(df, log_scale = False):
    
    "Compute Length of stay with a separate column tracking event type"
    
    # Compute Length of Stay
    hadm_los_ = df.groupby("subject_id").apply(lambda x: x['charttime'].max() - x['charttime'].min())
    
    # compute output_df
    output_tbl= df.groupby("subject_id").apply(lambda x: x["event_type"].iloc[0]).to_frame(name = "event_type")
    
    # add length of stay
    output_tbl['los (d)'] = hadm_los_.dt.total_seconds().divide(24*3600)
    
    # Conver to log scale if true
    if log_scale == True:
        output_tbl["los (d)"] = np.log(output_tbl.loc[:, "los (d)"].values)
    
    return output_tbl

    
def compute_samples(data, feature, num_pats = 10):
    
    """Compute samples of feature in data in descending according to time_to_outcome
    
    Returns a numpy array of size Num_pats x time_lengths"""
    
    import random 
    sel_pats_ = random.sample(list(data.subject_id.unique()), num_pats)
    sample_data_ = np.zeros(shape = (num_pats, data.time_to_outcome.nunique()))
    
    for pat_ in sel_pats_:
        
        " Select corresponding observations"
        pat_data_ = data[data.subject_id.eq(pat_)]
        id_   = sel_pats_.index(pat_)
        
        if feature == 'O2 ratio':
            fio2_ = pat_data_.sort_values(by = 'time_to_outcome', ascending = False)['FIO2']
            spo2_ = pat_data_.sort_values(by = 'time_to_outcome', ascending = False)['SPO2']
                        
            traj_ = spo2_ / fio2_
            
        else: 
            traj_ = pat_data_.sort_values(by = 'time_to_outcome', ascending = False)[feature]
    
        sample_data_[id_, -traj_.size:] = traj_.values
    
    return sample_data_, sel_pats_
    
def plot_samples(clusters, obvs_, features, by = None, num_pats = 10):
    
    """
    clusters: dataframe with subject_id as index, and contains one-hot encoding of the original outcomes.
    We assume it has a column "clus" which assigns a cluster assignment to each cluster.
    Should this not be the case, the assumption is that Clusters does not have this column, and 
    we are PLOTTING MEANS PER OUTCOME Group.
                                                                                                                                                                       
                                                                                                                                                                        
    obvs_: dataframe with subject_ids corresponding to subject_id of model_clus, and with temporal data for these patients.                                                                                                                                                                 
    
    feature: column of obvs_
    """
    plt.ioff()      # Don't display figures
    
    assert np.sum(clusters.index.values != obvs_.subject_id.unique()) == 0
    clusters_ = clusters.copy(deep = True)
    
    # Select number of clusters to iterate through
    if by is not None: # If clus (or whatever name feature us saved under exists), pick those clusters
        list_clusters_ = clusters_[by].unique().tolist()
        name_ = "Clusters"
        
    else:       # If clus does not exist, select outcomes columns (idxmax over the row)
        clusters_.loc[:, 'clus'] = clusters_.idxmax(axis = 1)
        list_clusters_ = clusters_.clus.unique().tolist()
        name_ = "Outcome Groups"
        
    
    if type(features) == str:        # We are only selecting 1 feature
        
        # Initialise figure
        fig, ax = plt.subplots(frameon = True)
        if len(list_clusters_) > 10:
            colors  = get_cmap(name = "tab20").colors
        else:
            colors = get_cmap('tab10').colors
        
        # For each sample, assign color given by outcome and label with cluster 
        if by is None:                
            colors = colors[2], colors[7], colors[3], colors[0]
            
        # Iterate through each cluster
        for cluster in list_clusters_:
                                                                                                                                                                            
            # Subset to data within corresponding cluster/group
            patients_in_cluster_ = clusters_[clusters_.clus.eq(cluster)].index
            data_cluster_ = obvs_[obvs_.subject_id.isin(patients_in_cluster_)]
            
            assert data_cluster_.subject_id.nunique() == patients_in_cluster_.size
            
            
            # Compute Time Index and Moments (Mean and Standard Deviation)
            time_idx = data_cluster_.loc[:, 'time_to_outcome'].value_counts().sort_index(
                ascending = False).index.values // (3600 * 10**9)     # Time index descending in hours.
            
            clus_sample_, pats_ = compute_samples(data_cluster_, features, num_pats)            
            
            "Plot sample with color given by outcome"
            for pat_ in pats_:
                
                out_dic = {'no event': 'H', 'death': 'D', 'cardiac': 'C', 'icu': 'I' }
                id_, out_ = pats_.index(pat_), pd.to_numeric(clusters_.loc[pat_, ['no event', 'death', 'cardiac', 'icu']]).idxmax(axis = 1)
                color_ = colors[list_clusters_.index(cluster)]
                
                mask_ = clus_sample_[id_, 6:]!= 0
                ax.plot(time_idx[6:][mask_], clus_sample_[id_, 6:][mask_], color = color_, linestyle = '-', label = '{}-{}-{} Clu: {}'.format(int(pat_), out_dic[out_], cluster, patients_in_cluster_.size))

            
        ax.set_xlabel('Time to Outcome (hours)')
        ax.set_ylabel(y_label(features))
        
        ax.invert_xaxis()
        
        ax.set_title('Sample trajectories of {}'.format(features))
        plt.legend(title = name_)
    
    elif type(features) == list:      # We are selecting multiple features
    
        # Initialise figure
        N = len(features)
        nrows, ncols = int(np.ceil(N/2)), 2
        
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = 'col', sharey = False, frameon = True,
                               figsize = (11.69, 8.27))
        ax      = ax.reshape(-1)
        if len(list_clusters_) > 10:
            colors  = get_cmap(name = "tab20").colors
        else:
            colors = get_cmap('tab10').colors
        
        if by is None:
            colors = colors[2], colors[7], colors[3], colors[0]
        
        for feature in features:      #Iterate through features
            
            ax_id_ = features.index(feature)
            labels_= []
            
            # Iterate through clusters
            for cluster in list_clusters_:
                
                # Subset to data within corresponding cluster/group
                patients_in_cluster_ = clusters_[clusters_.clus.eq(cluster)].index
                data_cluster_ = obvs_[obvs_.subject_id.isin(patients_in_cluster_)]
            
                if not '{} - N= {}'.format(cluster, patients_in_cluster_.size) in labels_:
                    labels_ = labels_ + list('{} - N= {}'.format(cluster, patients_in_cluster_.size))
                    
                assert data_cluster_.subject_id.nunique() == patients_in_cluster_.size
                
                
                # Compute Time Index and Moments (Mean and Standard Deviation)
                time_idx = data_cluster_.loc[:, 'time_to_outcome'].value_counts().sort_index(
                    ascending = False).index.values // (3600 * 10**9)     # Time index descending in hours.
                
                clus_sample_, pats_ = compute_samples(data_cluster_, feature, num_pats)
                
                "Plot sample with color given by outcome"
                for pat_ in pats_:
                    
                    out_dic = {'no event': 'H', 'death': 'D', 'cardiac': 'C', 'icu': 'I' }
                    id_, out_ = pats_.index(pat_), pd.to_numeric(clusters_.loc[pat_, ['no event', 'death', 'cardiac', 'icu']]).idxmax(axis = 1)
                    color_ = colors[list_clusters_.index(cluster)]
                    
                    mask_ = clus_sample_[id_, 6:]!= 0
                    ax[ax_id_].plot(time_idx[6:][mask_], clus_sample_[id_, 6:][mask_], color = color_, linestyle = '--', label = '{}-{}-{} Clu: {}'.format(int(pat_), out_dic[out_], cluster, patients_in_cluster_.size))

            ax[ax_id_].set_ylabel(feature, labelpad=5)
            
        ax[0].invert_xaxis()
        ax[0].set_xlabel('Time to Outcome (hours)')
        ax[1].invert_xaxis()
        ax[1].set_xlabel('Time to Outcome (hours)')
        
        fig.tight_layout()

        
    
    return fig, ax
 
    
    
    
    
