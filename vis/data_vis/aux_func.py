#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 March 2021

Author: Henrique Aguiar
you can reach me at henrique.aguiar@eng.ox.ac.uk
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter

import os, sys


frailty = pd.read_csv("../data-final/processed-clinical/frailty.csv", index_col = 0)
co2     = pd.read_csv("../data-final/processed-clinical/co2.csv", index_col = 0)
icd10   = pd.read_csv("../data-final/processed-clinical/icd10.csv", index_col = 0)
meds    = pd.read_csv("../data-final/processed-clinical/medications.csv", index_col = 0)
t2rf    = pd.read_csv("../data-final/processed-clinical/t2rf.csv", index_col = 0)


assert np.sum(frailty.index.values != co2.index.values) + np.sum(frailty.index.unique() != t2rf.index.values) == 0
assert np.sum(frailty.index.values != icd10.index.unique()) == 0
assert np.sum(frailty.index.values != meds.index.unique()) == 0




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
                                                                                                                                                        
    


def cluster_outcome_dist(patients_cluster_):
    
    "patients_cluster_ is a dataframe with one-hot encoding outcomes for patients, and corresponding to a set of patients assigned to the same model cluster"    
    
    cluster_size_ = patients_cluster_.shape[0]
    
    feats_ = ['no event', 'death', 'cardiac', 'icu']
    cluster_sum_  = patients_cluster_.loc[:, feats_].sum()
    
    return cluster_sum_ / cluster_size_
    
    
    

def plot_outcome_dist(model_clus_):    
    
    """model_clus_: dataframe with index given by subject_id, 
                a one-hot encoding of each patient's outcome',
                a column "clus" with resulting model cluster.
    """
    plt.ioff()      # Don't display figures
    
    assert model_clus_.iloc[:, :-1].sum(axis= 1).eq(1).sum() == model_clus_.shape[0]
    
    "groupby model cluster"
    dist_ = model_clus_.reset_index().groupby(by = 'clus').apply(lambda x: cluster_outcome_dist(x))
    size_ = model_clus_.reset_index().groupby(by = 'clus').apply(lambda x: x.shape[0])
    list_clusters_ = model_clus_.clus.unique().tolist()
    
    nrows = int(np.ceil(np.sqrt(len(list_clusters_))))
    ncols = int(np.ceil(len(list_clusters_) /nrows))
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = 'col', sharey = 'row',
                           figsize = (8.27, 11.69), frameon = True)
    ax = ax.reshape(-1)
    
    colors = get_cmap('tab10').colors
    outcomes = ['no event', 'death', 'cardiac', 'icu']
    
    ax_id_ = 0
    for clus_ in list_clusters_:
    
        ax[ax_id_].bar(outcomes, dist_.loc[clus_], color = colors[list_clusters_.index(clus_)])
        ax[ax_id_].set_title('Cluster {} - N= {}'.format(clus_, size_.loc[clus_]))
        ax_id_ += 1
        
    ax[0].set_ylabel('Fraction of Cluster Size')
    
    return fig, ax
    
    
    
    
def quantiles(data, feature):
    
    "print 3 quantiles in Median (IQR) format for feature column in data df"
    data_ = data[feature]
    
    median= data_.median()
    iqr25 = data_.quantile(0.25)
    iqr75 = data_.quantile(0.75)
    
    return median, iqr25, iqr75

def binary_info(data, feature):
    
    "print count, proportion of binary data column"
    assert data[feature].nunique() <= 2
    
    if feature == 'gender':
        try:
            data_ = data[feature].replace(to_replace = ['M', 'F'], value = [1, 0])
        except:
            data_ = data[feature]
        
    else:
        data_ = data[feature]
        
    count = data_.sum().astype(int)
    proportion = data_.sum() / data_.shape[0]
    
    return count, proportion
    
    
def cohort_information(cohort, 
                       clusters, 
                       by = None, 
                       with_t2rf    = None,  
                       with_meds    = None, 
                       with_co2     = None,
                       with_frailty = None,
                       with_icd10   = None, 
                       outcomes_with_priors = None,
                       outcomes_row= None):

    """function to determine static (cohort) information for patient clusters as determined by clusters[by] column. If by is None, we assume clusters is "outcomes", and we treat clusters as directly defined by their corresponding outcomes.
    
    
    Input: cohort is a dataframe with index subject_id of size 24437.
    It has columns gender, age, elective_admission, surgical_admisison, cci and potentially others.
    
    
    clusters: if by is not NONE, then this is a dataframe with index subject_id of size 24437. 
    It has columns with corresponding outcomes, and a column "by" which assigns each patient to a specific cluster.
    
    by - columns for which to consider patient clusterings. If None, we assume clusters has columns specifying outcomes in a one-hot format.
    """
    assert np.sum(cohort.index.values != clusters.index.values) == 0    

    
    if by is not None:
        clus_ = clusters[by]
        name_ = 'clusters'
    
    else:
        clus_ = clusters.idxmax(axis = 1)    
        name_ = 'outcomes'
        assert clusters.sum(axis = 1).eq(1).all()
        
    list_outcomes_ = ['no event', 'death', 'icu', 'cardiac']
    list_clusters_ = clus_.unique().tolist()
    output_df_     = pd.DataFrame(data = np.nan, index = ['N='], columns = list_clusters_)

    if (outcomes_with_priors is not None or outcomes_row is not None) and (by is not None):
        
        joint_clus_out_  = pd.DataFrame(data = 0.0, index = list_outcomes_, columns = clus_.sort_values().unique().tolist())
        
        # Need to compute Prob(C | outcome)
        for out_ in list_outcomes_:
            for cluster_ in list_clusters_:
                num_pats_ = clusters[clusters[out_].eq(1) & clusters[by].eq(cluster_)].sum().loc[out_]
                
                joint_clus_out_.loc[out_, cluster_] = num_pats_
                
        # Normalise
        joint_clus_out_ = joint_clus_out_ / joint_clus_out_.sum().sum()

        assert joint_clus_out_.sum(axis = 1).all()
        assert joint_clus_out_.sum(axis = 0).all()
        
    
    for cluster in list_clusters_:

        # Subset to patietn cohort in a certain cluster
        data = cohort[clus_ == cluster]
        output_df_.loc['N=', cluster] = data.shape[0]
        
        for feat_ in ['age', 'cci']:
            
            # Obtain quantiles
            med, iqr1, iqr2 = quantiles(data, feat_)
            feat_iqr_  = "{:.1f} ({:.1f} - {:.1f})".format(med, iqr1, iqr2)
            
            output_df_.loc[feat_, cluster] = feat_iqr_
            
        for feat_ in ['gender']:
            
            count, proportion = binary_info(data, feat_)
            feat_bin_  = "{} ({:.2f} %)".format(count, 100* proportion)
            id_ = y_label(feat_)
            
            output_df_.loc[id_, cluster] = feat_bin_
        
        for feat_ in ['elective_admission', 'surgical_admission']:
            
            count, proportion = binary_info(data, feat_)
            feat_bin_  = "{} ({:.2f} %)".format(count, 100* proportion)
            opp_feat_bin_ = "{} ({:.2f} %)".format(data.shape[0] - count, 100 * (1 - proportion))
            
            id_ , opp_id_  = y_label(feat_), 'No {}'.format(y_label(feat_))
            
            
            output_df_.loc[id_, cluster]     = feat_bin_
            output_df_.loc[opp_id_, cluster] = opp_feat_bin_
            
        
        if by is not None:
            
            # If by is not None, i.e. this is cluster result want also outcome and Outcome distribution
            for feat_ in list_outcomes_:
                
                # Load correct data
                data_ = clusters[clus_ == cluster]
                
                if outcomes_with_priors is not None:
                    
                    # Priors
                    list_outcomes_ = ['no event', 'death', 'icu', 'cardiac']
                    outcome_cts_   = clusters.sum()[list_outcomes_]
                    priors_        = outcome_cts_ / outcome_cts_.sum()
                    
                    # Divided by prior
                    weighted_out_given_clus = data_.sum().loc[list_outcomes_] / priors_
                    
                    outcomes_dist_ = "{} (Posterior {:.2f} %)".format(
                                data_.sum()[feat_], 100 * weighted_out_given_clus[feat_] / weighted_out_given_clus.sum())
                    
                    
                elif outcomes_row is not None and outcomes_with_priors is None:
                
                    # normalised joint probability
                    joint_prob_ = joint_clus_out_.copy(deep = True)
                    outcomes_dist_ = "{} (Row {:.2f} %)".format(
                                data_.sum()[feat_], 100 * joint_prob_.loc[feat_, cluster]/joint_prob_.loc[feat_, :].sum())
                
                else:                 
                    outcomes_dist_ = "{} ({:.2f} %)".format(
                                data_.sum()[feat_], 100 * data_.sum()[feat_] / data_.shape[0])
                
                output_df_.loc[feat_, cluster] = outcomes_dist_
                
        
        if with_frailty is not None:
            
            # Print frailty metrics
            frailty_clus_ = frailty.loc[data.index]
            
            med, iqr1, iqr2  = quantiles(frailty_clus_, 'frailty_score') 
            frailty_nan      = frailty_clus_.isna().sum()
            
            feat_iqr_        = "{:.1f} ({:.1f} - {:.1f})".format(med, iqr1, iqr2)
            
            output_df_.loc['frailty', cluster] = feat_iqr_
            output_df_.loc['NAs frailty', cluster] = frailty_nan.values[0]
            
        if with_co2 is not None:
            
            # Print CO2 metrics
            co2_clus_        = co2.loc[data.index]
            
            med, iqr1, iqr2  = quantiles(co2_clus_, 'CO2')
            co2_nan          = co2_clus_.isna().sum()
            
            feat_iqr_        = "{:.1f} ({:.1f} - {:.1f})".format(med, iqr1, iqr2)
            
            output_df_.loc['CO2', cluster] = feat_iqr_
            output_df_.loc['NAs CO2', cluster] = co2_nan.values[0]
            
        if with_t2rf is not None:
            
            # Print T2RF table metrics
            t2rf_clus_   = t2rf.loc[data.index]
            
            count_pres, prop_pres = binary_info(t2rf_clus_, 'pres')
            count_diag, prop_diag = binary_info(t2rf_clus_, 'diag')
            
            feat_pres_   = "{} ({:.2f} %)".format(count_pres, 100* prop_pres)
            feat_diag_   = "{} ({:.2f} %)".format(count_diag, 100* prop_diag)
            
            output_df_.loc['T2RF Pres', cluster] = feat_pres_
            output_df_.loc['T2RF Diag', cluster] = feat_diag_
            
        if with_icd10 is not None:
            
            # Print most common ICD10 codes
            icd10_clus_  = icd10.loc[data.index]
            icd10_cluscts= icd10_clus_.diagnosis_code.value_counts()
            
            # iterate through top icd10 codes
            for pos_ in range(5):
                
                code_, count_ = icd10_cluscts.index[pos_], icd10_cluscts.iloc[pos_]
                
                # Update for output_df
                output_df_.loc['Top {} ICD10'.format(pos_), cluster] = "{} (n = {})".format(code_, count_)
        
        if with_meds is not None:
                
            # Print most common medications
            meds_clus_     = meds.loc[data.index]
            
            # Need to select only one medication per patient
            meds_clus_.index.name = 'subject_id'
            meds_count_per_pat_ = meds_clus_.reset_index(drop = False).drop_duplicates(keep= 'first')
            
            meds_cluscts_  = meds_count_per_pat_.set_index('subject_id').catalog_cd_display.value_counts()
            
            # iterate through top medications
            for pos_ in range(10):
                med_, count_ = meds_cluscts_.index[pos_], meds_cluscts_.iloc[pos_]
                
                if 'hartmann\'s' in med_:
                    med_ = 'Sod Lact (Hartmann\')'
                
                # Update for output_df
                output_df_.loc['Top {} Med'.format(pos_), cluster] = "{} (n = {})".format(med_, count_)

    
    return output_df_, name_
                


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
                                                                                                                                                        
    
    
    
    
        
    
    
    
    
    
    
    
    
    
