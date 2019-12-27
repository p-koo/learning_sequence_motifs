from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd
import helper

#------------------------------------------------------------------------------------------------

arid3 = ['MA0151.1', 'MA0601.1', 'PB0001.1']
cebpb = ['MA0466.1', 'MA0466.2']
fosl1 = ['MA0477.1']
gabpa = ['MA0062.1', 'MA0062.2']
mafk = ['MA0496.1', 'MA0496.2']
max1 = ['MA0058.1', 'MA0058.2', 'MA0058.3']
mef2a = ['MA0052.1', 'MA0052.2', 'MA0052.3']
nfyb = ['MA0502.1', 'MA0060.1', 'MA0060.2']
sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3']
srf = ['MA0083.1', 'MA0083.2', 'MA0083.3']
stat1 = ['MA0137.1', 'MA0137.2', 'MA0137.3', 'MA0660.1', 'MA0773.1']
yy1 = ['MA0095.1', 'MA0095.2']

motifs = [[''],arid3, cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1]
motifnames = [ '','arid3', 'cebpb', 'fosl1', 'gabpa', 'mafk', 'max', 'mef2a', 'nfyb', 'sp1', 'srf', 'stat1', 'yy1']

#----------------------------------------------------------------------------------------------------


all_models = ['cnn_1', 'cnn_2', 'cnn_4', 'cnn_10', 'cnn_25', 'cnn_50', 'cnn_100',
              'cnn_50_2', 'cnn9_4', 'cnn9_25', 'cnn3_50', 'cnn3_2', 'cnn_2_1',
              'cnn_25_60', 'cnn_25_90', 'cnn_25_120']


num_trials = 5

# get performance statistics
mean_roc_trial = {}
mean_pr_trial = {}

for trial in range(num_trials):
    
    results_path = os.path.join('../results', 'synthetic_'+str(trial), 'results.tsv')
    df = pd.read_csv(results_path, delimiter='\t')

    if trial == 0:
        for i, model in enumerate(df['model']):
            mean_pr_trial[model] = []
            mean_roc_trial[model] = []
    ave_roc = df['ave roc']
    ave_pr = df['ave pr']

    tmp_roc = []
    tmp_pr = []
    for i, model in enumerate(df['model']):
        mean,_,std = ave_pr[i].split('$')
        mean_pr_trial[model].append(float(mean))
        
        mean,_,std = ave_roc[i].split('$')
        mean_roc_trial[model].append(float(mean))


print('Synthetic results')
for model_name in all_models:
    trial_match_any = []
    trial_qvalue = []
    trial_match_fraction = []
    trial_coverage = []
    for trial in range(num_trials):

        # save path
        results_path = os.path.join('../results', 'synthetic_'+str(trial))
        save_path = os.path.join(results_path, 'conv_filters')

        file_path = os.path.join(save_path, model_name, 'tomtom.tsv')
        best_qvalues, best_match, min_qvalue, match_fraction  = helper.match_hits_to_ground_truth(file_path, motifs)
            
        # store results
        trial_qvalue.append(min_qvalue)
        trial_match_fraction.append(match_fraction)
        trial_coverage.append((len(np.where(min_qvalue != 1)[0])-1)/12) # percentage of motifs that are covered
        df = pd.read_csv(os.path.join(file_path), delimiter='\t')
        trial_match_any.append((len(np.unique(df['Query_ID']))-3)/30) # -3 is because new version of tomtom adds 3 lines of comments under Query_ID 

    print("%s & %.3f$\pm$%.3f &  %.3f$\pm$%.3f &  %.3f$\pm$%.3f  \\\\"%(model_name, 
                                                  np.mean(mean_roc_trial[model_name]),
                                                  np.std(mean_roc_trial[model_name]),
                                                  np.mean(trial_match_any), 
                                                  np.std(trial_match_any),
                                                  np.mean(trial_match_fraction), 
                                                  np.std(trial_match_fraction) ) )




