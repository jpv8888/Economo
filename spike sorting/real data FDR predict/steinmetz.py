# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:59:35 2022

@author: jpv88
"""

import JV_utils
import os

import numpy as np

from tqdm import tqdm

data_dir = 'D:/JPV/spikeAndBehavioralData/allData'

exps = [x[0] for x in os.walk(data_dir)]

clusters_str = 'spikes.clusters.npy'
times_str = 'spikes.times.npy'
goCue_times_str = 'trials.goCue_times.npy'

clusters = np.load(exps[1] + '/' + clusters_str)
times = np.load(exps[1] + '/' + times_str)
goCue_times = np.load(exps[1] + '/' + goCue_times_str)

# %%

N = max(clusters).item() + 1

spikes = [[] for _ in range(N)]

for i, spike in enumerate(tqdm(times)):
    clust = clusters[i].item()
    spikes[clust].append(spike.item())


end_t = max(times).item()
num_bins = round(3/0.05)
num_trials = len(goCue_times)

PSTHs = np.zeros((N, num_bins))
ISI_viol = np.zeros(N)

for i, clust in enumerate(tqdm(spikes)):
    
    trials = [[] for _ in range(len(goCue_times))]
    
    for idx, j in enumerate(goCue_times):
        trials_spikes = np.array(clust)[(np.array(clust) < j+1.5) & (np.array(clust) > j-1.5)]
        trials[idx] = np.array(trials_spikes) - j + 1.5
        
    viols = 0
    n_spikes = 0
    for trial in trials:
        n_spikes += len(trial)
        viols += sum(np.diff(trial) < 0.0025)
    
    ISI_viol[i] = viols/n_spikes if n_spikes != 0 else 0
        
    trials_flat = np.concatenate(trials, axis=0)
    PSTH = JV_utils.spikes_to_firing_rates(trials_flat, num_trials, T=3, N=num_bins)
    PSTHs[i,:] = PSTH
    
np.save(exps[1] + '/' + 'PSTHs.npy', PSTHs)
    


