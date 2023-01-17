# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:19:11 2023

@author: jpv88
"""
from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np

from os import listdir
from os.path import isfile, join
# %%

mypath = r"C:\\Users\\jpv88\\OneDrive\\Documents\\GitHub\\Economo\\spike sorting\\"

file = 'sub-699733573_ses-715093703_probe-810755797_ecephys.nwb'

nwb_read = NWBHDF5IO(mypath + file, "r").read()

# %%

spike_times = nwb_read.units.spike_times.data[:]
spike_times_index = nwb_read.units.spike_times_index.data[:]
cue_times = nwb_read.trials['cue_times'].data[:]

n_units = len(spike_times_index)
spike_times_index = np.insert(spike_times_index, 0, 0)

units = []

for idx in range(n_units):
    idx1 = spike_times_index[idx]
    idx2 = spike_times_index[idx+1]
    
    units.append(spike_times[idx1:idx2])
    
# length of time after go cue to use for PSTH
T = 2

ISI_viol_temp = []
aligned_units = []
for unit in units:
    aligned_spikes = []
    viols = 0
    spikes = 0
    for cue in cue_times:
        temp = unit - cue
        temp = temp[(temp <= T) & (temp >= 0)]
        n_spikes = len(temp)
        viols += sum(np.diff(temp) < 0.0025)
        spikes += n_spikes
        aligned_spikes.extend(temp)
        
    aligned_units.append(aligned_spikes)
    ISI_viol_temp.append(viols/spikes)
    
    
bin_size = 50
PSTHs_temp = []
for unit in aligned_units:
    PSTHs_temp.append(JV_utils.gen_PSTH(unit, 265, T, bin_size))
    
pred = JV_utils.pred_FDR(np.stack(PSTHs_temp), ISI_viol_temp)

FDRs.extend(pred)
PSTHs.extend(PSTHs_temp)
ISI_viol.extend(ISI_viol_temp)
FDR_avg.append(np.mean(pred))