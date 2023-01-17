# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:33:48 2023

@author: jpv88
"""

from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np

# the final equation
def FDR_master(ISIviol, Rtot, Rout_unit, N, tau=2.5, tau_c=0):
    
    Rtot_mag = np.linalg.norm(Rtot)
    Rtot_avg = np.average(Rtot)
    Rtot_unit = Rtot/Rtot_mag
    
    tau = tau*(10**-3)
    n = len(Rtot)
    D = np.dot(Rtot_unit, Rout_unit)

    if N == float('inf'):
        N_correction1 = 1
        N_correction2 = 1
    else:
        N_correction1 = N/(N + 1)
        N_correction2 = (N + 1)/N
    
    sqrt1 = (Rtot_mag**2)*(D**2)
    sqrt2 = (N_correction2*Rtot_avg*ISIviol*n)/(tau - tau_c)
    Rout_mag = N_correction1*(Rtot_mag*D - (sqrt1 - sqrt2)**(0.5))
    
    Rout_avg = np.average(Rout_mag*np.array(Rout_unit))
    FDR = Rout_avg/Rtot_avg
    
    return FDR

# %%

nwb_read = NWBHDF5IO("sub-MX170903_ses-20180304T135947_behavior.nwb", "r").read()

spike_times = nwb_read.units.spike_times.data[:]
spike_times_index = nwb_read.units.spike_times_index.data[:]
cue_times = nwb_read.trials['cue_times'].data[:]

n_units = len(spike_times_index)
spike_times_index = np.insert(spike_times_index, 0, 0)

# %%

units = []

for idx in range(n_units):
    idx1 = spike_times_index[idx]
    idx2 = spike_times_index[idx+1]
    
    units.append(spike_times[idx1:idx2])
    
# length of time after go cue to use for PSTH
T = 2

aligned_units = []
for unit in units:
    aligned_spikes = []
    for cue in cue_times:
        temp = unit - cue
        temp = temp[(temp <= T) & (temp >= 0)]
        aligned_spikes.extend(temp)
    aligned_units.append(aligned_spikes)
    
    
ISI_viol = []
for unit in units:
    ISI_viol.append(sum(np.diff(unit) < 0.0025)/len(unit))
    
bin_size = 50
PSTHs = []
for unit in aligned_units:
    PSTHs.append(JV_utils.gen_PSTH(unit, 265, T, bin_size))

PSTHs = np.stack(PSTHs)




    
# %%

ISI_viol = sum(np.diff(units[0]) < 0.0025)/len(units[0])

bin_size = 50

PSTH1 = JV_utils.gen_PSTH(aligned_units[0], 265, T, bin_size)
PSTH2 = JV_utils.gen_PSTH(aligned_units[6], 265, T, bin_size)

test = FDR_master(ISI_viol, PSTH1, PSTH2/np.linalg.norm(PSTH2), 1)
