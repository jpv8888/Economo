# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:18:30 2022

@author: jpv88
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import neuronsim
from scipy.signal import savgol_filter
import matplotlib.cm as cm

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

Rtot = np.loadtxt('Rtot.csv')
F_v = np.loadtxt('F_v.csv')
ranges = np.loadtxt('ranges.csv', delimiter=',')
ranges = ranges - 1
ranges = ranges.astype(int)
sessions = np.loadtxt('sessions.csv', delimiter=',')


# %%


import csv
  
with open('trials.csv', 'r') as read_obj:
  
    # Return a reader object which will
    # iterate over lines in the given csvfile
    csv_reader = csv.reader(read_obj)
  
    # convert string to list
    trials = list(csv_reader)

for i, test_list in enumerate(trials):
    trials[i] = [int(el)-1 for el in test_list if el]
    
with open('spikes.csv', 'r') as read_obj:
  
    # Return a reader object which will
    # iterate over lines in the given csvfile
    csv_reader = csv.reader(read_obj)
  
    # convert string to list
    spikes = list(csv_reader)

for i, test_list in enumerate(spikes):
    spikes[i] = [float(el) for el in test_list if el]
    
# %%


def spikes_to_firing_rates(spikes, n, T=6, N=100):
    delta = T/N
    bins = np.zeros(N)
    for i in range(N):
        for j in spikes:
            if (j >= i*delta) and (j < (i+1)*delta):
                bins[i] += 1
                
    bins = bins/(delta*n)
    # bins = savgol_filter(bins, 11, 4)
    return bins
    
# %%

sessions_idx = np.unique(sessions)
# iterate through sessions
for idx in sessions_idx[:1]:
    session_mask = (sessions == idx)
    session_ranges = ranges[session_mask]
    session_spikes = [i for (i, v) in zip(spikes, session_mask) if v]
    session_trials = [i for (i, v) in zip(trials, session_mask) if v]
    
    good_pairs = []
    for i, unit_range in enumerate(session_ranges):
        for j, unit_range2 in enumerate(session_ranges[i+1:,:]):
            if all(unit_range == unit_range2):
                good_pairs.append([i, j+i+1])
          
    t_viol = 0.0025
    sessions_track = []
    F_v = []
    F_v_sim = []
    corr = []
    
    # iterate through pairs in a session
    for pair in good_pairs:
        neuron1 = session_spikes[pair[0]]
        neuron2 = session_spikes[pair[1]]
        neuron1_tri = session_trials[pair[0]]
        neuron2_tri = session_trials[pair[1]]
        
        pair_range = session_ranges[pair[0]]
        pair_range = range(pair_range[0], pair_range[1])
        
        # iterate through trials in a pair
        ISI_viol = 0
        spks1_tot = 0
        spks2_tot = 0
        spks1_all = []
        spks2_all = []
        for i in tqdm(pair_range):
            idx_1 = [(el == i) for el in neuron1_tri]
            idx_2 = [(el == i) for el in neuron2_tri]
            spks1 = [i for (i, v) in zip(neuron1, idx_1) if v]
            spks2 = [i for (i, v) in zip(neuron2, idx_2) if v]
            if (len(spks1) == 0) and (len(spks2) == 0):
                continue
            spks_tot = spks1 + spks2
            spks_tot.sort()
            ISI_viol += sum(np.diff(spks_tot) < t_viol)
            spks1_tot += len(spks1)
            spks2_tot += len(spks2)
            spks1_all.extend(spks1)
            spks2_all.extend(spks2)
        
        tot = spks1_tot + spks2_tot
        F_v.append(ISI_viol/tot)
        sessions_track.append(idx)
        
        num_tri = len(pair_range)
        Rtot1 = spks1_tot/(num_tri*6)
        Rtot2 = spks2_tot/(num_tri*6)
        
        spks1_PSTH = spikes_to_firing_rates(spks1_all, num_tri)
        spks2_PSTH = spikes_to_firing_rates(spks2_all, num_tri)
        
        corr.append(stats.pearsonr(spks1_PSTH, spks2_PSTH)[0])
        
        F_v_sim.append(neuronsim.sim_Fv(Rtot1, Rtot2, out_refrac=2.5))
        
# %%
F_v = np.array(F_v)
F_v = F_v.reshape(-1, 1)
F_v_sim = np.array(F_v_sim)
F_v_sim = F_v_sim.reshape(-1, 1)
reg = LinearRegression().fit(F_v_sim, F_v)

colors = cm.coolwarm(corr)

for x, y, c in zip(F_v_sim, F_v, colors):
    plt.scatter(x, y, color=c)

# plt.scatter(F_v_sim, F_v, s=4)


dim = max(np.vstack((F_v, F_v_sim))) + 0.01
x = np.linspace(0, dim, 2)
y = reg.coef_*x + reg.intercept_
plt.plot(x, y, c='b')

x = np.linspace(0, dim, 2)
y = x
plt.plot(x, y, c='r')

plt.xlabel('F_v (sim)')
plt.ylabel('F_v (data)')
plt.xlim(0, dim)
plt.ylim(0, dim)


        
    
    

# %%

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    fp = predRout/Rtot
    
    if isinstance(fp, complex):
        fp = 1

    return fp

fps = economo_eq(Rtot, F_v)

# %%  



plt.boxplot(np.column_stack((Rtot, F_v, fps)))



