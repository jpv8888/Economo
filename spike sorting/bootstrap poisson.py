# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:57:24 2022

@author: jpv88
"""

import matplotlib.pyplot as plt
import numpy as np

from random import choice, sample

import neuronsim


N = 50

spikes = neuronsim.sim_spikes(2, 2, N=N, t_stop=6)
true_Fv = neuronsim.sim_Fv(2, 2, N=1000, t_stop=100)[0]
guess_Fv = np.mean([sum(np.diff(subset) < 0.0025)/len(subset) for subset in spikes])
spikes_flat = [item for sublist in spikes for item in sublist]
spikes_flat = sorted(spikes_flat)

# %%
ISIs = []
for subset in spikes:
    ISIs.extend(list(np.diff(subset)))


# %%

Fv = []
for _ in range(10000):
    bins = 10
    T = 6
    boundaries = np.linspace(0, T, bins)
    subset = []
    
    for i in range(bins-1):
        rand_train = choice(spikes)
        for spk in rand_train:
            if spk >= boundaries[i] and spk < boundaries[i+1]:
                subset.append(spk)
    
    Fv.append(sum(np.diff(subset) < 0.0025)/len(subset))

print(np.mean(Fv))
    
    


# %%

Fv = []
for _ in range(10000):
    subset = sample(spikes_flat, int(len(spikes_flat)/N))
    subset = sorted(subset)
    Fv.append(sum(np.diff(subset) < 0.0025)/len(subset))
    

# %%

# little n is bins, big N is trials
def estimator(spikes, h, n, T, N):
    
    intensity = np.linspace(0, T, n)

    for i, _ in enumerate(intensity):
        tot_sum = 0
        for spike in spikes:
            val = (intensity[i]-spike)/h
            tot_sum += 0.9375*(1-val**2)**2
        intensity[i] = tot_sum/h
        
    return intensity/N
        
test = estimator(spikes, 4, 100, 6, 100)