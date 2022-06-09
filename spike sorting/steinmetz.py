# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:48:16 2022

@author: jpv88
"""

import cmath
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/Users/jpv88/OneDrive/Documents/DATA/spikeAndBehavioralData/allData/'
folder = 'Cori_2016-12-14/'

file = 'spikes.clusters.npy'
spikes_clusters = np.load(path + folder + file)

file = 'spikes.times.npy'
spikes_times = np.load(path + folder + file)

spikes_clusters = [el[0] for el in spikes_clusters]
spikes_times = [el[0] for el in spikes_times]

# %%

clusters = np.unique(spikes_clusters).tolist()

spikes = [[] for clust in clusters]

for i, clust in enumerate(spikes_clusters):
    spikes[clust].append(spikes_times[i])

# %%

T = spikes_times[-1] - spikes_times[0]
t_viol = 0.0025

Rtot = []
F_v = []
for clust in spikes:
    Rtot.append(len(clust)/T)
    F_v.append(sum(np.diff(clust) < t_viol)/len(clust))
    
# %%
    
@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):
    
    Rtot = np.array(Rtot)
    F_v = np.array(F_v)

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + cmath.sqrt(b**2 - 4*a*c))/(2*a)
    
    fp = predRout/Rtot
    
    if fp.imag != 0:
        fp = 1

    return fp.real

fps = economo_eq(Rtot, F_v)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.set_size_inches(10, 6)

ax1.boxplot(Rtot)
ax1.set_title('Rtot')
ax2.boxplot(F_v)
ax2.set_title('ISI Violation Fraction')
ax3.boxplot(fps)
ax3.set_title('False Positive Fraction')

# shenoy trautman spike sorting

    
