# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:08:04 2022

@author: jpv88
"""

import matplotlib.pyplot as plt
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


import pickle

with open('PSTHs1.pickle', 'rb') as pickle_file:
    PSTHs1 = pickle.load(pickle_file)
    
with open('PSTHs2.pickle', 'rb') as pickle_file:
    PSTHs2 = pickle.load(pickle_file)

L = {array.tobytes(): array for array in PSTHs1}
PSTHs1 = list(L.values())

uniq_means = [np.mean(el) for el in PSTHs1]

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

# %%

ISI_viol = 0.02
pred_FDR1 = []
pred_FDRinf = []

main_idx = 3
other_idx = list(range(len(PSTHs1)))
del other_idx[main_idx]

for second_idx in other_idx:
    Rtot = PSTHs1[main_idx]
    Rout = PSTHs1[second_idx]
    pred_FDR1.append(FDR_master(ISI_viol, Rtot, Rout/vector_mag(Rout), N=1))
    
for second_idx in other_idx:
    Rtot = PSTHs1[main_idx]
    Rout = PSTHs1[second_idx]
    pred_FDRinf.append(FDR_master(ISI_viol, Rtot, Rout/vector_mag(Rout), N=float('inf')))
    
PSTHs1_unit = [el/vector_mag(el) for el in np.array(PSTHs1)[other_idx]]
PSTH_global = np.mean(PSTHs1_unit, axis=0)
PSTH_global = PSTH_global/vector_mag(PSTH_global)
pred_FDRinfinf = FDR_master(ISI_viol, Rtot, PSTH_global, N=float('inf'))

# %%

from random import sample

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

# big N is number of confound neurons, little n is number of combos to build
# the distribution off of
def calc_FDR_N(PSTHs, N, n, main_idx, ISIviol):
    
    num_PSTH = len(PSTHs)
    idxs = range(num_PSTH)
    
    other_idx = list(idxs)
    del other_idx[main_idx]
    
    Rtot = PSTHs[main_idx]
    
    if N != float('inf'):
        
        combos = []
        for _ in range(n):
            combos.append(sample(other_idx, N))
            
        pred_FDR = []
        for i in range(n):
            PSTH_unit = [el/vector_mag(el) for el in np.array(PSTHs)[combos[i]]]
            PSTH_avg = np.mean(PSTH_unit, axis=0)
            PSTH_avg = PSTH_avg/vector_mag(PSTH_avg)
            pred_FDR.append(FDR_master(ISIviol, Rtot, PSTH_avg, N))
    else:
        
        PSTHs_unit = [el/vector_mag(el) for el in np.array(PSTHs)[other_idx]]
        PSTH_global = np.mean(PSTHs_unit, axis=0)
        PSTH_global = PSTH_global/vector_mag(PSTH_global)
        pred_FDR = [FDR_master(ISIviol, Rtot, PSTH_global, N)]*n
        
    return pred_FDR

# %%

Ns = [1, 2, 5, 10, 20, float('inf')]
main_idx = 74
ISI_viol = 0.008

dists = []
for N in Ns:
    dists.append(calc_FDR_N(PSTHs1, N, 1000, main_idx, ISI_viol))

# %%

num_bins = 20
FDR_min = min([min(el) for el in dists])
FDR_max = max([max(el) for el in dists])
FDR_range = FDR_max - FDR_min
bin_size = FDR_range/num_bins

edges = [FDR_min]
i = 0
while i < num_bins:
    edges.append(edges[-1] + bin_size) 
    i += 1

import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.hist(dists[0])
xlim = ax.get_xlim()
ylim = (ax.get_ylim()[0], len(dists[0]))
plt.close()

fig, ax = plt.subplots()

def ani_hist(frame):
    ax.clear()
    ax.vlines(np.mean(np.array(dists[0])[~np.isnan(dists[0])]), ylim[0], 
              ylim[1], colors='k')
    ax.vlines(np.mean(dists[-1]), ylim[0], ylim[1], ls='--', colors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel('Predicted FDR', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('$Unit_{idx}$ = ' + f'{main_idx}' + f', N = {Ns[frame]}', fontsize=18)
    ax.hist(dists[frame], bins=edges)
    plt.tight_layout()

ani = animation.FuncAnimation(fig, ani_hist, frames=len(dists), interval=1000)
ani.save('Unit_idx = ' + f'{main_idx}' + '.gif', dpi=300, 
         writer=animation.PillowWriter(fps=1))


# %%


pred_FDR_total = np.column_stack((pred_FDR1, pred_FDRinf))

plt.hist(pred_FDR_total, histtype='step')

# %%

import JV_utils

full_dist = np.concatenate((pred_FDR1, pred_FDRinf))
full_dist = full_dist[~np.isnan(full_dist)]

print(JV_utils.ci_95(full_dist))

# %%

PSTHs1_unit = [el/vector_mag(el) for el in PSTHs1]

PSTH_global = np.mean(PSTHs1_unit, axis=0)
PSTH_global = PSTH_global/vector_mag(PSTH_global)