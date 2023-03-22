# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:16:32 2023

@author: jpv88
"""

import matplotlib.pyplot as plt
import neuronsim

import numpy as np

@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025, t_c=0):

    a = -2*(tviol-t_c)*Rtot
    b = 2*(tviol-t_c)*Rtot
    c = -F_v
    
    if Rtot != 0:
        FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    else:
        FDR = 0
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 0.5

    return FDR

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025, t_c=0):

    tviol -= t_c
    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    # if isinstance(FDR, complex):
    #     FDR = 1

    return FDR

pred_FDR = np.zeros((10, 5))
true_FDR = np.zeros((10, 5))
t_stop = 100000
for i, Rin in enumerate(np.arange(1, 11, 1)):
    for k, Rout in enumerate(np.arange(1, 6, 1)):
        spikes, ids = neuronsim.sim_spikes(Rin, Rout, N=1, t_stop=t_stop)
        spikes = spikes[0]
        
        old_spikes = spikes.copy()
        censor = 0.001
        
        spikes_c = []
        ids_c = []
        for j, n in enumerate(spikes):
            if not spikes_c or abs(n - spikes_c[-1]) >= censor:
                spikes_c.append(n)
                ids_c.append(ids[j])
        
        ISI_viol_c = sum(np.diff(spikes_c) < 0.0025)/len(spikes_c)
        Rtot_c = len(spikes_c)/t_stop
        
        pred_FDR[i,k] = economo_eq(Rtot_c, ISI_viol_c, t_c=0)
        true_FDR[i,k] =  sum(np.array(ids_c)==1)/len(ids_c)
        
# %%

plt.scatter(true_FDR, pred_FDR)
plt.plot([0, 0.9], [0, 0.9], 'k', ls='--')
plt.xlim(0, 0.9)
plt.ylim(0, 0.9)
plt.xlabel('True FDR')
plt.ylabel('Predicted FDR')
plt.title('Effect of a 1 ms censor period', fontsize=18)
plt.tight_layout()

# %%

plt.scatter(true_FDR, pred_FDR)
plt.plot([0, 0.9], [0, 0.9], 'k', ls='--')
plt.xlim(0, 0.9)
plt.ylim(0, 0.9)
plt.xlabel('True FDR')
plt.ylabel('Predicted FDR')
plt.title('Effect of a 1 ms censor period (Attempted correction)', fontsize=16)
plt.tight_layout()
