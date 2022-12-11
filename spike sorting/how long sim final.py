# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:30:38 2022

@author: jpv88
"""

import matplotlib.pyplot as plt
import numpy as np

import neuronsim

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

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    if Rtot != 0:
        FDR = predRout/Rtot
    else:
        FDR = 0
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR


# %%

FRs = np.arange(1, 21, 1)
FDRs = np.arange(0, 0.5, 0.025)
FDRs[0] = 0.001

inputs = np.dstack(np.meshgrid(FRs, FDRs))
stds = np.zeros((20, 20))

for i in range(20):
    for j in range(20):

        Rtot = inputs[i,j,0]
        Rout = inputs[i,j,1]*Rtot
        Rin = Rtot - Rout
        
        temp_Fvs = []
        for _ in range(30):
            temp_Fvs.append(neuronsim.sim_Fv(Rin, Rout, t_stop=3600, N=1)[0])
            
        pred_FDR = economo_eq([Rtot]*30, temp_Fvs)
        pred_FDR = np.array(pred_FDR)[~np.isnan(pred_FDR)]
        stds[i,j] = np.std(pred_FDR)
        

# %%

fig, ax = plt.subplots()
xx, yy = np.meshgrid(FRs, FDRs)
plt.pcolor(xx, yy, stds)
cbar = plt.colorbar()
cbar.set_label('$\sigma$ of predicted FDR')     
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('FDR')   
plt.title('Recording time = 1 hour', fontsize=18)
plt.gca().invert_yaxis()
plt.tight_layout()

    
