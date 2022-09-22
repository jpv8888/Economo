# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:25:11 2022

@author: jpv88
"""

from random import sample

import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np

import JV_utils
import neuronsim

with open('PSTHs1.pickle', 'rb') as pickle_file:
    PSTHs1 = pickle.load(pickle_file)
    
with open('PSTHs2.pickle', 'rb') as pickle_file:
    PSTHs2 = pickle.load(pickle_file)

L = {array.tobytes(): array for array in PSTHs1}
PSTHs1 = list(L.values())

uniq_means = [np.mean(el) for el in PSTHs1]

# %%

sim_Fv = []

sim_Rtot = []

@np.vectorize
def economo_old(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*Rin*Rout*2*tviol
    Fv = Rviol/(Rin + Rout)

    return Fv

def economo_PSTH(Rtot, Rout, Fv, tviol=0.0025):
    mag_Rtot = np.linalg.norm(Rtot)
    mag_Rout = np.linalg.norm(Rout)
    avg_Rtot = np.average(Rtot)
    avg_Rviol = avg_Rtot*Fv
    # Rviol_mag = np.linalg.norm(Rviol)
    
    Rtot_unit = np.array(Rtot)/mag_Rtot
    Rout_unit = np.array(Rout)/mag_Rout
    D = np.dot(Rtot_unit, Rout_unit)
    
    a = -1/2
    b = mag_Rtot*D
    c = -avg_Rviol/(2*tviol)
    
    predRout_mag = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    actual_Rout = predRout_mag*np.array(Rout_unit)
    predRout = np.average(actual_Rout)
    
    FDR = predRout/avg_Rtot
    #FDR = predRout
    
    return FDR
    
    

main_idx = 25
for i in list(range(len(PSTHs1)))[:main_idx] + list(range(len(PSTHs1)))[main_idx+1:]:
    Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[main_idx], PSTHs1[i], FDR=0.1)
    sim_Fv.append(Fv_temp)
    sim_Rtot.append(Rtot_temp)
    
# %%

# eq_Fv = economo_Fv(np.mean(PSTHs1[main_idx]), 0.05*np.mean(PSTHs1[main_idx]))

pred_FDR = []
for i in list(range(len(PSTHs1)))[:main_idx] + list(range(len(PSTHs1)))[main_idx+1:]:
    F_v_locs, Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[main_idx], PSTHs1[i], FDR=0.5)
    pred_FDR.append(economo_PSTH(PSTHs1[main_idx], PSTHs1[i], Fv_temp))
    
# %% 
plt.figure()
plt.hist(sim_Fv)
plt.title('Fv sim, Rtot = 15.3 Hz, FDR = 5%', fontsize=18)
plt.xlabel('Fv', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=eq_Fv, ymin=0, ymax=17.5, colors='k', ls='--')
plt.tight_layout()

plt.figure()
plt.hist(economo_old([np.mean(PSTHs1[main_idx])*1.05]*85, sim_Fv))
plt.title('Predicted FDR, Rtot = 15.3 Hz, FDR = 5%', fontsize=18)
plt.xlabel('FDR', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=0.05, ymin=0, ymax=17.5, colors='k', ls='--')
plt.tight_layout()

# %% pairwise comparison vs random mixing

sim_Fv = []

sim_Rtot = []

@np.vectorize
def economo_old(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*Rout*Rout*2*tviol
    Fv = Rviol/(Rin + Rout)

    return Fv

main_idx = 1
for i in list(range(len(PSTHs1)))[:main_idx] + list(range(len(PSTHs1)))[main_idx+1:]:
    Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[main_idx], PSTHs1[i], FDR=0.05)
    sim_Fv.append(Fv_temp)
    sim_Rtot.append(Rtot_temp)
    
eq_Fv = economo_Fv(np.mean(PSTHs1[main_idx]), 0.05*np.mean(PSTHs1[main_idx]))
pred_FDR1 = economo_old([np.mean(PSTHs1[main_idx])*1.05]*85, sim_Fv)
    
plt.figure()
plt.hist(sim_Fv)
plt.title('Fv sim pairwise, Rtot = 5.5 Hz, FDR = 5%', fontsize=18)
plt.xlabel('Fv', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=eq_Fv, ymin=0, ymax=17.5, colors='k', ls='--')
plt.tight_layout()

plt.figure()
plt.hist(pred_FDR1)
plt.title('Predicted FDR pairwise, Rtot = 5.5 Hz, FDR = 5%', fontsize=18)
plt.xlabel('FDR', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=0.05, ymin=0, ymax=23, colors='k', ls='--')
plt.vlines(x=JV_utils.ci_95(pred_FDR1), ymin=0, ymax=23, colors='k', ls='--')
plt.tight_layout()

sim_Fv2 = []

num_mix = 10
for i in list(range(len(PSTHs1) - 1)):
    idxs = list(range(len(PSTHs1)))[:main_idx] + list(range(len(PSTHs1)))[main_idx+1:]
    samples = sample(idxs, num_mix)
    out_mix = np.sum(PSTHs1[idx] for idx in samples)
    Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[main_idx], out_mix, FDR=0.05)
    sim_Fv2.append(Fv_temp)
    sim_Rtot.append(Rtot_temp)
    
pred_FDR2 = economo_old([np.mean(PSTHs1[main_idx])*1.05]*85, sim_Fv2)
    
plt.figure()
plt.hist(sim_Fv2)
plt.title('Fv sim multi, Rtot = 5.5 Hz, FDR = 5%', fontsize=18)
plt.xlabel('Fv', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=eq_Fv, ymin=0, ymax=17.5, colors='k', ls='--')
plt.tight_layout()

plt.figure()
plt.hist(pred_FDR2)
plt.title('Predicted FDR multi, Rtot = 5.5 Hz, FDR = 5%', fontsize=18)
plt.xlabel('FDR', fontsize=14)
plt.ylabel('#', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.vlines(x=0.05, ymin=0, ymax=17.5, colors='k', ls='--')
plt.vlines(x=JV_utils.ci_95(pred_FDR2), ymin=0, ymax=17.5, colors='k', ls='--')
plt.tight_layout()

# %%

FDR_mean = []
FDR_std = []

num_mix = list(range(10))[1:]

for j in num_mix:
    sim_Fv2 = []
    sim_Rtot = []
    for i in list(range(len(PSTHs1) - 1)):
        idxs = list(range(len(PSTHs1)))[:main_idx] + list(range(len(PSTHs1)))[main_idx+1:]
        samples = sample(idxs, j)
        out_mix = np.sum(PSTHs1[idx] for idx in samples)
        Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[main_idx], out_mix, FDR=0.05)
        sim_Fv2.append(Fv_temp)
        sim_Rtot.append(Rtot_temp)
    
    pred_FDR2 = economo_old([np.mean(PSTHs1[main_idx])*1.05]*85, sim_Fv2)
    FDR_mean.append(np.mean(pred_FDR2))
    FDR_std.append(np.std(pred_FDR2))
    
# %%

plt.figure()
plt.plot(num_mix, np.array(FDR_mean)/FDR_mean[0], lw=3)
plt.plot(num_mix, np.array(FDR_std)/FDR_std[0], lw=3)
plt.legend(['Mean', 'Standard Dev.'], fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Mixing #', fontsize=14)
plt.ylabel('Value Normalized to Pairwise Comparison', fontsize=14)
plt.title('Effect of Mixing Number on Probability Distribution', fontsize=18)
plt.tight_layout()

    
    
    
    

# %%

rates = PSTHs1[:50]
FDRs = [0.05] * 50
spike_trains, electrode_idx = neuronsim.sim_neurons(rates, FDRs)

PSTHs = [JV_utils.spikes_to_firing_rates(el, 1) for el in spike_trains]

@np.vectorize
def economo_eq(Rtot_mag, Rtot_mean, n, D, Dprime, F_v, tviol=0.0025):

    
    a = tviol*Dprime - 2*tviol*D
    b = Rtot_mag*D*2*tviol
    c = F_v*n*Rtot_mean
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot_mean
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

@np.vectorize
def economo_old(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

# %%

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

PSTHs1_norm = [el/vector_mag(el) for el in PSTHs1]
PSTHs2_norm = [el/vector_mag(el) for el in PSTHs2]

PSTHs1_means = [np.mean(el) for el in PSTHs1]
PSTHs2_means = [np.mean(el) for el in PSTHs2]

Rtots = []

for i in range(len(PSTHs1)):
    PSTH1_temp = PSTHs1[i]
    PSTH2_temp = PSTHs2[i]
    Rtots.append(PSTH1_temp + PSTH2_temp)
    
Rtots_mag = [vector_mag(el) for el in Rtots]
Rtots_mean = [np.mean(el) for el in Rtots]

mags1 = [vector_mag(el) for el in PSTHs1]
mags2 = [vector_mag(el) for el in PSTHs2]

FDR_eq = []
FDR_real = []
F_v_sim = []
pred_FDR = []
pred_FDR2 = []
for i in range(len(PSTHs1) - 700):
    print(i)
    R_in = mags1[i]
    R_out = mags2[i]
    
    Rtot_mean = PSTHs1_means[i] + PSTHs2_means[i]

    D = np.dot(PSTHs1_norm[i], PSTHs2_norm[i])

    Dprime = np.dot(PSTHs2_norm[i], PSTHs2_norm[i])

    Rtot_mag = R_in + R_out
    
    F_v_sim_temp, FDR_temp = neuronsim.sim_Fv_PSTH(PSTHs1[i], PSTHs2[i])[0:2]
    F_v_sim.append(F_v_sim_temp)
    FDR_real.append(FDR_temp)
    FDR_eq.append(economo_eq(Rtot_mag, Rtot_mean, 100, D, Dprime, F_v_sim_temp))
    pred_FDR.append(R_out/Rtots_mag[i])
    pred_FDR2.append(PSTHs2_means[i]/Rtots_mean[i])

FDR_eq = [el.item() for el in FDR_eq]

# %%

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

PSTHs1_norm = [el/vector_mag(el) for el in PSTHs1]
PSTHs2_norm = [el/vector_mag(el) for el in PSTHs2]

PSTHs1_means = [np.mean(el) for el in PSTHs1]
PSTHs2_means = [np.mean(el) for el in PSTHs2]

Rtots = []

for i in range(len(PSTHs1)):
    PSTH1_temp = PSTHs1[i]
    PSTH2_temp = PSTHs2[i]
    Rtots.append(PSTH1_temp + PSTH2_temp)
    
Rtots_mag = [vector_mag(el) for el in Rtots]
Rtots_mean = [np.mean(el) for el in Rtots]

mags1 = [vector_mag(el) for el in PSTHs1]
mags2 = [vector_mag(el) for el in PSTHs2]

FDR_eq = []
FDR_real = []
F_v_sim = []
pred_FDR = []
pred_FDR2 = []
for i in range(len(PSTHs1) - 700):
    print(i)
    R_in = mags1[i]
    R_out = mags2[i]
    
    Rtot_mean = PSTHs1_means[i] + PSTHs2_means[i]

    D = np.dot(PSTHs1_norm[i], PSTHs2_norm[i])

    Dprime = np.dot(PSTHs2_norm[i], PSTHs2_norm[i])

    Rtot_mag = R_in + R_out
    
    F_v_sim_temp, FDR_temp = neuronsim.sim_Fv_PSTH(PSTHs1[i], PSTHs2[i])[0:2]
    F_v_sim.append(F_v_sim_temp)
    FDR_real.append(FDR_temp)
    FDR_eq.append(economo_eq(Rtot_mag, Rtot_mean, 100, D, Dprime, F_v_sim_temp))
    pred_FDR.append(R_out/Rtots_mag[i])
    pred_FDR2.append(PSTHs2_means[i]/Rtots_mean[i])

FDR_eq = [el.item() for el in FDR_eq]


