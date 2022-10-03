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

from math import sqrt

def wolfram(Rtot, Rout, Rin, Fv, tviol=0.0025):
    Rin_unit = np.array(Rin)/vector_mag(Rin)
    Rout_unit = np.array(Rout)/vector_mag(Rout)
    z = np.dot(Rin_unit, Rout_unit)
    n = len(Rtot)
    y = vector_mag(Rtot)
    w = Fv*np.average(Rtot)
   
    x = sqrt((1/2)*sqrt((((4*n*w*(z**2))/tviol - (2*n*w)/tviol - 4*(y**2)*(z**2))**2) - (4*(n**2)*(w**2))/tviol**2) - (2*n*w*z**2)/tviol + (n*w)/tviol + 2*y**2*z**2)
    return np.average(x*Rout_unit)

    
idx1 = 0
idx2 = 20

Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[idx1], PSTHs1[idx2], FDR=0.1)

Rtot = PSTHs1[idx1] + PSTHs1[idx2]
Rout = PSTHs1[idx2]
Rin = PSTHs1[idx1]
Fv = Fv_temp

print(wolfram(Rtot, Rout, Rin, Fv))
# %%
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

# %% dot product of two constant unit vectors is always 1

# vec1_avg = 10000
# vec2_avg = 143
# N = 20

# vec1 = [vec1_avg]*N

# vec2 = [vec2_avg]*N

# vec1 = np.array(vec1)

# vec2 = np.array(vec2)

# print(np.dot(vec1/vector_mag(vec1), vec2/vector_mag(vec2)))


# %%

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

PSTHs_unit = []

for el in PSTHs1:
    PSTHs_unit.append(el/vector_mag(el))
    

dots = []
idx = 0

best_pair = [0, 0]
best_dot = 0
for idx in range(len(PSTHs_unit)):
    other_PSTH = PSTHs_unit[idx+1:]
    for j, el in enumerate(other_PSTH):
        dots.append(np.dot(PSTHs_unit[idx], el))
        if dots[-1] + 0.5*np.dot(el, el) > best_dot:
            best_pair[0] = PSTHs_unit[idx]
            best_pair[1] = PSTHs_unit[j+idx+1]
            best_dot = dots[-1] + 0.5*np.dot(el, el)
            
from scipy import interpolate

x = np.arange(len(best_pair[0]))
cs1 = CubicSpline(x, best_pair[0])
cs2 = CubicSpline(x, best_pair[1])
xs = np.arange(0, 100, 0.001)

plt.plot(xs, cs1(xs))
plt.plot(xs, cs2(xs))
#plt.plot(savgol_filter(best_pair[1], 25, 2))
            
        
# %%
        
def economo_PSTH(Rtot, Rout, Fv, D, tviol=0.0025):
    mag_Rtot = np.linalg.norm(Rtot)
    mag_Rout = np.linalg.norm(Rout)
    avg_Rtot = np.average(Rtot)
    avg_Rviol = avg_Rtot*Fv
    
    Rout_unit = np.array(Rout)/mag_Rout
    n = len(Rtot)
    
    a = -1/2
    b = (mag_Rtot*D)
    c = -avg_Rviol*n/(2*tviol)
    
    predRout_mag = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    actual_Rout = predRout_mag*np.array(Rout_unit)
    predRout = np.average(actual_Rout)
    
    FDR = predRout/avg_Rtot
    
    return FDR

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)


pred_FDR = []
Rtots = []
for i in range(15):

    for j in range(15)[i+1:]:

        idx1 = i
        idx2 = j
        
        FDR = 0.2
        
        Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(PSTHs1[idx1], PSTHs1[idx2], FDR=FDR)
        
        from scipy.optimize import minimize_scalar
        scale = minimize_scalar(Rout_scale_ob, 
                                args=[PSTHs1[idx2], FDR*np.average(PSTHs1[idx1])],
                                method='bounded', bounds=[0, 100]).x
        
        Rout = scale*PSTHs1[idx2]
        Rin = PSTHs1[idx1]
        Rtot =  Rin + Rout
        
        Fv = Fv_temp
        D = np.dot(Rtot/vector_mag(Rtot), Rout/vector_mag(Rout))
        
        pred_FDR.append(economo_PSTH(Rtot, Rout, Fv, D))
        Rtots.append(Rtot_temp)
    

# %%

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

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

from scipy.optimize import minimize_scalar

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

pred_FDR = []
Rtots = []
Rin_out_dot = []
Rtot_out_dot = []
Fvs = []
R_out_mags = []
R_out_units = []
new = []
for i in range(10):

    for j in range(10)[i+1:]:

        idx1 = i
        idx2 = j
        
        FDR = 0.5
        
        Rin = PSTHs1[idx1]
        
        scale = minimize_scalar(Rout_scale_ob, args=[Rin, 10], 
                                method='bounded', bounds=[0, 100]).x
        Rin = Rin*scale
        
        Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2(Rin, PSTHs1[idx2], FDR=FDR)
        
        scale = minimize_scalar(Rout_scale_ob, 
                                args=[PSTHs1[idx2], (FDR/(1-FDR))*np.average(Rin)],
                                method='bounded', bounds=[0, 100]).x
        
        Rout = scale*PSTHs1[idx2]
        R_out_mags.append(vector_mag(Rout))
        Rtot =  Rin + Rout
        R_out_units.append(Rout/vector_mag(Rout))
        
        Rin_out_dot.append(np.dot(Rin, Rout))
        Rtot_out_dot.append(np.dot(Rtot, Rout))
        new.append(np.dot(Rin, Rout) + 0.5*np.dot(Rout, Rout))
        
        Fv = Fv_temp
        Fvs.append(Fv)
        D = np.dot(Rtot/vector_mag(Rtot), Rout/vector_mag(Rout))
        
        pred_FDR.append(FDR_master(Fv, [np.average(Rtot)]*100, 
                                   [1/10]*100, N=float('inf')))
        Rtots.append(Rtot_temp)

# Fv_temp, Rtot_temp = neuronsim.sim_Fv_PSTH2([np.average(Rin)]*100, 
#                                             [np.average(Rout)]*100, FDR=FDR)

# pred_FDR.append(FDR_master(Fv_temp, [np.average(Rtot)]*100, [1/10]*100, 
#                            N=float('inf')))

# Rin_out_dot.append(1)

# %%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

ax.scatter(2*np.array(Rtot_out_dot), -np.array(R_out_mags)**2, Fvs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# %%
fig, ax = plt.subplots()
plt.scatter(Rtot_out_dot, pred_FDR)
xlims = ax.get_xlim()
plt.hlines(0.05, xlims[0], xlims[1])






        
# %%

import random

Rin = [random.randint(1, 10)] * 100
Rout = [random.randint(1, 10)] * 100

Rtot = np.array(Rin + Rout)

print((vector_mag(Rtot)**2 + vector_mag(Rout)**2)**(1/2))
print(vector_mag(Rin))

# %%

pos = [random.randint(1, 10)] * 100
neg = np.array(pos) * -1

print(vector_mag(pos))
print(vector_mag(neg))
        





