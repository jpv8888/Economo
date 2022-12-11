# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:30:08 2022

@author: 17049
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import JV_utils

from tqdm import tqdm

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

# the final equation
def FDR_master_mod(ISIviol, Rtot, Rout_unit, N, tau=2.5, tau_c=0):
    
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
    sqrt = float(sqrt1 - sqrt2)
    if sqrt >= 0:
        Rout_mag = N_correction1*(Rtot_mag*D - abs((sqrt)**(0.5)))
    elif sqrt < 0:
        Rout_mag = N_correction1*(Rtot_mag*D + abs((sqrt)**(0.5)))
        
    
    Rout_avg = np.average(Rout_mag*np.array(Rout_unit))
    FDR = Rout_avg/Rtot_avg
    
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

FDRs = []
ISI_viols = np.arange(0, 1, 0.0001)
for ISI_viol in ISI_viols:
    FDRs.append(FDR_master_mod(ISI_viol, PSTHs[i,:], others_unit, N=float('inf')))

plt.plot(ISI_viols, FDRs)

# %% Nuo 2022

mat_contents = sio.loadmat('nuo_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('nuo_ISIviol')
ISI_viol = mat_contents['ISI_viol']

N = len(PSTHs)

PSTHs_unit = np.zeros(np.shape(PSTHs))

for i in range(N):
    PSTHs_unit[i,:] = PSTHs[i,:]/np.linalg.norm(PSTHs[i,:])
    
FDRs = []
for i in tqdm(range(N)):
    
    other_idx = list(range(N))
    del other_idx[i]
    others_PSTH = PSTHs[other_idx,:]
    others_mean = np.mean(others_PSTH, axis=0)
    others_unit = others_mean/np.linalg.norm(others_mean)
    
    inf_FDR = FDR_master(ISI_viol[i], PSTHs[i,:], others_unit, N=float('inf'))
    
    one_FDRs = []
    for j in other_idx:
        temp = FDR_master(ISI_viol[i], PSTHs[i,:], PSTHs_unit[j,:], N=1)
        if not np.isnan(temp):
            one_FDRs.append(temp)
        
    one_FDR = np.mean(one_FDRs)
    if not np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(np.mean((inf_FDR, one_FDR)))
    elif np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(one_FDR)
    elif not np.isnan(inf_FDR) and np.isnan(one_FDR):
        FDRs.append(inf_FDR)
    else:
        FDRs.append(float('nan'))
        
# %%

final_FDR = np.mean(np.array(FDRs)[~np.isnan(FDRs)])
final_FDRs_ones = np.array(FDRs)
final_FDRs_ones[np.isnan(final_FDRs_ones)] = 1
final_FDR_ones = np.mean(final_FDRs_ones)

# %%

plt.hist(final_FDRs_ones)
plt.xlabel('predicted FDR', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Nuo 2022', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# %% Hidehiko 2019

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('hidehiko_ISIviol')
ISI_viol = mat_contents['ISI_viol']

N = len(PSTHs)

PSTHs_unit = np.zeros(np.shape(PSTHs))

for i in range(N):
    PSTHs_unit[i,:] = PSTHs[i,:]/np.linalg.norm(PSTHs[i,:])
    
inf_FDRs = []
FDRs = []
for i in tqdm(range(N)):
    
    other_idx = list(range(N))
    del other_idx[i]
    others_PSTH = PSTHs[other_idx,:]
    others_mean = np.mean(others_PSTH, axis=0)
    others_unit = others_mean/np.linalg.norm(others_mean)
    
    inf_FDR = FDR_master(ISI_viol[i], PSTHs[i,:], others_unit, N=float('inf'))
    inf_FDRs.append(inf_FDR)
    
    one_FDRs = []
    max_dot = 0
    idx = None
    for j in other_idx:
        temp = FDR_master(ISI_viol[i], PSTHs[i,:], PSTHs_unit[j,:], N=1)
        dot = np.dot(PSTHs_unit[179], PSTHs_unit[j])
        if dot > max_dot:
            max_dot = dot
            idx = j
        if not np.isnan(temp):
            one_FDRs.append(temp)
        
    one_FDR = np.mean(one_FDRs)
    if not np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(np.mean((inf_FDR, one_FDR)))
    elif np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(one_FDR)
    elif not np.isnan(inf_FDR) and np.isnan(one_FDR):
        FDRs.append(inf_FDR)
    else:
        FDRs.append(float('nan'))
        
# %%

final_FDR = np.mean(np.array(FDRs)[~np.isnan(FDRs)])
final_FDRs_ones = np.array(FDRs)
final_FDRs_ones[np.isnan(final_FDRs_ones)] = 1
final_FDR_ones = np.mean(final_FDRs_ones)

# %%

plt.hist(FDRs)
plt.xlabel('predicted FDR', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Hidehiko 2019', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# %% Steinmetz 2019


N = max(clusters).item() + 1

spikes = [[] for _ in range(N)]

for i, spike in enumerate(times):
    clust = clusters[i].item()
    spikes[clust].append(spike.item())

# %%

ISI_viol = []
tau = 0.0025

for clust in spikes:
    ISI_viol.append(sum(np.diff(clust) < tau)/len(clust))
    
# %%

end_t = max(times).item()
num_bins = round(3/0.05)
num_trials = len(goCue_times)

PSTHs = np.zeros((N, num_bins))

for i, clust in enumerate(spikes):
    trials = [[] for _ in range(len(goCue_times))]
    for idx, j in enumerate(goCue_times):
        trials_spikes = np.array(clust)[(np.array(clust) < j+1.5) & (np.array(clust) > j-1.5)]
        trials[idx] = np.array(trials_spikes) - j + 1.5
    trials = np.concatenate(trials, axis=0)
    PSTH = JV_utils.spikes_to_firing_rates(trials, num_trials, T=3, N=num_bins)
    PSTHs[i,:] = PSTH
    print(i)
    
# %%

N = len(PSTHs)

PSTHs_unit = np.zeros(np.shape(PSTHs))

for i in range(N):
    PSTHs_unit[i,:] = PSTHs[i,:]/np.linalg.norm(PSTHs[i,:])
    
FDRs = []
for i in tqdm(range(N)):
    
    other_idx = list(range(N))
    del other_idx[i]
    others_PSTH = PSTHs[other_idx,:]
    others_mean = np.mean(others_PSTH, axis=0)
    others_unit = others_mean/np.linalg.norm(others_mean)
    
    inf_FDR = FDR_master(ISI_viol[i], PSTHs[i,:], others_unit, N=float('inf'))
    
    one_FDRs = []
    for j in other_idx:
        temp = FDR_master(ISI_viol[i], PSTHs[i,:], PSTHs_unit[j,:], N=1)
        if not np.isnan(temp):
            one_FDRs.append(temp)
        
    one_FDR = np.mean(one_FDRs)
    if not np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(np.mean((inf_FDR, one_FDR)))
    elif np.isnan(inf_FDR) and not np.isnan(one_FDR):
        FDRs.append(one_FDR)
    elif not np.isnan(inf_FDR) and np.isnan(one_FDR):
        FDRs.append(inf_FDR)
    else:
        FDRs.append(float('nan'))
        
# %%

final_FDR = np.mean(np.array(FDRs)[~np.isnan(FDRs)])
final_FDRs_ones = np.array(FDRs)
final_FDRs_ones[np.isnan(final_FDRs_ones)] = 1
final_FDR_ones = np.mean(final_FDRs_ones)

# %%

plt.hist(final_FDRs_ones)
plt.xlabel('predicted FDR', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Steinmetz 2019', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# %%
import neuronsim

Fvs = []

for _ in range(1000):
    Fvs.append(neuronsim.sim_Fv(0.4, 1.6, N=1, t_stop=600)[0])
    
true_Fv = neuronsim.sim_Fv(0.4, 1.6, N=100, t_stop=10000)[0]
    
    
# %%

from scipy import stats
k2, p = stats.shapiro(np.array(Fvs)[:50]*100)

fig, ax = plt.subplots()
plt.hist(np.array(Fvs)*100, bins=30)
plt.xlabel('% $ISI_{viol}$', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('$R_{in}$ = 0.4 Hz, $R_{out}$ = 1.6 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.vlines(true_Fv*100, 0, 175, color='k', ls='--')
fig.patches.extend([plt.Rectangle((0.5, 0), ax.get_xlim()[1]-0.5, ax.get_ylim()[1],
                                          color='r', alpha=0.2, zorder=1000, 
                                          fill=True, transform=ax.transData)])

plt.tight_layout() 

fig, ax = plt.subplots()
plt.boxplot(np.array(Fvs)*100)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
plt.title('$R_{in}$ = 0.4 Hz, $R_{out}$ = 1.6 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()     

# %%

FDRs = []

for Fv in Fvs:
    FDRs.append(economo_old(2, Fv).item())
    
FDRs = np.array(FDRs)

FDRs[np.isnan(FDRs)] = 1

    
# %%

ordered = sorted(Fvs)

one_sig = [np.mean(Fvs) - np.std(Fvs), np.mean(Fvs) + np.std(Fvs)]
two_sig = [np.mean(Fvs) - 2*np.std(Fvs), np.mean(Fvs) + 2*np.std(Fvs)]
three_sig = [np.mean(Fvs) - 3*np.std(Fvs), np.mean(Fvs) + 3*np.std(Fvs)]

one_count = 0
for val in ordered:
    if val >= one_sig[0] and val <= one_sig[1]:
        one_count += 1
        
two_count = 0
for val in ordered:
    if val >= two_sig[0] and val <= two_sig[1]:
        two_count += 1
        
three_count = 0
for val in ordered:
    if val >= three_sig[0] and val <= three_sig[1]:
        three_count += 1
        
percentiles = [one_count/len(Fvs), two_count/len(Fvs), three_count/len(Fvs)]

# %%



F_v = np.arange(0, 0.06, 0.0001)

fig, ax = plt.subplots()
plt.plot(F_v*100, economo_old(20, F_v), lw=4)
plt.xlabel('% $ISI_{viol}$', fontsize=16)
plt.ylabel('predicted FDR', fontsize=16)
plt.title('$R_{tot}$ = 20 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout() 

# %%

frs = np.mean(PSTHs, axis=1)
t_stop = 38.5844*60
FDRs = []

nans = 0
for fr in frs:
    R_in = 0.7*fr
    R_out = 0.3*fr
    Fv = neuronsim.sim_Fv(R_in, R_out, N=1, t_stop=t_stop)[0]
    FDRs.append(economo_old(fr, Fv).item())
    
print(sum(np.isnan(FDRs))/len(frs))

# %% normal distribution

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('hidehiko_ISIviol')
ISI_viol = mat_contents['ISI_viol']

loc = 0.1
scale = 0.125
FDRs = np.random.normal(loc, scale, size=len(PSTHs))
FDRs[FDRs < 0] = 0
Fvs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=386, FDR=FDRs[i]))

pred_FDRs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    pred_FDRs.append(FDR_master(Fvs[i], PSTH, 
                                PSTH_out/np.linalg.norm(PSTH_out), 
                                N=float('inf')))
    
# %% half normal distribution

from scipy.stats import halfnorm

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('hidehiko_ISIviol')
ISI_viol = mat_contents['ISI_viol']

scale = 0.15
FDRs = halfnorm.rvs(scale=scale, size=len(PSTHs))
Fvs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=386, FDR=FDRs[i]))

pred_FDRs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    pred_FDRs.append(FDR_master(Fvs[i], PSTH, 
                                PSTH_out/np.linalg.norm(PSTH_out), 
                                N=float('inf')))
    
# %% cauchy distribution

from scipy.stats import cauchy

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('hidehiko_ISIviol')
ISI_viol = mat_contents['ISI_viol']

scale = 0.05
FDRs = cauchy.rvs(scale=scale, size=10000)
FDRs = FDRs[FDRs >= 0]
FDRs = FDRs[FDRs < 1]
FDRs = np.random.choice(FDRs, size=len(PSTHs))


Fvs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=386, FDR=FDRs[i]))

pred_FDRs = []

for i in range(len(PSTHs)):
    PSTH = PSTHs[i,:]
    PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
    pred_FDRs.append(FDR_master(Fvs[i], PSTH, 
                                PSTH_out/np.linalg.norm(PSTH_out), 
                                N=float('inf')))
    
# %%

plt.hist(FDRs)
plt.xlabel('true FDR', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.title('halfnormal distribution, sigma=0.15', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout() 

fig, ax = plt.subplots()
pred_FDRs = np.array(pred_FDRs)
pred_FDRs[np.isnan(pred_FDRs)] = 1
plt.hist(pred_FDRs)
plt.xlabel('predicted FDR', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.title('predicted FDRs (simulation)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout() 

fig, ax = plt.subplots()
inf_FDRs = np.array(inf_FDRs)
inf_FDRs[np.isnan(inf_FDRs)] = 1
plt.hist(inf_FDRs)
plt.xlabel('predicted FDR', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.title('predicted FDRs (real data)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

