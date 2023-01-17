# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:14:38 2023

@author: jpv88
"""

import JV_utils
import neuronsim

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
PSTHs = np.delete(PSTHs)

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

ks = [5, 10, 100, 500, 1000, 2000, 3000, 4000]
FDR_avg = []
for k in ks:

    N_con = [1, 2, 5, 10]
    N_con = np.array(choices(N_con, k=k), dtype='float')
    N_con[N_con == 10] = float('inf')
    
    Rtots = [4, 8, 12]
    Rtots = choices(Rtots, k=k)
    
    scale = 0.05
    FDRs = cauchy.rvs(scale=scale, size=10000)
    FDRs = FDRs[FDRs >= 0]
    FDRs = FDRs[FDRs < 1]
    FDR_dist = FDRs
    FDRs = np.random.choice(FDR_dist, size=k)
        
    PSTH_idx = list(range(len(PSTHs)))
    idx_pairs = []
    for _ in range(k):
        idx_pairs.append(sample(PSTH_idx, 2))
    
    pred_FDR = []  
    OF = []
    Fvs = []
    PSTHs_run = []
    for i in range(k):
        
        Rin = PSTHs[idx_pairs[i][0]]
        Rout = PSTHs[idx_pairs[i][1]]

        
        Rout_target = FDRs[i]*Rtots[i]
        Rin_target = Rtots[i] - Rout_target
        
        scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                            method='bounded', bounds=[0, 100]).x
        Rin = Rin*scale
    
        scale = minimize_scalar(Rout_scale_ob, 
                            args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                            method='bounded', bounds=[0, 100]).x
        
        Rout = scale*Rout
        
        Rin[Rin<0] = 0
        Rout[Rout<0] = 0
        
        Rtot = Rin + Rout
        
        center = np.average(Rin)*np.average(Rout[0])
        OF.append((np.dot(Rin, Rout)/1000 - center)/center)
        
        Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                    neurons=N_con[i], N=100)
        Fvs.append(Fv)
        
        PSTHs_run.append(Rtot)
    
        
    PSTHs_run = np.stack(PSTHs_run)
    pred_FDR = JV_utils.pred_FDR(PSTHs_run, Fvs)
    
    FDR_avg.append(np.mean(pred_FDR))
    
# %%

k = 1000

scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25]
locs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25]
FDR_avg = np.zeros((len(locs), len(scales)))
FDR_avg_old = np.zeros((len(locs), len(scales)))
FDR_dist_true = np.zeros((len(locs), len(scales)))


N_con = [1, 2, 5, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
N_con[N_con == 10] = float('inf')

Rtots = [4, 8, 12, 16]
Rtots = choices(Rtots, k=k)

for m, loc in enumerate(locs):
    for j, scale in enumerate(scales):
        FDRs = cauchy.rvs(loc=loc, scale=scale, size=10000)
        FDRs = FDRs[FDRs >= 0]
        FDRs = FDRs[FDRs < 1]
        FDR_dist = FDRs
        FDRs = np.random.choice(FDR_dist, size=k)
            
        PSTH_idx = list(range(len(PSTHs)))
        idx_pairs = []
        for _ in range(k):
            idx_pairs.append(sample(PSTH_idx, 2))
        
        pred_FDR = []  
        OF = []
        Fvs = []
        PSTHs_run = []
        for i in range(k):
            
            Rin = PSTHs[idx_pairs[i][0]]
            Rout = PSTHs[idx_pairs[i][1]]
        
            
            Rout_target = FDRs[i]*Rtots[i]
            Rin_target = Rtots[i] - Rout_target
            
            scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                                method='bounded', bounds=[0, 100]).x
            Rin = Rin*scale
        
            scale = minimize_scalar(Rout_scale_ob, 
                                args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                                method='bounded', bounds=[0, 100]).x
            
            Rout = scale*Rout
            
            Rin[Rin<0] = 0
            Rout[Rout<0] = 0
            
            Rtot = Rin + Rout
            
            center = np.average(Rin)*np.average(Rout[0])
            OF.append((np.dot(Rin, Rout)/1000 - center)/center)
            
            Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                        neurons=N_con[i], N=100)
            Fvs.append(Fv)
            
            PSTHs_run.append(Rtot)
        
            
        PSTHs_run = np.stack(PSTHs_run)
        pred_FDR = JV_utils.pred_FDR(PSTHs_run, Fvs)
        pred_FDR = np.array(pred_FDR)
        pred_FDR_old = pred_FDR
        
        FDR_avg_old[m,j] = np.mean(pred_FDR)
        
        nan_mask = (pred_FDR == 0.75)
        len_mask = sum(nan_mask)
        pred_FDR = np.delete(pred_FDR, nan_mask)
        for _ in range(len_mask): 
            pred_FDR = np.delete(pred_FDR, pred_FDR.argmin())
        
        FDR_avg[m,j] = np.mean(pred_FDR)
        FDR_dist_true[m,j] = np.mean(FDR_dist)
    
# %%

FDR_avg_flat = FDR_avg_old.flatten()
FDR_dist_flat = FDR_dist_true.flatten()

plt.scatter(FDR_avg_flat, FDR_dist_flat)

plt.plot([0, 0.4], [0, 0.4], ls='--', alpha=0.5)

# FDR_avg = np.array(FDR_avg)
# FDR_avg_old = np.array(FDR_avg_old)
# FDR_dist_true = np.array(FDR_dist_true)

# new_error = FDR_avg - FDR_dist_true
# old_error = FDR_avg_old - FDR_dist_true