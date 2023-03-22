# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:14:38 2023

@author: jpv88
"""

import JV_utils
import neuronsim

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy
from sklearn.metrics import r2_score

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
# index 512 gets kind of buggy when you try to scale it
PSTHs = np.delete(PSTHs, 512, 0)

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

scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3]
locs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3]
FDR_avg = np.zeros((len(locs), len(scales)))
FDR_avg_old = np.zeros((len(locs), len(scales)))
FDR_dist_true = np.zeros((len(locs), len(scales)))
FDR_median = np.zeros((len(locs), len(scales)))
FDR_true_median = np.zeros((len(locs), len(scales)))


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
        FDR_median[m,j] = np.median(pred_FDR)
        FDR_true_median[m,j] = np.median(FDR_dist)
        
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

# %%

FDR_true_median = [1.42E-02, 2.20E-02, 4.92E-02, 9.28E-02,	1.75E-01,	1.97E-01,
                   1E-01, 2.23E-02,	2.81E-02,	5.19E-02,	9.51E-02,	1.69E-01,
                   5E-01,	2.24E-01, 5.06E-02,	5.32E-02,	6.97E-02,	1.03E-01,
                   4E-01,	2.08E-01,	2.33E-01, 1.00E-01,	1.02E-01,	1.10E-01,
                   2E-01,	1.93E-01,	2.24E-01,	2.49E-01, 2.00E-01,	2.01E-01,
                   5E-01,	2.16E-01,	2.53E-01,	2.88E-01,	3.00E-01, 2.50E-01,
                   1E-01,	2.54E-01,	2.62E-01,	2.90E-01,	3.09E-01,	3.27E-01,
                   3.00E-01,	3.01E-01,	3.02E-01,	3.08E-01,	3.35E-01,
                   6E-01,	3.50E-01]

FDR_median = [1.59E-02,	2.24E-02,	5.31E-02,	1.07E-01,	2.06E-01,	2.25E-01,
              7E-01, 2.19E-02,	2.95E-02,	5.22E-02,	1.16E-01,	2.04E-01,
              1E-01,	2.51E-01, 5.11E-02,	5.65E-02,	7.73E-02,	1.15E-01,
              3E-01,	2.36E-01,	2.62E-01, 1.14E-01,	1.22E-01,	1.26E-01,
              1E-01,	2.34E-01,	2.59E-01,	2.64E-01, 2.41E-01,	2.44E-01,
              9E-01,	2.51E-01,	2.82E-01,	3.21E-01,	3.27E-01, 3.12E-01,
              6E-01,	3.07E-01,	3.09E-01,	3.31E-01,	3.43E-01,	3.51E-01,
              3.63E-01,	3.64E-01, 3.56E-01,	3.57E-01,	3.75E-01,	3.62E-01,	3.45E-01]




# %%

fig, ax = plt.subplots()
plt.scatter(FDR_true_median, FDR_median, color='b', s=20)


y_pred, reg, R2 = JV_utils.lin_reg(FDR_true_median, FDR_median)


x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)

plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.plot([0, 0.4], [0, 0.4], 'k', ls='--')
plt.xlabel('True Median FDR')
plt.ylabel('Predicted Median FDR')
# plt.title('MEDIAN', fontsize=18)


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(FDR_true_median, FDR_median)
plt.annotate('R^2 = 0.85', (0.2, 0.05), fontsize=20)


# %%

plt.scatter(FDR_dist_true, FDR_avg_old, color='b', s=20)

y_pred, reg, R2 = JV_utils.lin_reg(FDR_dist_true, FDR_avg_old)


x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)

plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.plot([0, 0.4], [0, 0.4], 'k', ls='--')
plt.xlabel('True Mean FDR')
plt.ylabel('Predicted Mean FDR')

# plt.title('MEAN', fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(FDR_dist_true, FDR_avg_old)
plt.annotate('R^2 = 0.87', (0.2, 0.05), fontsize=20)

