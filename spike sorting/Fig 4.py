# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:34:22 2023

@author: jpv88
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import JV_utils
import neuronsim

FDR = 0.2
rec_times = [5, 10, 15, 20, 25, 30]
frs = [1, 2, 3, 4, 5, 6]

Fv_stds = np.zeros((len(frs), len(rec_times)))
FDR_stds = np.zeros((len(frs), len(rec_times)))

for i, fr in enumerate(frs):
    for j, rec_time in enumerate(rec_times):
        
        rate_out = 0.2*fr
        rate_in = fr - rate_out
        Fvs = []
        
        for _ in range(1000):
            Fvs.append(neuronsim.sim_Fv(rate_in, 
                                        rate_out, 
                                        t_stop=rec_time*60, 
                                        N=1)[0])
            
        FDRs = []
        for Fv in Fvs:
            FDR = JV_utils.FDR_master(Fv, 
                                      [fr]*200, 
                                      ([rate_out]*200)/np.linalg.norm([rate_out]*200), 
                                      float('inf'))
            
            if ~np.isnan(FDR):
                FDRs.append(FDR)
            else:
                FDRs.append(1)
        
        Fv_stds[i,j] = np.std(Fvs)
        FDR_stds[i,j] = np.std(FDRs)


# %%

from scipy.interpolate import RegularGridInterpolator

x = np.array(frs)
y = np.array(rec_times)

xg, yg = np.meshgrid(x, y, indexing='ij')
data = FDR_stds*2
interp = RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)

xx = np.linspace(1, 6, 100)
yy = np.linspace(5, 30, 100)
X, Y = np.meshgrid(xx, yy, indexing='ij')
Z = interp((X, Y))

normalized = Z/Z.max()  # rescale to between 0 and 1
corrected = np.power(normalized, 0.1) # try values between 0.5 and 2 as a start point

cmap = plt.colormaps['plasma']
plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, rasterized=True)
plt.xlabel('R_tot (Hz)')
plt.ylabel('t (min)')
plt.colorbar()
plt.clim(vmin=0, vmax=0.5)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%

from scipy.interpolate import RegularGridInterpolator

x = np.array(frs)
y = np.array(rec_times)

xg, yg = np.meshgrid(x, y, indexing='ij')
data = Fv_stds*2
interp = RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)

xx = np.linspace(1, 6, 100)
yy = np.linspace(5, 30, 100)
X, Y = np.meshgrid(xx, yy, indexing='ij')
Z = interp((X, Y))

normalized = Z/Z.max()  # rescale to between 0 and 1
corrected = np.power(normalized, 0.1) # try values between 0.5 and 2 as a start point

cmap = plt.colormaps['plasma']
plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, rasterized=True)
plt.xlabel('R_tot (Hz)')
plt.ylabel('t (min)')
plt.colorbar()
plt.clim(vmin=0.001, vmax=0.004)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%

frs = [2, 4, 8]
FDR_true = 0.3
rec_time = 10
iters = 2000

Fvs_mat = np.zeros((len(frs), iters))
FDRs_mat = np.zeros((len(frs), iters))

for i, fr in enumerate(frs):
    
    rate_out = FDR_true*fr
    rate_in = fr - rate_out
    Fvs = []
    
    for _ in range(iters):
        Fvs.append(neuronsim.sim_Fv(rate_in, 
                                    rate_out, 
                                    t_stop=rec_time*60, 
                                    N=1)[0])
    
    Fvs_mat[i,:] = Fvs
    
    FDRs = []
    for Fv in Fvs:
        FDR = JV_utils.FDR_master(Fv, 
                                  [fr]*200, 
                                  ([rate_out]*200)/np.linalg.norm([rate_out]*200), 
                                  float('inf'))
        
        if ~np.isnan(FDR):
            FDRs.append(FDR)
        else:
            FDRs.append(1)
    
    FDRs_mat[i,:] = FDRs
    
# %%

fig, ax = plt.subplots()
plt.hist(Fvs_mat[0,:])
plt.hist(Fvs_mat[1,:])
plt.hist(Fvs_mat[2,:])

# %%
fig, ax = plt.subplots()
plt.hist(FDRs_mat[0,:])
plt.hist(FDRs_mat[1,:])
plt.hist(FDRs_mat[2,:])

# %%

fig, ax = plt.subplots()
sns.kdeplot(FDRs_mat[0,:], clip=[0,1], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')
sns.kdeplot(FDRs_mat[1,:], clip=[0,1], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')
sns.kdeplot(FDRs_mat[2,:], clip=[0,1], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')

plt.vlines(0.3, 0, 8, ls='--', colors='k')

plt.xlabel('Predicted FDR')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# %%

fig, ax = plt.subplots()
sns.kdeplot(Fvs_mat[0,:], clip=[0,0.05], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')
sns.kdeplot(Fvs_mat[1,:], clip=[0,0.05], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')
sns.kdeplot(Fvs_mat[2,:], clip=[0,0.05], bw_adjust=2, common_norm=True, 
            fill=True, multiple='layer')

plt.vlines(0.3, 0, 8, ls='--', colors='k')
plt.xlim(0, 0.015)
plt.xlabel('ISI_viol')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)





        