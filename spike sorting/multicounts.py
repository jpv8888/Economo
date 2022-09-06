# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:48:00 2022

@author: jpv88
"""

import neuronsim
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

R_in_range = list(range(1, 51, 10))
R_out_range = list(range(1, 51, 10))
        
counts = np.zeros((len(R_in_range), len(R_out_range)))
for i, R_in in enumerate(R_in_range):
    for j, R_out in enumerate(R_out_range):
        counts_temp = neuronsim.sim_Fv_multicount(R_in, R_out, t_stop=100, N=10)[3]
        counts_temp = [el for el in counts_temp if el != 0]
        counts[i,j] = np.mean(counts_temp)
        
# %%

cmap = mpl.cm.hot
plt.contourf(R_in_range, R_out_range, np.transpose(counts), cmap=cmap)

norm = mpl.colors.Normalize(vmin=np.amin(counts), vmax=np.amax(counts))
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))


plt.xlabel('$R_{in}$ (Hz)', fontsize=14)
plt.ylabel('$R_{out}$ (Hz)', fontsize=14)
plt.title('Average ISI violations per ISI violation', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar.set_label('Average ISI violations', rotation=270, fontsize=14)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

# %%

windows = np.zeros((len(R_in_range), len(R_out_range)))
R_tots = np.zeros((len(R_in_range), len(R_out_range)))
for i, R_in in enumerate(R_in_range):
    for j, R_out in enumerate(R_out_range):
        fr = R_in + R_out
        windows[i, j] = fr*2*0.0025
        R_tots[i, j] = fr

R_ins = np.transpose([R_in_range] * 5)


counts_avg = []
R_in_avg = []
counts_std = []
for val in np.unique(R_ins):
    idx_mask = (R_ins == val)
    counts_avg.append(np.mean(counts[idx_mask]))
    R_in_avg.append(val)
    counts_std.append(np.std(counts[idx_mask]))
    
plt.scatter(R_in_avg, counts_avg)
plt.errorbar(R_in_avg, counts_avg, yerr=counts_std, ls='none')

plt.xlabel('$R_{in}$ (Hz)', fontsize=14)
plt.ylabel('ISI violations per violation', fontsize=14)
plt.title('Counts vs. $R_{in}$', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# %%

windows = np.zeros((len(R_in_range), len(R_out_range)))
R_tots = np.zeros((len(R_in_range), len(R_out_range)))
for i, R_in in enumerate(R_in_range):
    for j, R_out in enumerate(R_out_range):
        fr = R_in + R_out
        windows[i, j] = fr*2*0.0025
        R_tots[i, j] = fr

R_outs = np.tile(R_out_range, (len(R_in_range), 1))


counts_avg = []
R_out_avg = []
counts_std = []
for val in np.unique(R_outs):
    idx_mask = (R_outs == val)
    counts_avg.append(np.mean(counts[idx_mask]))
    R_out_avg.append(val)
    counts_std.append(np.std(counts[idx_mask]))
    
plt.scatter(R_out_avg, counts_avg)
plt.errorbar(R_out_avg, counts_avg, yerr=counts_std, ls='none')

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('ISI violations per violation', fontsize=14)
plt.title('Counts vs. $R_{out}$', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# %%

plt.scatter(R_ins, counts)
