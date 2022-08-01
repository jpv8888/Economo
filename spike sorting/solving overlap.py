# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:21:56 2022

@author: jpv88
"""

import neuronsim

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from scipy.optimize import curve_fit

R_in_range = list(range(1, 61, 10))
R_out_range = list(range(1, 61, 10))

def economo_viol_time_eq(R_tot, tau=2.5):
    R_in = R_tot[0]
    R_out = R_tot[1]
    return(2*(tau/1000)*R_in + 2*(tau/1000)*R_out)

def economo_viol_time_eq_adj(R_tot, tau=2.5):
    R_in = R_tot[0]
    R_out = R_tot[1]
    return(1.97*(tau/1000)*R_in + 1.93*(tau/1000)*R_out)

R_tot = []
for i in R_in_range:
    for j in R_out_range:
        R_tot.append([i, j])

viol_times_pred = []
for val in R_tot:
    viol_times_pred.append(economo_viol_time_eq(val))
    
viol_times_pred_adj = []
for val in R_tot:
    viol_times_pred_adj.append(economo_viol_time_eq_adj(val))

# %%
viol_times = np.zeros((len(R_in_range), len(R_out_range)))
for i, R_in in enumerate(R_in_range):
    for j, R_out in enumerate(R_out_range):
        viol_times[i, j] = neuronsim.sim_Fv_times(R_in, R_out)[3]
        
# %%

windows = np.zeros((len(R_in_range), len(R_out_range)))
R_tots = np.zeros((len(R_in_range), len(R_out_range)))
for i, R_in in enumerate(R_in_range):
    for j, R_out in enumerate(R_out_range):
        fr = R_in + R_out
        windows[i, j] = fr*2*0.0025
        R_tots[i, j] = fr

viol_times_avg = []
windows_avg = []
windows_std = []
for val in np.unique(windows):
    idx_mask = (windows == val)
    viol_times_avg.append(np.mean(viol_times[idx_mask]))
    windows_avg.append(val)
    windows_std.append(np.std(viol_times[idx_mask]))
    
    
x = [0, 0.42]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel('Violation Window Actual', fontsize=14)
plt.ylabel('Violation Window Predicted', fontsize=14)
plt.title('Violation Window Prediction', fontsize=18)

plt.scatter(viol_times_avg, windows_avg, s=30)
plt.errorbar(viol_times_avg, windows_avg, xerr=windows_std, ls='none')
plt.tight_layout()

# %%

err = windows - viol_times

err_avg = []
R_tots_avg = []
err_std = []
for val in np.unique(R_tots):
    idx_mask = (R_tots == val)
    err_avg.append(np.mean(err[idx_mask]))
    R_tots_avg.append(val)
    err_std.append(np.std(err[idx_mask]))

plt.scatter(R_tots_avg, err_avg)

plt.errorbar(R_tots_avg, err_avg, yerr=err_std, ls='none')

plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('Violation Window Error', fontsize=14)
plt.title('Error vs. $R_{tot}$', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# %%

R_outs = np.tile(R_out_range, (len(R_in_range), 1))
cmap = mpl.cm.hot
plt.scatter(R_tots, err, c=R_outs/R_tots, cmap=cmap)

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('Violation Window Error', fontsize=14)
plt.title('Error vs. $R_{tot}$ and $R_{out}$', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

cbar.set_label('$R_{out}$ proportion', rotation=270, fontsize=14)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

# %%

err = windows - viol_times
cmap = mpl.cm.hot
plt.contourf(R_in_range, R_out_range, np.transpose(err), cmap=cmap)

norm = mpl.colors.Normalize(vmin=np.amin(err), vmax=np.amax(err))
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))


plt.xlabel('$R_{in}$ (Hz)', fontsize=14)
plt.ylabel('$R_{out}$ (Hz)', fontsize=14)
plt.title('Overlap Effect', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar.set_label('Violation Window Error', rotation=270, fontsize=14)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

        
# %%

x = [0, 0.13]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

plt.scatter(viol_times, viol_times_pred, s=20)

# %% 

x = [0, 0.13]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

plt.scatter(viol_times, viol_times_pred_adj, s=20)

# %%

R_in = [el[0] for el in R_tot]
R_out = [el[1] for el in R_tot]

plt.scatter(R_in, viol_times)

def func(x, a, b):
    return (a*0.0025*x[0] + b*0.0025*x[1])

popt, pcov = curve_fit(func, np.array(R_tot[:-20]).T, np.array(viol_times[:-20]))
print(popt)

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(R_in, R_out, viol_times)