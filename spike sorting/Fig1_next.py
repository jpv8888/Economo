# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:09:45 2022

@author: jpv88
"""

import neuronsim
import numpy as np
import matplotlib.pyplot as plt
import JV_utils

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

# %%

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    Fv = Rviol/(Rin + Rout)

    return Fv

def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    Fv = Rviol/(Rin + Rout)

    return Fv

Rtot = [1, 4, 16]
FDRs = np.arange(0, 1.1, 0.1)

economo_sim_Fv = np.zeros((len(FDRs), len(Rtot)))
kleinfeld_sim_Fv = np.zeros((len(FDRs), len(Rtot)))

for i, val in enumerate(FDRs):
    for j, val2 in enumerate(Rtot):
        economo_sim_Fv[i,j] = neuronsim.sim_Fv_Fig1(val2, FDR=val, 
                                                    t_stop=1000)[0]
        kleinfeld_sim_Fv[i,j] = neuronsim.sim_Fv_Fig1(val2, FDR=val, 
                                                      t_stop=1000, 
                                                      out_refrac=2.5)[0]


FDRs_2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]
sim_Fv_N_2 = np.zeros((len(FDRs_2), len(Rtot)))
for i, val in enumerate(FDRs_2):
    for j, val2 in enumerate(Rtot):
        sim_Fv_N_2[i,j] = neuronsim.sim_Fv_neurons(val2, FDR=val, neurons=2, 
                                                 t_stop=1000)
        
FDRs_5 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.83]
sim_Fv_N_5 = np.zeros((len(FDRs_5), len(Rtot)))
for i, val in enumerate(FDRs_5):
    for j, val2 in enumerate(Rtot):
        sim_Fv_N_5[i,j] = neuronsim.sim_Fv_neurons(val2, FDR=val, neurons=5, 
                                                 t_stop=1000)
    
    
# %%

fig, ax = plt.subplots()

for idx in range(0, 3):

    R_out = FDRs*Rtot[idx]
    R_in = Rtot[idx] - R_out
    
    plt.scatter(FDRs, np.array(economo_sim_Fv[:,idx])*100, s=20, c='darkblue')
    plt.scatter(FDRs[:6], np.array(kleinfeld_sim_Fv[:6,idx])*100, s=20, c='darkred')
    plt.plot(FDRs, economo_Fv(R_in, R_out)*100, lw=3, c='blue', label='N = ∞')
    plt.plot(FDRs[:6], kleinfeld_Fv(R_in[:6], R_out[:6])*100, lw=3, c='red', label='N = 1')
    # plt.scatter(FDR_N, np.array(sim_Fv_N[:,idx])*100, s=20, c='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(-0.2, 4.2)
    # bottom_line = list(kleinfeld_Fv(R_in[:6], R_out[:6])*100) + \
        # list(np.array(sim_Fv_N[:,idx])*100) + [economo_Fv(R_in[-1], R_out[-1])*100]
    # plt.plot(FDRs[5:], bottom_line[5:], lw=3, c='black')
    # ax.fill_between(FDRs, economo_Fv(R_in, R_out)*100, bottom_line, 
                    # facecolor="none", hatch="+", edgecolor="k", linewidth=0.0, 
                    # alpha=0.2)

    
    path = Path(np.vstack((np.vstack((FDRs[:6], kleinfeld_Fv(R_in[:6], R_out[:6])*100)).T, 
                       [FDRs[5], kleinfeld_Fv(R_in[5], R_out[5])*100], 
                       [FDRs[-1], economo_Fv(R_in[-1], R_out[-1])*100],
                       np.flip(np.vstack((FDRs, economo_Fv(R_in, R_out)*100)).T, axis=0))))
    patch = PathPatch(path, facecolor ='none', edgecolor='none')
    
    ax.add_patch(patch)
    # im = ax.imshow([[0.,1.], [0.,1.]], interpolation ='bilinear', cmap = plt.cm.gray,
    #                clip_path = patch, clip_on = True)
    
    lims = [ax.get_xlim(), ax.get_ylim()]
    lims = [item for t in lims for item in t]
    im = ax.imshow([[1.,0.], [1.,0.]], interpolation ='bicubic', cmap = 'Purples',
                   extent=lims, aspect='auto', clip_path = patch, clip_on = True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('FDR', fontsize=16)
# plt.title('$R_{tot}$ = 16 Hz', fontsize=18)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
# plt.legend(prop={'size': 16})

for col in sim_Fv_N_2.T:
    plt.plot(FDRs_2, col*100, c='black', alpha=0.2, ls='--')
for col in sim_Fv_N_5.T:
    plt.plot(FDRs_5, col*100, c='black', alpha=0.2, ls='--')
    
# plt.text(1.05, 0.15, '1 Hz', fontsize='16')
# plt.text(0.8, 1.4, '10 Hz', fontsize='16')
# plt.text(0.8, 3.5, '20 Hz', fontsize='16')
plt.text(1.05, 0.15, '1 Hz', fontsize='16')
plt.text(1.05, 0.95, '4 Hz', fontsize='16')
plt.text(1.05, 3.9, '16 Hz', fontsize='16')
plt.tight_layout()

# %%

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv

def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv


FDRs = [0.05, 0.20, 0.5]
Rtot = np.arange(0, 21, 1)

economo_sim_Fv = np.zeros((len(FDRs), len(Rtot)))
kleinfeld_sim_Fv = np.zeros((len(FDRs), len(Rtot)))

for i , val in enumerate(Rtot):
    for j, val2 in enumerate(FDRs):
        economo_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000)[0]
        kleinfeld_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000, 
                                                      out_refrac=2.5)[0]
    
# %%

F_v_economo = np.zeros((len(FDRs), len(Rtot)))
F_v_kleinfeld = np.zeros((len(FDRs), len(Rtot)))

for i, val in enumerate(FDRs):
    R_out = val*Rtot
    R_in = Rtot - R_out
    for j, val2 in enumerate(range(len(R_out))):
        F_v_economo[i, j] = economo_Fv(R_in[j], R_out[j])*100
        F_v_kleinfeld[i, j] = kleinfeld_Fv(R_in[j], R_out[j])*100
        
fig, ax1 = plt.subplots()
plt.scatter(Rtot, economo_sim_Fv[0,:]*100, s=20, c='darkblue')
plt.scatter(Rtot, kleinfeld_sim_Fv[0,:]*100, s=20, c='darkblue')
plt.scatter(Rtot, economo_sim_Fv[1,:]*100, s=20, c='darkgreen')
plt.scatter(Rtot, kleinfeld_sim_Fv[1,:]*100, s=20, c='darkgreen')
plt.scatter(Rtot, economo_sim_Fv[2,:]*100, s=20, c='darkred')
plt.scatter(Rtot, kleinfeld_sim_Fv[2,:]*100, s=20, c='darkred')
plt.plot(Rtot, F_v_economo[0,:], lw=3, c='blue')
plt.plot(Rtot, F_v_kleinfeld[0,:], lw=3, c='blue')
plt.plot(Rtot, F_v_economo[1,:], lw=3, c='green')
plt.plot(Rtot, F_v_kleinfeld[1,:], lw=3, c='green')
plt.plot(Rtot, F_v_economo[2,:], lw=3, c='red')
plt.plot(Rtot, F_v_kleinfeld[2,:], lw=3, c='red')

ax1.fill_between(Rtot, F_v_economo[0,:], F_v_kleinfeld[0,:], facecolor="none", 
                hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
ax1.fill_between(Rtot, F_v_economo[1,:], F_v_kleinfeld[1,:], facecolor="none", 
                hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
ax1.fill_between(Rtot, F_v_economo[2,:], F_v_kleinfeld[2,:], facecolor="none", 
                hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax2 = ax1.twiny() # ax1 and ax2 share y-axis
ax3 = ax1.twinx()
ax2.set_xticks([0, 1, 2, 3, 4 ,5])
ax1.set_xticks([0, 4, 8, 12, 16, 20])
ax2.set_xticklabels([0, 1, 2, 3, 4 ,5], fontsize=14)
ax1.set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=14)
ax2.set_xlim(-0.25, 5.25)
ax1.set_ylim(-0.1875, 4.1875)
ax1.set_xlabel('$R_{tot}$ (Hz)', fontsize=16)
ax1.set_ylabel('% $ISI_{viol}$', fontsize=16)
# plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
left_ticks= [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]
ax1.set_yticks(left_ticks)
ax1.set_yticklabels(left_ticks, fontsize=14)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)
ax3.set_ylim(0-(3/64), 1+(3/64))
right_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax3.set_yticks(right_ticks)
ax3.set_yticklabels(right_ticks, fontsize=14)

ax1.spines['right'].set_color('violet')
ax2.spines['right'].set_color('violet')
ax3.spines['right'].set_color('violet')
ax2.spines['top'].set_color('violet')
ax1.spines['top'].set_color('violet')
ax3.spines['top'].set_color('violet')
ax2.tick_params(axis='x', colors='violet')
ax3.tick_params(axis='y', colors='violet')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

#plt.xscale('log')
ax1.grid(axis='both', which='both', ls ='--', alpha=0.3)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c='k'),
                Line2D([0], [0], c='k', ls='dashed')]
# ax.legend(custom_lines, ['N = ∞', 'N = 1'], prop={'size': 16})
# plt.text(21, 3, '50%', fontsize='16')
# plt.text(21, 1.6, '20%', fontsize='16')
# plt.text(21, 0.4, '5%', fontsize='16')
#plt.text(5.1, 0.75, '50%', fontsize='16')
#plt.text(5.1, 0.4, '20%', fontsize='16')
#plt.text(5.1, 0.1, '5%', fontsize='16')
#plt.xlim(-1.0, 21.0)
# plt.ylim(0, 1)

plt.tight_layout()

# %%

Rtot = 5
tstop = np.array([100, 300, 500, 700, 1000, 2000, 4000, 6000, 8000])
sim_Fv_dist = []
FDR = 0.5
for t in tstop:
    sim_Fv_dist.append(neuronsim.sim_Fv_dist(Rtot, t_stop=t, N=1000, FDR=FDR))
    
true_Fv = np.mean(neuronsim.sim_Fv_dist(Rtot, t_stop=10000, N=1000, FDR=FDR))
true_Fv_bounds = [0.95*true_Fv, 1.05*true_Fv]

# %%

low_bound = []
up_bound = []
for dist in sim_Fv_dist:
    dist = np.sort(dist)
    interval = [dist[49], dist[949]]
    low_bound.append(interval[0]/true_Fv - 1)
    up_bound.append(interval[1]/true_Fv - 1)
    
CV = [np.std(el)/np.mean(el) for el in sim_Fv_dist]

def func(x, a, b, c):
    return a*(np.exp(-b*x)) + c

popt, pcov = curve_fit(func, np.array(tstop*Rtot), np.array(CV), maxfev=100000)

@np.vectorize
def func(x, a, b, c):
    return a*(np.exp(-b*x)) + c

x = np.linspace(0, 50000, 1000)
y_pred = func(x, popt[0], popt[1], popt[2])
    
fig, ax = plt.subplots() 
# plt.scatter(tstop*Rtot, low_bound, s=20, c='blue')
# plt.scatter(tstop*Rtot, up_bound, s=20, c='red')
plt.scatter(tstop*Rtot, CV, s=20, c='black')
plt.hlines([0.05, 0.1], ax.get_xlim()[0], ax.get_xlim()[1], ls='--')
plt.xlabel('# Spikes', fontsize=16)
# plt.ylabel('$ISI_{viol}$', fontsize=16)
plt.ylabel('CV', fontsize=16)
plt.plot(x, y_pred)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0)
plt.tight_layout()

# %%
sim_Fv = []
FDRs = np.arange(0, 1.01, 0.01)
for FDR in FDRs:
    sim_Fv.append(neuronsim.sim_Fv_neurons(16, FDR=FDR, neurons=3))
    
# %%
ISIs = []
ISIs.append(np.concatenate(neuronsim.sim_Fv_Fig1(1, FDR=0.1, t_stop=36)[1]))
ISIs.append(ISIs[0]*-1)
ISIs = np.concatenate(ISIs)
ISIs = ISIs*1000

bins = np.arange(-25.5, 25.5, 0.5)
hist = np.histogram(ISIs[0], bins=bins)[0]

# %%
Rtots = [1, 4, 16]
FDRs = [0.05, 0.25, 0.5]
fig, ax = plt.subplots(3, 3)
for i, Rtot in enumerate(Rtots):
    for j, FDR in enumerate(FDRs):
        ISIs = []
        ISIs.append(np.concatenate(neuronsim.sim_Fv_Fig1(Rtot, FDR=FDR, N=1,
                                                         t_stop=3600)[1]))
        ISIs2 = np.concatenate(ISIs)
        Fv = sum(ISIs2 < 0.0025)/len(ISIs2)
        ISIs.append(ISIs[0]*-1)
        ISIs = np.concatenate(ISIs)
        ISIs = ISIs*1000
        
        bins = np.arange(-25.5, 25.5, 0.5)
        ax[i,j].hist(ISIs, bins, edgecolor='black', zorder=3)
        plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)
        fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax[i,j].get_ylim()[1],
                                          color='r', alpha=0.2, zorder=1000, 
                                          fill=True, transform=ax[i,j].transData)])
        ax[i,j].set_title('$ISI_{viol} = $' + str(round(Fv*100, 1)) + '%', 
                          fontsize=18)


plt.tight_layout()
        
# %%
plt.hist(ISIs, bins, edgecolor='black', zorder=3)
# plt.yscale('log')
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)
fig.patches.extend([plt.Rectangle((-2.5, 1), 5, ax.get_ylim()[1], color='r', 
                          alpha=0.2, zorder=1000, 
                          fill=True, transform=ax.transData)])
plt.xlim(-25, 25)
ax.grid(zorder=0)
plt.ylim(1)
plt.xlabel('ISI (ms)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title('FDR = 10%, $R_{tot}$ = 2 Hz', fontsize=18)
plt.tight_layout()

# %%

def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    # if isinstance(FDR, complex):
    #     FDR = 1

    return FDR


def kleinfeld_eq(Rtot, F_v, tviol=0.0025):

    
    a = -2*tviol*Rtot
    b = 2*tviol*Rtot
    c = -F_v
    
    FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

def economo_ob(F_v, args):
    Rtot, FDR = args
    return abs(FDR - economo_eq(Rtot, F_v))
    

Rtots = np.arange(1, 21, 1)
FDRs = np.linspace(0.01, 0.5, num=20)
tol = 0.025

F_v_opt = []
for Rtot in Rtots:
    for FDR in FDRs:
        u_tol = FDR + tol
        l_tol = FDR - tol
        temp = []
        u_bound = Rtot*0.0025
        temp.append(minimize_scalar(economo_ob, args=[Rtot, l_tol], 
                                    bounds=[0, u_bound], method='bounded').x)
        temp.append(minimize_scalar(economo_ob, args=[Rtot, FDR], 
                                    bounds=[0, u_bound], method='bounded').x)
        temp.append(minimize_scalar(economo_ob, args=[Rtot, u_tol], 
                                    bounds=[0, u_bound], method='bounded').x)
        F_v_opt.append(temp)

CVs = []
for tols in F_v_opt:
    temp = 0
    temp += abs(tols[1]- tols[0])
    temp += abs(tols[2] - tols[1])
    temp /= 2
    CVs.append(temp/tols[1])

idx = 0
CVs_opt = np.zeros((len(Rtots), len(FDRs)))
for i, Rtot in enumerate(Rtots):
    for j, FDR in enumerate(FDRs):
        CVs_opt[i,j] = CVs[idx]
        idx += 1
        
heatmap = plt.imshow(CVs_opt)
x_positions = np.arange(0,20,3)
y_positions = np.arange(0,20,1)
plt.xticks(x_positions, [round(el,2) for el in FDRs[::3]])
plt.yticks(y_positions, Rtots)

plt.xlabel('FDR', fontsize=16)
plt.ylabel('Rtot', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cbar = plt.colorbar(heatmap)
cbar.set_label('CV of $ISI_{viol}$')
plt.title('FDR +/- 1%', fontsize=18)
# plt.title('FDR = 10%, $R_{tot}$ = 2 Hz', fontsize=18)
plt.tight_layout()

# %% Mike's ([N-1]/N idea)

@np.vectorize
def economo_Fv_mod(Rin, Rout, N, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + ((N-1)/N)*0.5*(Rout**2)*2*tviol
    
    if Rin + Rout != 0:
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv

@np.vectorize
def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv

@np.vectorize
def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv

N = 6
FDRs = np.arange(0, 1.1, 0.1)
Rtot = 20

Rout = FDRs*Rtot
Rin = Rtot - Rout
eq_Fv = economo_Fv_mod(Rin, Rout, [N]*len(Rout))
old_economo_eq_Fv = economo_Fv(Rin, Rout)
old_kleinfeld_eq_Fv = kleinfeld_Fv(Rin, Rout)

sim_Fv = []
for FDR in FDRs:  
    sim_Fv.append(neuronsim.sim_Fv_neurons(Rtot, FDR=FDR, neurons=N, 
                                           t_stop=1000))

sim_Fv = np.array(sim_Fv)*100
eq_Fv = np.array(eq_Fv)*100
old_economo_eq_Fv = np.array(old_economo_eq_Fv)*100
old_kleinfeld_eq_Fv = np.array(old_kleinfeld_eq_Fv)*100
 
# %%

fig, ax = plt.subplots()
plt.scatter(FDRs, sim_Fv, label='sim')
plt.plot(FDRs, eq_Fv, label='new eq')
plt.plot(FDRs, old_economo_eq_Fv, label='old_economo_eq')
plt.plot(FDRs, old_kleinfeld_eq_Fv, label='old_kleinfeld_eq')

# bounds = [0, max([max(sim_Fv), max(eq_Fv)])]
# plt.plot(bounds, bounds, ls='--')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('FDR', fontsize=16)
plt.ylabel('%$ISI_{viol}$', fontsize=16)
plt.title('Rtot = 20, N = 6', fontsize=18)
plt.legend()
plt.tight_layout()
