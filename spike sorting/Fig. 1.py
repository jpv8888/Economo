# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:05:28 2022

@author: jpv88
"""

import scipy.stats

from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
import numpy as np

import JV_utils
import neuronsim

# %% Data generation

FDRs = [0.03, 0.1, 0.3]
rates = [2, 5, 10]

for i in FDRs:
    for j in rates:
        Fv, ISIs, Rtot = neuronsim.sim_Fv_Fig1(j, FDR=i, t_stop=10000, N=1000)
        ISIs = np.concatenate(ISIs)
        fig, ax = plt.subplots()
        bins = np.arange(0, np.max(ISIs) + 1, 0.0005)
        plt.hist(ISIs, bins, density=True)
        plt.xlim(0, 0.04)
        plt.title('Fv = {}'.format(Fv))
        fig.patches.extend([plt.Rectangle((0, 0), 0.0025, ax.get_ylim()[1], color='r', 
                                  alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])
        file_name = 'FDR = {}, fr = {}.png'.format(i, j)
        plt.savefig(file_name)
        

# %%

hist = np.histogram(ISIs, bins=1000)
hist_dist = scipy.stats.rv_histogram(hist)
X = np.linspace(0, 0.04, 1000)
plt.plot(X, hist_dist.pdf(X))


# %%
fig, ax = plt.subplots()
bins = np.arange(0, np.max(ISIs) + 1, 0.0005)
plt.hist(ISIs, bins, density=True)
plt.xlim(0, 0.04)
fig.patches.extend([plt.Rectangle((0, 0), 0.0025, ax.get_ylim()[1], color='r', 
                                  alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

rates = np.arange(0, 21, 1)
FDRs = np.arange(0, 1.1, 0.1)
sim_Fv = []
rates_full = []
FDRs_full = []

for i in FDRs:
    for j in rates:
        Fv, ISIs, Rtot = neuronsim.sim_Fv_Fig1(j, FDR=i, t_stop=1000, N=100,
                                               out_refrac=2.5)
        sim_Fv.append(Fv)
        rates_full.append(j)
        FDRs_full.append(i)
        
sim_Fv = [el*100 for el in sim_Fv]

# %%

f = interp2d(rates_full, sim_Fv, FDRs_full, kind="cubic", bounds_error=False)

FDR_full = f(rates, np.linspace(0, 5, 0.01))
FDR_full = np.ma.masked_invalid(FDR_full)
xv, yv = np.meshgrid(rates, sim_Fv)
fig, ax = plt.subplots()
levels = np.linspace(0, 1, 11)
yv = yv*100
mesh = plt.contourf(xv, yv, FDR_full, cmap='jet', levels=levels)
plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('$F_{v}$ (%)', fontsize=14)
mesh.set_clim(0, 1)
plt.title('N = ∞ (Eq)', fontsize=18)
cb = plt.colorbar(mesh, ax=ax)
cb.set_label(label='FDR', size=14)
cb.ax.tick_params(labelsize=12)
ax.patch.set(hatch='x', edgecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

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


@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025):

    
    a = -2*tviol*Rtot
    b = 2*tviol*Rtot
    c = -F_v
    

    
    if Rtot != 0:
        FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    else:
        FDR = 0
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 0.5

    return FDR



FDR_eq = []
for i in range(len(rates_full)):
    if rates_full[i] == 0:
        FDRs_full[i] = 0
    FDR_eq.append(kleinfeld_eq(rates_full[i], sim_Fv[i]*0.01).item())
    

for i, e in reversed(list(enumerate(FDR_eq))):
    if np.isnan(e):
        del sim_Fv[i]
        del rates_full[i]
        del FDRs_full[i]
        del FDR_eq[i]
    elif FDRs_full[i] > 0.5:
        del sim_Fv[i]
        del rates_full[i]
        del FDRs_full[i]
        del FDR_eq[i]
        

pred, reg = JV_utils.lin_reg(FDRs_full, FDR_eq)

fig, ax = plt.subplots()
plt.scatter(FDRs_full, FDR_eq, c=sim_Fv)
plt.plot(FDRs_full, pred, lw=3, c='k')
plt.plot([0, 0.5], [0, 0.5], lw=3, ls='--', c='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.xlabel('FDR (sim)', fontsize=16)
plt.ylabel('FDR (eq)', fontsize=16)
plt.title('Kleinfeld Equation Accuracy', fontsize=18)
plt.text(0.3, 0.1, '$R^{2}$ = 0.99', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# %%

rates = np.arange(0, 21, 1)
tau = np.arange(0, 2.5, 0.3)

rates_full = []
tau_full = []
sim_Fv = []
for i in tau:
    for j in rates:
        Fv, ISIs, Rtot = neuronsim.sim_Fv_Fig1(j, FDR=0.1, t_stop=1000, N=100,
                                               out_refrac=i)
        sim_Fv.append(Fv)
        rates_full.append(j)
        tau_full.append(i)

sim_Fv = [el*100 for el in sim_Fv]

# %%



economo_FDRs = []
kleinfeld_FDRs = []
for i in range(len(sim_Fv)):
    economo_FDRs.append(economo_eq(rates_full[i], sim_Fv[i]*0.01).item())
    kleinfeld_FDRs.append(kleinfeld_eq(rates_full[i], sim_Fv[i]*0.01).item())
    
fig, ax = plt.subplots()
plt.scatter(rates_full, economo_FDRs, c='b', s=4, label='Economo')
plt.scatter(rates_full, kleinfeld_FDRs, c='r', s=4, label='Kleinfeld')
plt.hlines(0.1, 0, 20, lw=3, ls='--', colors='black')
plt.ylim(0.05, 0.15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$R_{tot}$ (Hz)', fontsize=16)
plt.ylabel('Predicted FDR', fontsize=16)
plt.legend(prop={'size': 16})
plt.title('True FDR = 0.1', fontsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()



        
# %%
        
fig, ax = plt.subplots()
mesh = ax.tripcolor(rates_full, sim_Fv, FDRs_full, shading='gouraud', cmap='inferno')
# plt.scatter(rates_full, sim_Fv, c=FDRs_full, cmap='inferno')
plt.colorbar(mesh, ax=ax)
plt.xlim(0, 20)
plt.ylim(0, 2)

# %%

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
        FDR = float('NaN')
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

Rtot = np.arange(0, 20.1, 0.001)
F_v = np.arange(0, 0.0501, 0.0001)

# FDR = []
# Rtot_full = []
# F_v_full = []
# for i in Rtot:
#     for j in F_v:
#         FDR.append(economo_eq(i, j))
#         Rtot_full.append(i)
#         F_v_full.append(j)

# F_v_full = [el*100 for el in F_v_full]

xv, yv = np.meshgrid(Rtot, F_v)
fig, ax = plt.subplots()
FDR_full = economo_eq(xv, yv)
FDR_full = np.ma.masked_invalid(FDR_full)
levels = np.linspace(0, 1, 11)
yv = yv*100
mesh = plt.contourf(xv, yv, FDR_full, cmap='jet', levels=levels)
plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('$F_{v}$ (%)', fontsize=14)
mesh.set_clim(0, 1)
plt.title('N = ∞ (Eq)', fontsize=18)
cb = plt.colorbar(mesh, ax=ax)
cb.set_label(label='FDR', size=14)
cb.ax.tick_params(labelsize=12)
ax.patch.set(hatch='x', edgecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
#plt.xlim(0, 20)
#plt.ylim(0, 2)

# %%

Rtot = np.arange(0, 20.1, 0.001)
F_v = np.arange(0, 0.0501, 0.0001)


xv, yv = np.meshgrid(Rtot, F_v)
fig, ax = plt.subplots()
FDR_full = economo_eq(xv, yv)
plt.pcolormesh(xv, yv, FDR_full)
FDR_full = np.ma.masked_invalid(FDR_full)
levels = np.linspace(0, 1, 11)
yv = yv*100
mesh = plt.pcolormesh(xv, yv, FDR_full, cmap='jet', shading='gouraud')
plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('$F_{v}$ (%)', fontsize=14)
mesh.set_clim(0, 1)
plt.title('N = ∞ (Eq)', fontsize=18)
cb = plt.colorbar(mesh, ax=ax)
cb.set_label(label='FDR', size=14)
cb.ax.tick_params(labelsize=12)
ax.patch.set(hatch='..', edgecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025):

    
    a = -2*tviol*Rtot
    b = 2*tviol*Rtot
    c = -F_v
    

    
    if Rtot != 0:
        FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    else:
        FDR = float('NaN')
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

Rtot = np.arange(0, 20.1, 0.001)
F_v = np.arange(0, 0.0501, 0.0001)


xv, yv = np.meshgrid(Rtot, F_v)
fig, ax = plt.subplots()
FDR_full = kleinfeld_eq(xv, yv)
plt.pcolormesh(xv, yv, FDR_full)
FDR_full = np.ma.masked_invalid(FDR_full)
levels = np.linspace(0, 1, 11)
yv = yv*100
mesh = plt.pcolormesh(xv, yv, FDR_full, cmap='jet', shading='gouraud')
plt.xlabel('$R_{tot}$ (Hz)', fontsize=14)
plt.ylabel('$F_{v}$ (%)', fontsize=14)
mesh.set_clim(0, 1)
plt.title('N = 1 (Eq)', fontsize=18)
cb = plt.colorbar(mesh, ax=ax)
cb.set_label(label='FDR', size=14)
cb.ax.tick_params(labelsize=12)
ax.patch.set(hatch='..', edgecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

