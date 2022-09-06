# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:09:45 2022

@author: jpv88
"""

import neuronsim
import numpy as np
import matplotlib.pyplot as plt

# %%

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    Fv = Rviol/(Rin + Rout)

    return Fv

def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    Fv = Rviol/(Rin + Rout)

    return Fv

Rtot = 1
FDRs = np.arange(0, 0.6, 0.1)

economo_sim_Fv = []
kleinfeld_sim_Fv = []

for i in FDRs:
    economo_sim_Fv.append(neuronsim.sim_Fv_Fig1(Rtot, FDR=i, t_stop=10000)[0])
    kleinfeld_sim_Fv.append(neuronsim.sim_Fv_Fig1(Rtot, FDR=i, t_stop=10000, 
                                                  out_refrac=2.5)[0])
    
# %%
R_out = FDRs*Rtot
R_in = Rtot - R_out
fig, ax = plt.subplots()
plt.scatter(FDRs, np.array(economo_sim_Fv)*100, s=20, c='darkblue')
plt.scatter(FDRs, np.array(kleinfeld_sim_Fv)*100, s=20, c='darkred')
plt.plot(FDRs, economo_Fv(R_in, R_out)*100, lw=3, c='blue', label='N = ∞')
plt.plot(FDRs, kleinfeld_Fv(R_in, R_out)*100, lw=3, c='red', label='N = 1')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.fill_between(FDRs, economo_Fv(R_in, R_out)*100, 
                kleinfeld_Fv(R_in, R_out)*100, facecolor="none", hatch="+", 
                edgecolor="k", linewidth=0.0, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('FDR', fontsize=16)
plt.title('$R_{tot}$ = 1 Hz', fontsize=18)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
plt.legend(prop={'size': 16})
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


FDR = 0.25
Rtot = np.arange(0, 21, 1)

economo_sim_Fv = []
kleinfeld_sim_Fv = []

for i in Rtot:
    economo_sim_Fv.append(neuronsim.sim_Fv_Fig1(i, FDR=FDR, t_stop=10000)[0])
    kleinfeld_sim_Fv.append(neuronsim.sim_Fv_Fig1(i, FDR=FDR, t_stop=10000, 
                                                  out_refrac=2.5)[0])
    
# %%

R_out = FDR*Rtot
R_in = Rtot - R_out
fig, ax = plt.subplots()

F_v_economo = []
F_v_kleinfeld = []
for i in range(len(R_out)):
    F_v_economo.append(economo_Fv(R_in[i], R_out[i])*100)
    F_v_kleinfeld.append(kleinfeld_Fv(R_in[i], R_out[i])*100)
plt.scatter(Rtot, np.array(economo_sim_Fv)*100, s=20, c='darkblue')
plt.scatter(Rtot, np.array(kleinfeld_sim_Fv)*100, s=20, c='darkred')
plt.plot(Rtot, F_v_economo, lw=3, c='blue', label='N = ∞')
plt.plot(Rtot, F_v_kleinfeld, lw=3, c='red', label='N = 1')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.fill_between(Rtot, F_v_economo, 
                F_v_kleinfeld, facecolor="none", hatch="+", 
                edgecolor="k", linewidth=0.0, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('$R_{tot}$ (Hz)', fontsize=16)
plt.title('FDR = 25%', fontsize=18)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
plt.legend(prop={'size': 16})
plt.tight_layout()