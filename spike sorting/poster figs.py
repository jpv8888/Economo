# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:38:46 2022

@author: jpv88
"""
import random

import JV_utils
import neuronsim

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy import linalg as LA
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# %% Fig. 1: FNR and FDR
# PSTH preprocessing

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
PSTHs[387,76:100] = np.mean(PSTHs[387,76])*24 + np.random.normal(np.mean(PSTHs[387,60:80]), 
                                                                 0.1, [24,]) 

unit_PSTHs = []

t = np.linspace(0, 6, 100)
t_new = np.linspace(0, 6, 1000)

for i in range(len(PSTHs)):
    unit = PSTHs[i,:]/LA.norm(PSTHs[i,:])
    smoothed = gaussian_filter1d(unit, 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    unit_PSTHs.append(y_new)

# %% PSTH selection random

a, b = random.sample(range(len(unit_PSTHs)), 2)

plt.vlines([204, 729], 0, 0.2)
plt.plot(unit_PSTHs[a])
plt.plot(unit_PSTHs[b])

# sample cue is at sample 204, go cue is at sample 729
# putative a's = 206, 244, 202, 687
# putative b's = 387

# %% PSTH selection guided

sample_idxs = []
for i in range(len(unit_PSTHs)):
    if (np.argmax(unit_PSTHs[i]) >= 204) & (np.argmax(unit_PSTHs[i]) <= 304):
        sample_idxs.append(i)

plt.vlines([204], 0, 0.2)
plt.plot(unit_PSTHs[sample_idxs[4]])

# %% final selection

plt.vlines([204, 729], 0, 0.2)
plt.plot(unit_PSTHs[206])
plt.plot(unit_PSTHs[244])
plt.plot(unit_PSTHs[202])
plt.plot(unit_PSTHs[687])

fig, ax = plt.subplots()
plt.vlines([204, 729], 0, 0.2)
plt.plot(unit_PSTHs[687])
plt.plot(unit_PSTHs[387])

# final choices are 687 and 387
# actually I'm just gonna use the flipped version of true as the rogue (687)

# %% plotting testing

def preprocess(idx, sigma):
    unit = PSTHs[idx,:]/LA.norm(PSTHs[idx,:])
    smoothed = gaussian_filter1d(unit, sigma)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    return y_new

true_neuron = preprocess(687, 5)
rogue_neuron = preprocess(387, 5)
true_neuron = true_neuron*100
rogue_neuron = rogue_neuron*85
# rogue_neuron[760:] = np.flip(rogue_neuron[520:760])
# rogue_neuron = gaussian_filter1d(rogue_neuron, 2)

fig, ax = plt.subplots()
plt.vlines([204, 729], 0, 10)
plt.plot(true_neuron)
plt.plot(rogue_neuron)

# %% final plotting

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

fig, ax = plt.subplots()

plt.plot(t_new, true_neuron, lw=2)
bar = AnchoredSizeBar(ax.transData, 1, '1 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)
bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)
plt.vlines([t_new[204], t_new[729]], ax.get_ylim()[0], ax.get_ylim()[1], 
           colors='k', ls='--')

lims = [ax.get_ylim()[0], ax.get_ylim()[1]]

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
# plt.axis('off')

# %%

spikes = neuronsim.sim_spikes_PSTH(true_neuron, rogue_neuron, N=300, FDR=0)
spikes = np.concatenate(spikes)
half_num = int(len(spikes)/2)
spikes = np.random.choice(spikes, half_num, replace=False)
spikes = JV_utils.spikes_to_firing_rates(spikes, 300, N=120)
smoothed = gaussian_filter1d(spikes, 3)
t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)
f = interpolate.interp1d(t, smoothed, kind='cubic')
y_new = f(t_new)

fig, ax = plt.subplots()

plt.plot(t_new, y_new, ls='dashdot',lw=2)
ax.set_ylim(lims)
plt.vlines([t_new[204], t_new[729]], ax.get_ylim()[0], ax.get_ylim()[1], 
           colors='k', ls='--')


bar = AnchoredSizeBar(ax.transData, 1, '1 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)
bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

# %%

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

fig, ax = plt.subplots()

plt.plot(t_new, true_neuron, lw=2)
plt.plot(t_new, rogue_neuron, lw=2)
bar = AnchoredSizeBar(ax.transData, 1, '1 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)
bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)
plt.vlines([t_new[204], t_new[729]], ax.get_ylim()[0], ax.get_ylim()[1], 
           colors='k', ls='--')

lims = [ax.get_ylim()[0], ax.get_ylim()[1]]

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%

spikes = neuronsim.sim_spikes_PSTH(true_neuron, rogue_neuron, N=300, FDR=0)
spikes = np.concatenate(spikes)

false_spikes = neuronsim.sim_spikes_PSTH(rogue_neuron, true_neuron, N=300, FDR=0)
false_spikes = np.concatenate(false_spikes)

spikes = np.concatenate((spikes, false_spikes))
spikes = JV_utils.spikes_to_firing_rates(spikes, 300, N=120)
smoothed = gaussian_filter1d(spikes, 3)
t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)
f = interpolate.interp1d(t, smoothed, kind='cubic')
y_new = f(t_new)

fig, ax = plt.subplots()

plt.plot(t_new, y_new, ls='dashdot',lw=2)
ax.set_ylim(lims)
plt.vlines([t_new[204], t_new[729]], lims[0], lims[1], 
           colors='k', ls='--')


bar = AnchoredSizeBar(ax.transData, 1, '1 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)
bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

# %% infinity neurons

def Fv_from_FDR_Rtot(FDR, Rtot, tau=0.0025):
    return 2*tau*(1-FDR)*FDR*Rtot + tau*(FDR**2)*Rtot

def Rtot_from_FDR_Fv(FDR, Fv, tau=0.0025):
    return Fv/(2*tau*(1-FDR)*FDR + tau*(FDR**2))

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

def FDR_with_N(Rtot, N, Fv, tau=0.0025):
    a = ((N-1)/N)*tau*Rtot - 2*tau*Rtot
    b = 2*tau*Rtot
    c = -Fv
    
    predFDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    return predFDR

test = Fv_from_FDR_Rtot(0.5, 20)
test2 = Rtot_from_FDR_Fv(0.5, test)
test3 = economo_eq(20, 0.025)
test4 = FDR_with_N(20, 2, 0.025)

# %% 1 neuron

def Fv_from_FDR_Rtot(FDR, Rtot, tau=0.0025):
    return 2*tau*(1-FDR)*FDR*Rtot

def Rtot_from_FDR_Fv(FDR, Fv, tau=0.0025):
    return Fv/(2*tau*(1-FDR)*FDR)

test = Fv_from_FDR_Rtot(0.5, 20)
test2 = Rtot_from_FDR_Fv(0.2, test)

# %% ISI histograms

def Fv_calc(trials):
    viols = 0
    spks = 0
    for trial in trials:
        viols += sum(np.diff(trial) < 0.0025)
        spks += len(trial)
    
    return viols/spks

spks_low = neuronsim.sim_spikes(19, 1, N=100, t_stop=6, out_refrac=2.5)
Fv_low = Fv_calc(spks_low)
Rtot = len(np.concatenate(spks_low))/600

spks_low_subset = spks_low[:20]
for i, trial in enumerate(spks_low_subset):
    trial = np.array(trial)
    spks_low_subset[i] = trial[trial < 1]

fig, ax = plt.subplots()
ax.eventplot(spks_low_subset, colors='black', linewidths=1, linelengths=0.5)

bar = AnchoredSizeBar(ax.transData, 0.5, '0.5 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)

bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

ISIs_low = []
for trial in spks_low:
    ISIs_low.append(np.diff(trial))
ISIs_low = np.concatenate(ISIs_low)
ISIs_low = ISIs_low*1000
ISIs_low = np.concatenate((ISIs_low, ISIs_low*-1))
            
fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_low, bins, facecolor='blue', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_low*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_low, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

spks_high = neuronsim.sim_spikes(1.9, 1.9, N=100, t_stop=6, out_refrac=2.5)
Fv_high = Fv_calc(spks_high)
Rtot = len(np.concatenate(spks_high))/600

spks_high_subset = spks_high[:20]
for i, trial in enumerate(spks_high_subset):
    trial = np.array(trial)
    spks_high_subset[i] = trial[trial < 1]

fig, ax = plt.subplots()
ax.eventplot(spks_high_subset, colors='black', linewidths=1, linelengths=0.5)

bar = AnchoredSizeBar(ax.transData, 0.5, '0.5 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)

bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

ISIs_high = []
for trial in spks_high:
    ISIs_high.append(np.diff(trial))
ISIs_high = np.concatenate(ISIs_high)
ISIs_high = ISIs_high*1000
ISIs_high = np.concatenate((ISIs_high, ISIs_high*-1))

fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_high, bins, facecolor='red', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_high*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_high, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

spks_mid = neuronsim.sim_spikes(0.8*5.9375, 0.2*5.9375, N=100, t_stop=6, out_refrac=2.5)
Fv_mid = Fv_calc(spks_mid)
Rtot = len(np.concatenate(spks_mid))/600

spks_mid_subset = spks_mid[:20]
for i, trial in enumerate(spks_mid_subset):
    trial = np.array(trial)
    spks_mid_subset[i] = trial[trial < 1]

fig, ax = plt.subplots()
ax.eventplot(spks_mid_subset, colors='black', linewidths=1, linelengths=0.5)

bar = AnchoredSizeBar(ax.transData, 0.5, '0.5 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)

bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

ISIs_mid = []
for trial in spks_mid:
    ISIs_mid.append(np.diff(trial))
ISIs_mid = np.concatenate(ISIs_mid)
ISIs_mid = ISIs_mid*1000
ISIs_mid = np.concatenate((ISIs_mid, ISIs_mid*-1))

fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_mid, bins, facecolor='green', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_mid*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_mid, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %% now for different Ns


def Fv_calc(trials):
    viols = 0
    spks = 0
    for trial in trials:
        viols += sum(np.diff(trial) < 0.0025)
        spks += len(trial)
    
    return viols/spks

spks_low = neuronsim.sim_spikes(0.7071067811865476*20, 
                                0.2928932188134524*20, N=100, t_stop=6, out_refrac=0)
Fv_low = Fv_calc(spks_low)
Rtot = len(np.concatenate(spks_low))/600

ISIs_low = []
for trial in spks_low:
    ISIs_low.append(np.diff(trial))
ISIs_low = np.concatenate(ISIs_low)
ISIs_low = ISIs_low*1000
ISIs_low = np.concatenate((ISIs_low, ISIs_low*-1))
            
fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_low, bins, facecolor='blue', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_low*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_low, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

spks_high = neuronsim.sim_spikes(10, 10, N=100, t_stop=6, out_refrac=2.5)
Fv_high = Fv_calc(spks_high)
Rtot = len(np.concatenate(spks_high))/600

ISIs_high = []
for trial in spks_high:
    ISIs_high.append(np.diff(trial))
ISIs_high = np.concatenate(ISIs_high)
ISIs_high = ISIs_high*1000
ISIs_high = np.concatenate((ISIs_high, ISIs_high*-1))

fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_high, bins, facecolor='red', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_high*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_high, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

spks_mid = neuronsim.sim_Fv_neurons(20, neurons=2, N=100, t_stop=6, 
                                    FDR=0.3333333333333333)[1]
Fv_mid = Fv_calc(spks_mid)
Rtot = len(np.concatenate(spks_mid))/600

ISIs_mid = []
for trial in spks_mid:
    ISIs_mid.append(np.diff(trial))
ISIs_mid = np.concatenate(ISIs_mid)
ISIs_mid = ISIs_mid*1000
ISIs_mid = np.concatenate((ISIs_mid, ISIs_mid*-1))

fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs_mid, bins, facecolor='green', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv_mid*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs_mid, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %% ISI histogram vs PSTH

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

def preprocess(PSTH, sigma):
    unit = PSTH/LA.norm(PSTH)
    smoothed = gaussian_filter1d(unit, sigma)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    return y_new

true1 = PSTHs[687]
rogue_base = true1 + np.random.normal(0, 1, [100,])
rogue1 = np.roll(rogue_base, 50)
rogue2 = np.roll(rogue_base, 85)
rogue3 = np.roll(rogue_base, 100)

sigma = 5
true1 = preprocess(true1, sigma)
rogue1 = preprocess(rogue1, sigma)
rogue2 = preprocess(rogue2, sigma)
rogue3 = preprocess(rogue3, sigma)

fig, ax = plt.subplots()
plt.plot(true1)
plt.plot(rogue1)

fig, ax = plt.subplots()
plt.plot(true1)
plt.plot(rogue2)

fig, ax = plt.subplots()
plt.plot(true1)
plt.plot(rogue3)

# %%

FDR1 = FDR_master(0.005, (true1 + 0.577*rogue1)*100, rogue1/np.linalg.norm(rogue1), float('inf'))
FDR2 = FDR_master(0.005, (true1 + 0.6*rogue2)*100, rogue2/np.linalg.norm(rogue2), float('inf'))
FDR3 = FDR_master(0.005, (true1 + 0.6*rogue3)*100, rogue3/np.linalg.norm(rogue3), float('inf'))

# %%

# 0.5, 0.2, 0.1 for FDR to get 0.5% ISI viol every time

spikes = neuronsim.sim_spikes_PSTH(true1*80, rogue1*100, N=100, FDR=0.5, 
                                   out_refrac=2.5)

def Fv_calc(trials):
    viols = 0
    spks = 0
    for trial in trials:
        viols += sum(np.diff(trial) < 0.0025)
        spks += len(trial)
    
    return viols/spks

Fv = Fv_calc(spikes)

ISIs= []
for trial in spikes:
    ISIs.append(np.diff(trial))
ISIs = np.concatenate(ISIs)
ISIs = ISIs*1000
ISIs = np.concatenate((ISIs, ISIs*-1))

fig, ax = plt.subplots()
bins = np.arange(-25.5, 25.5, 0.5)
ax.hist(ISIs, bins, facecolor='green', edgecolor='black', zorder=3)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)

ax.set_title('$ISI_{viol} = $' + str(round(Fv*100, 1)) + '%', 
                  fontsize=18)
ax.set_xticks([-20, 20])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

y_max = max(np.histogram(ISIs, bins)[0])
ax.set_yticks([0, y_max])
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax.get_ylim()[1],
                                  color='r', alpha=0.2, zorder=1000, 
                                  fill=True, transform=ax.transData)])

# %%

fig, ax = plt.subplots()

plt.plot(t_new, true1*80, c='blue', lw=2)
plt.plot(t_new, rogue1*80, c='red', lw=2)
plt.plot()

bar = AnchoredSizeBar(ax.transData, 1, '1 s', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)

bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

ax.set_ylim((-0.4168590567958442, 18.507076509749563))

# %% heterogeneous firing diagram

true_neuron = np.roll(PSTHs[387], 45)
rogue_neuron = np.roll(PSTHs[387], 40)



sigma = 5
t = np.linspace(0, 6, 100)
true1 = preprocess(true_neuron, sigma)
rogue1 = preprocess(rogue_neuron, sigma)

t = np.linspace(0, 2, 1000)

fig, ax = plt.subplots()
plt.plot(true1*45, c='blue', lw=2)
plt.plot(rogue1*45, c='red', lw=2)

bar = AnchoredSizeBar(ax.transData, 250, '500 ms', 'lower right', frameon=False, 
                      sep=4)
ax.add_artist(bar)

bar = AnchoredSizeBar(ax.transData, 0, '1 s', 'lower left', frameon=False, 
                      sep=4, size_vertical=2)
ax.add_artist(bar)

# %%

# mat_contents = sio.loadmat('hidehiko_PSTHs')
# PSTHs = mat_contents['PSTHs']

# main_idx = 
# PSTH_idx = list(range(len(PSTHs)))
# other_idx = np.delete(PSTH_idx, main_idx)

# Fv = []
# tot_PSTHs = []
# for i in other_idx:
#     temp_Fv, temp_PSTH = neuronsim.sim_Fv_PSTH2(PSTHs[main_idx], PSTHs[i], FDR=0.2)
#     Fv.append(temp_Fv)
#     tot_PSTHs.append(temp_PSTH)
#     print(i)

# FDR_hetero = []
# for i, j in enumerate(other_idx):
#     FDR_hetero.append(FDR_master(Fv[i], tot_PSTHs[i], 
#                                  PSTHs[j]/np.linalg.norm(PSTHs[j]), 1))

# %%

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 100)
t_new = np.linspace(0, 6, 1000)

PSTHs_new = []
for i in range(len(PSTHs)):
    smoothed = gaussian_filter1d(PSTHs[i], 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    PSTHs_new.append(y_new)
    
PSTHs = np.vstack(PSTHs_new)



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

pred_FDR = []
Rtots = []
Rin_out_dot = []
Rtot_out_dot = []
Fvs = []
R_out_mags = []
R_out_units = []
pred_FDR_true = []

# ideal idx is 693
main_idx = 693
other_idx = list(range(len(PSTHs)))
del other_idx[main_idx]

other_idx = other_idx[:100]

idx1 = main_idx
Rin = PSTHs[idx1]

scale = minimize_scalar(Rout_scale_ob, args=[Rin, 10], 
                        method='bounded', bounds=[0, 100]).x
Rin = Rin*scale

for second_idx in other_idx:
    print(second_idx)

    idx2 = second_idx
    
    FDR = 0.2
    
    scale = minimize_scalar(Rout_scale_ob, 
                            args=[PSTHs[idx2], (FDR/(1-FDR))*np.average(Rin)],
                            method='bounded', bounds=[0, 100]).x
    
    Rout = scale*PSTHs[idx2]
    
    Fv_temp = neuronsim.sim_Fv_PSTH3(Rin, Rout, FDR=FDR, out_refrac=2.5, 
                                     N=10000)
    
    R_out_mags.append(vector_mag(Rout))
    Rtot =  Rin + Rout
    R_out_units.append(Rout/vector_mag(Rout))
    
    Rin_out_dot.append(np.dot(Rin, Rout))
    Rtot_out_dot.append(np.dot(Rtot, Rout))
    
    Fv = Fv_temp
    Fvs.append(Fv)
    D = np.dot(Rtot/vector_mag(Rtot), Rout/vector_mag(Rout))
    
    pred_FDR.append(FDR_master(Fv, [np.average(Rtot)]*1000, 
                               ([np.mean(Rout)]*1000)/vector_mag([np.mean(Rout)]*1000), 
                               1))
    
    pred_FDR_true.append(FDR_master(Fv, Rtot, Rout/vector_mag(Rout), 1))
    
    
# %%
import JV_utils

pred_FDR_true = np.array(pred_FDR_true)

fig, ax = plt.subplots()

Rout = []
for i in range(len(R_out_mags)):
    
    Rout.append(R_out_mags[i]*R_out_units[i])

center = np.average(Rin)*np.average(Rout[0])
plt.scatter((np.array(Rin_out_dot)/1000 - center)/center, pred_FDR, s=20, c='g')

x = np.array(Rin_out_dot)[np.invert(np.isnan(pred_FDR))]
y = np.array(pred_FDR)[np.invert(np.isnan(pred_FDR))]

y_pred, reg, R2 = JV_utils.lin_reg(x, y)
temp = (x/1000 - center)/center
xs = (x/1000 - center)/center
ys = y_pred
ys = JV_utils.sort_list_by_list(xs, ys)
xs = sorted(xs)
xs = [xs[0], xs[-1]]
ys = [ys[0], ys[-1]]

plt.plot(xs, ys, c='g', ls='dotted')
ax.axvline(0, c='k', ls='--', alpha=0.3)
ax.axhline(0.2, c='k', ls='--', alpha=0.3)
plt.ylabel('Predicted FDR', fontsize=16)
plt.xlabel(r'$\overline{R_{in}R_{out}}$ (ẟ)', 
           fontsize=16)
plt.scatter((x/1000 - center)/center, pred_FDR_true[~np.isnan(pred_FDR)], 
            s=10, marker='x', zorder=0, c='b')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.text(0.6, 0.12,'$R^{}$ = {}'.format(2, str(round(R2, 2))), fontsize=16)
# plt.title('$Unit_{idx}$ = 60', fontsize=18)

import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()

# %%

def closest_value(input_list, input_value):

  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin()

  return i, arr[i]

overlap_factor = (np.array(Rin_out_dot)/1000 - center)/center

print(closest_value(overlap_factor, 0))

# idxs to use: 70, 9, 17
idx1 = 693
idx2 = 70

fig, ax = plt.subplots()
plt.plot(PSTHs[idx1]/vector_mag(PSTHs[idx1]))
plt.plot(PSTHs[idx2]/vector_mag(PSTHs[idx2]))



# %%

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

k = 100

N_con = [1, 2, 5, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
N_con[N_con == 10] = float('inf')

Rtots = [4, 8, 12, 20]
Rtots = choices(Rtots, k=k)

FDRs = []
for _ in range(k):
    FDRs.append(uniform(0, 0.5))
    
PSTH_idx = list(range(len(PSTHs)))
idx_pairs = []
for _ in range(k):
    idx_pairs.append(sample(PSTH_idx, 2))

pred_FDR = []  
OF = []
for i in range(k):
    
    Rin = PSTHs[idx_pairs[i][0]]
    Rout = PSTHs[idx_pairs[i][1]]
    Rin[Rin<0] = 0
    Rout[Rout<0] = 0
    
    Rout_target = FDRs[i]*Rtots[i]
    Rin_target = Rtots[i] - Rout_target
    
    scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                        method='bounded', bounds=[0, 100]).x
    Rin = Rin*scale

    scale = minimize_scalar(Rout_scale_ob, 
                        args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                        method='bounded', bounds=[0, 100]).x
    
    Rout = scale*Rout
    
    Rtot = Rin + Rout
    
    center = np.average(Rin)*np.average(Rout[0])
    OF.append((np.dot(Rin, Rout)/1000 - center)/center)
    
    Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                neurons=N_con[i], N=10000)
    
    
    pred_FDR.append(FDR_master(Fv, Rtot, Rout/vector_mag(Rout), N_con[i]))
    
# %%


pred_FDR = np.array(pred_FDR)
FDRs = np.array(FDRs)

idxs = np.array(N_con) == 1
fig, ax = plt.subplots()
plt.scatter(pred_FDR, FDRs, c='blue', s=24)
plt.plot([0, 0.5], [0, 0.5], ls='dashed', c='k', lw=2)
plt.xlabel('Predicted FDR', fontsize=16)
plt.ylabel('True FDR', fontsize=16)
plt.text(0.3, 0.1, '$R^2$ = 0.98', fontsize=16)

y_pred, reg, R2 = JV_utils.lin_reg(pred_FDR, FDRs)

x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

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

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 100)
t_new = np.linspace(0, 6, 1000)

PSTHs_new = []
for i in range(len(PSTHs)):
    smoothed = gaussian_filter1d(PSTHs[i], 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    PSTHs_new.append(y_new)
    
PSTHs = np.vstack(PSTHs_new)

from random import sample

# ideal idx is 693
main_idx = 693
other_idx = list(range(len(PSTHs)))
del other_idx[main_idx]

N_con = 1, 2, 5

idxs_1 = other_idx
idxs_2 = []
idxs_5 = []

for _ in range(len(idxs_1)):
    idxs_2.append(sample(other_idx, 2))
    idxs_5.append(sample(other_idx, 5))
    
ISI_viol = 0.001

FDRs_1 = []
for idx in idxs_1:
    FDRs_1.append(FDR_master(ISI_viol, PSTHs[main_idx], 
                             PSTHs[idx]/vector_mag(PSTHs[idx]), 1))
    
FDRs_2 = []
for idxs in idxs_2:
    total_PSTH = PSTHs[idxs[0]]
    iteridx = iter(idxs)
    next(iteridx)
    for idx in iteridx:
        total_PSTH = total_PSTH + PSTHs[idx]
    
    Rout_unit = total_PSTH/vector_mag(total_PSTH)
    FDRs_2.append(FDR_master(ISI_viol, PSTHs[main_idx], Rout_unit, 2))
    
    
FDRs_5 = []
for idxs in idxs_5:
    total_PSTH = PSTHs[idxs[0]]
    iteridx = iter(idxs)
    next(iteridx)
    for idx in iteridx:
        total_PSTH = total_PSTH + PSTHs[idx]
    
    Rout_unit = total_PSTH/vector_mag(total_PSTH)
    FDRs_5.append(FDR_master(ISI_viol, PSTHs[main_idx], Rout_unit, 5))
    
nans1 = sum(np.isnan(FDRs_1))
diff2 = nans1 - sum(np.isnan(FDRs_2))



bins = np.arange(0, 0.15, 0.003)
hist1 = np.histogram(FDRs_1, bins=bins)
hist2 = np.histogram(FDRs_2, bins=bins)
hist5 = np.histogram(FDRs_5, bins=bins)

sigma = 3
smoothed = gaussian_filter1d(hist1[0], sigma)
plt.plot(smoothed)

smoothed = gaussian_filter1d(hist2[0], sigma)
plt.plot(smoothed)

smoothed = gaussian_filter1d(hist5[0], sigma)
plt.plot(smoothed)



fig, ax = plt.subplots()
plt.hist(FDRs_1, bins=20, edgecolor='black')
plt.hist(FDRs_2, bins=20, edgecolor='black')
plt.hist(FDRs_5, bins=20, edgecolor='black')

mean1 = np.nanmean(FDRs_1)
mean2 = np.nanmean(FDRs_2)
mean5 = np.nanmean(FDRs_5)

total_PSTH = PSTHs[0]
iteridx = iter(other_idx)
next(iteridx)
for idx in iteridx:

    total_PSTH = total_PSTH + PSTHs[idx]
    
Rout_unit = total_PSTH/vector_mag(total_PSTH)
meaninf = FDR_master(ISI_viol, PSTHs[main_idx], Rout_unit, float('inf'))

plt.vlines([mean1, mean2, mean5, meaninf], 0, 200)

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%


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

@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025):

    
    a = -2*tviol*Rtot
    b = 2*tviol*Rtot
    c = -F_v
    
    FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

Fvs = []

for _ in range(1000):
    Fvs.append(neuronsim.sim_Fv(1.9, 0.1, N=1, t_stop=600, out_refrac=2.5)[0])
    
true_Fv = neuronsim.sim_Fv(1.9, 0.1, N=100, t_stop=10000, out_refrac=2.5)[0]

FDRs = kleinfeld_eq(2, Fvs)
    
    
# %%

bins = np.arange(0, 0.009, 0.0001)
Fvs_hist = np.histogram(Fvs, bins=bins)
sigma = 2
smoothed = gaussian_filter1d(Fvs_hist[0], sigma)
plt.plot(smoothed)

# %%

from scipy import stats
k2, p = stats.shapiro(np.array(Fvs)[:50]*100)

fig, ax = plt.subplots()
plt.hist(np.array(Fvs)*100, bins=20, edgecolor='k')
plt.xlabel('% $ISI_{viol}$', fontsize=16)
plt.ylabel('Count', fontsize=16)
# plt.title('$R_{in}$ = 0.4 Hz, $R_{out}$ = 1.6 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.vlines(true_Fv*100, 0, 175, color='k', ls='--')
fig.patches.extend([plt.Rectangle((0.25, 0), ax.get_xlim()[1]-0.25, ax.get_ylim()[1],
                                          color='r', alpha=0.2, zorder=1000, 
                                          fill=True, transform=ax.transData)])
plt.tight_layout()


fig, ax = plt.subplots()
FDRs_plot = FDRs
FDRs_plot[np.isnan(FDRs_plot)] = 0.5
plt.hist(FDRs_plot, bins=20, edgecolor='k')
plt.xlabel('Predicted FDR', fontsize=16)
plt.ylabel('Count', fontsize=16)
# plt.title('$R_{in}$ = 0.4 Hz, $R_{out}$ = 1.6 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()

plt.tight_layout() 

fig, ax = plt.subplots()
plt.boxplot(np.array(Fvs)*100)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
plt.title('$R_{in}$ = 0.4 Hz, $R_{out}$ = 1.6 Hz', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout() 



    
    
    
    
    
    
    
    
    