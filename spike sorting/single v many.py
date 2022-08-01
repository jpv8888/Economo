# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:58:53 2022

@author: jpv88
"""

from scipy.stats import entropy

from math import sqrt
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import neuronsim
import numpy as np

def hellinger(p,q):
    """Hellinger distance between distributions"""
    return sum([(sqrt(t[0])-sqrt(t[1]))*(sqrt(t[0])-sqrt(t[1]))\
                for t in zip(p,q)])/sqrt(2.)
        

import random

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

# %%

refrac = 0
fr = 10
singles = []
hists = []
singles.append(neuronsim.sim_effective_refrac(fr, refractory_period=refrac))
for neurons in range(1, 10):
    hists.append(neuronsim.sim_effective_refrac(fr/neurons, neurons=neurons,
                                                refractory_period=2.5))

singles = [el/sum(el) for el in singles]
hists = [el/sum(el) for el in hists]

# %%

Rtots = [1] + list(range(5, 35, 5))
tau_effs = []

for Rtot in Rtots:
    
    singles = []
    refracs = np.arange(0, 2.51, 0.01)
    for i in np.arange(0, 2.51, 0.01):
        print(i)
        singles.append(neuronsim.sim_effective_refrac(Rtot, refractory_period=i))
    
    singles = [el/sum(el) for el in singles]
        
    hists = []
    for neurons in range(1, 11):
        hists.append(neuronsim.sim_effective_refrac(Rtot/neurons, neurons=neurons, 
                                                    refractory_period=2.5))
    
    hists = [el/sum(el) for el in hists]
    
    tau_eff = []
    for hist in hists:
        dists = [hellinger(hist, el) for el in singles]
        val, idx = min((val, idx) for (idx, val) in enumerate(dists))
        tau_eff.append(refracs[idx])
        
    tau_effs.append(tau_eff)
    
# %%

taus = np.array(tau_effs)
means = np.mean(taus, axis=0)
stds = np.std(taus, axis=0)

means = np.array([el/2.5 for el in means])
stds = np.array([el/2.5 for el in stds])
neurons = list(range(1, 11))

fig, ax = plt.subplots()
ax.plot(neurons, means)
ax.fill_between(neurons, (means-stds), (means+stds), color='b', alpha=.1)

plt.ylim(0, 1)
plt.xlim(1, 10)
plt.xlabel('# Neurons', fontsize=14)
plt.ylabel('Effective τ', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fit = [2.5/el for el in neurons]
fit = np.array([el/2.5 for el in fit])

ax.plot(neurons, fit)


R2 = str(round(r2_score(means, fit), 2))

plt.text(7, 0.82, '$R^{2}$ = ' + R2, fontsize=14)
plt.text(7, 0.75, '$τ_{eff}$ =  τ/N', fontsize=14)
plt.title("Equal Distribution of $R_{tot}$", fontsize=18)
plt.tight_layout()

# %%

Rtots = list(range(15, 30, 5))
tau_effs = []
rates_list = []

for Rtot in Rtots:
    
    singles = []
    refracs = np.arange(0, 2.51, 0.01)
    for i in np.arange(0, 2.51, 0.01):
        print(i)
        singles.append(neuronsim.sim_effective_refrac(Rtot, 
                                                      refractory_period=i))
    
    singles = [el/sum(el) for el in singles]
        
    hists = []
    
    for k in range(0, 5):
        for j, neurons in enumerate(range(1, 11)):
            rates = constrained_sum_sample_pos(neurons, Rtot)
            rates_list.append(rates)
            hists.append(neuronsim.sim_effective_refrac(rates, 
                                                        neurons=neurons, 
                                                        refractory_period=2.5))
    
    hists = [el/sum(el) for el in hists]
    
    tau_eff = []
    for hist in hists:
        dists = [hellinger(hist, el) for el in singles]
        val, idx = min((val, idx) for (idx, val) in enumerate(dists))
        tau_eff.append(refracs[idx])
        
    tau_effs.append(tau_eff)
    
# %%

entropies = [entropy(el)/entropy(np.ones(len(el))) for el in rates_list]

neurons = []
for i in range(3):
    for k in range(0, 5):
        neurons.extend(range(1, 11))
    
tau_fit = []
for i, n in enumerate(neurons):
    tau_fit.append(2.5/(n*entropies[i]))

# %%

        
dists = [hellinger(singles[0], el) for el in hists]

plt.plot(range(1, 10), dists)

# %%

neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tau1 = [2.5, 1.28, 0.88, 0.63, 0.52, 0.43, 0.41, 0.33, 0.33, 0.18]
tau2 = [2.49, 1.25, 0.84, 0.63, 0.4, 0.38, 0.58, 0.25, 0.19, 0.29]
tau3 = [2.5, 1.24, 0.86, 0.64, 0.51, 0.41, 0.33, 0.32, 0.3, 0.32]

taus = np.array([tau1, tau2, tau3])
means = np.mean(taus, axis=0)
stds = np.std(taus, axis=0)

fig, ax = plt.subplots()
ax.plot(neurons, means)
ax.fill_between(neurons, (means-stds), (means+stds), color='b', alpha=.1)
plt.ylim(0, 2.5)
plt.xlim(1, 10)
plt.xlabel('# Neurons', fontsize=14)
plt.ylabel('Effective τ', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%


    



