# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:58:53 2022

@author: jpv88
"""

from math import sqrt
import scipy

import matplotlib.pyplot as plt
import neuronsim
import numpy as np

def hellinger(p,q):
    """Hellinger distance between distributions"""
    return sum([(sqrt(t[0])-sqrt(t[1]))*(sqrt(t[0])-sqrt(t[1]))\
                for t in zip(p,q)])/sqrt(2.)

# %%

refrac = 0
fr = 10
singles = []
hists = []
singles.append(neuronsim.sim_effective_refrac(fr, refractory_period=refrac))
for neurons in range(1, 10):
    hists.append(neuronsim.sim_effective_refrac(fr/neurons, neurons=neurons,
                                                refractory_period=2.5))
        
# %%


hists = []
neurons = 10

hists.append(neuronsim.sim_effective_refrac(20/neurons, neurons=neurons, 
                                            refractory_period=2.5))
# %%
singles = []
refracs = np.arange(0, 2.51, 0.01)
for i in np.arange(0, 2.51, 0.01):
    print(i)
    singles.append(neuronsim.sim_effective_refrac(20, refractory_period=i))

# %%
dists = [hellinger(hists[0], el) for el in singles]
plt.plot(dists)

val, idx = min((val, idx) for (idx, val) in enumerate(dists))
print(refracs[idx])

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

