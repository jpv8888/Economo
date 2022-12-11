# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 20:25:29 2022

@author: jpv88
"""

import numpy as np
import neuronsim
import matplotlib.pyplot as plt


# %% does recording time affect it, yes
n = 1000
t_stops = np.arange(60, 600, 120)
Fvs = np.zeros((len(t_stops), n))
Rin = 10
Rout = 10

for i, t_stop in enumerate(t_stops):
    for j in range(n):
        Fvs[i,j] = neuronsim.sim_Fv(Rin, Rout, t_stop=t_stop, N=1)[0]
    
true_Fv = neuronsim.sim_Fv(Rin, Rout, t_stop=86400, N=10)[0]
CVs = np.std(Fvs, axis=1)/true_Fv

# %% does total firing rate affect it, no?

n = 1000
Rtots = np.arange(2, 20, 5)
Fvs = np.zeros((len(Rtots), n))

for i, Rtot in enumerate(Rtots):
    for j in range(n):
        Rin = Rtot/2
        Rout = Rtot/2
        Fvs[i,j] = neuronsim.sim_Fv(Rin, Rout, t_stop=600, N=1)[0]
    
CVs = np.std(Fvs, axis=1)/true_Fv

# %% does FDR affect it, yes

n = 1000
FDRs = np.arange(0.1, 1, 0.1)
Rtot = 10
Fvs = np.zeros((len(FDRs), n))

for i, FDR in enumerate(FDRs):
    for j in range(n):
        Rout = FDR*Rtot
        Rin = Rtot - Rout
        Fvs[i,j] = neuronsim.sim_Fv(Rin, Rout, t_stop=600, N=1)[0]
    
CVs = np.std(Fvs, axis=1)/true_Fv

# %% does tau affect it, yes

n = 1000
refracs = np.arange(1, 3, 0.5)
Rin = 5
Rout = 5
Fvs = np.zeros((len(refracs), n))

for i, refrac in enumerate(refracs):
    for j in range(n):
        Fvs[i,j] = neuronsim.sim_Fv(Rin, Rout, t_stop=600, N=1, 
                                    refractory_period=refrac)[0]
    
CVs = np.std(Fvs, axis=1)/true_Fv

# %% does the number of confound neurons affect it, yes?

n = 1000
neurons = np.arange(1, 20, 3)
Rtot = 10
Fvs = np.zeros((len(neurons), n))

for i, neuron in enumerate(neurons):
    for j in range(n):
        Fvs[i,j] = neuronsim.sim_Fv_neurons(Rtot, t_stop=600, N=1, 
                                            neurons=neuron, FDR=0.5)
    
CVs = np.std(Fvs, axis=1)/true_Fv




# %%

fig, ax = plt.subplots()
plt.hist(Fvs[0,:], bins=20)
ylims = ax.get_ylim()
plt.vlines(true_Fv, ylims[0], ylims[1])

# %%

CVs = np.std(Fvs, axis=1)/true_Fv
plt.plot(t_stops, CVs)

# %%

from scipy.optimize import curve_fit

def func(x, a, b):
    return (a*(x**-2) + b)

popt, pcov = curve_fit(func, t_stops, CVs)
print(popt)

test = popt[0]*(1/(t_stops**2)) + popt[1]
plt.plot(t_stops, test)
plt.plot(t_stops, CVs)