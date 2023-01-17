# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:16:32 2023

@author: jpv88
"""

import neuronsim

import numpy as np

@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025, t_c=0):

    a = -2*(tviol-t_c)*Rtot
    b = 2*(tviol-t_c)*Rtot
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

t_stop = 100000
spikes, ids = neuronsim.sim_spikes(16, 4, N=1, t_stop=t_stop)
spikes = spikes[0]

old_spikes = spikes.copy()

censor = 0.001

spikes_c = []
for n in spikes:
    if not spikes_c or abs(n - spikes_c[-1]) >= censor:
        spikes_c.append(n)

ISI_viol_c = sum(np.diff(spikes_c) < 0.0025)/len(spikes_c)
ISI_viol = sum(np.diff(old_spikes) < 0.0025)/len(old_spikes)
Rtot = len(spikes)/t_stop
Rtot_c = len(spikes_c)/t_stop

FDR_no_correct = kleinfeld_eq(Rtot_c, ISI_viol_c)
FDR_correct = kleinfeld_eq(Rtot_c, ISI_viol_c, t_c=0.001)



