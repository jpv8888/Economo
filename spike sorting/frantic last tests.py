# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:37:09 2023

@author: jpv88
"""

import JV_utils
import neuronsim
import scipy.io as sio
import numpy as np

path = r'C:\\Users\\jpv88\\OneDrive\\Documents\\GitHub\\Economo\\spike sorting\\Real Data FDR Predictions\\Inagaki et al 2019\\'

    
mat_contents = sio.loadmat(path + 'Inagaki2019_PSTHs.mat')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat(path + 'Inagaki2019_ISI_viol.mat')
ISI_viol = mat_contents['ISI_viol']

# %%

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


PSTH1 = PSTHs[4]
PSTH2 = PSTHs[70]
neurons = float('inf')

Fv = neuronsim.sim_Fv_PSTH4(PSTH1, PSTH2, neurons=neurons, N=10000)

pred_FDR = FDR_master(Fv, PSTH1 + PSTH2, 
                      PSTH2/np.linalg.norm(PSTH2), neurons)

true_FDR = np.mean(PSTH2)/(np.mean(PSTH1) + np.mean(PSTH2))




    