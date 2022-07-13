# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:14:55 2022

@author: jpv88
"""
import numpy as np

import neuronsim

F_v = []
FDR = []
Rtot = []

for i in range(20,21):
    for j in np.linspace(50, 200, 5):
        F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv_PSTH(norm1, norm2) 
        F_v.append(F_v_temp)
        FDR.append(FDR_temp) 
        Rtot.append(Rtot_temp)