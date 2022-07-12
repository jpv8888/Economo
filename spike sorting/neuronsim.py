# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:48:46 2022

@author: jpv88
"""

from elephant.spike_train_generation import NonStationaryPoissonProcess
from elephant.spike_train_generation import StationaryPoissonProcess
from matplotlib import cm
from neo.core import AnalogSignal
from random import sample
from scipy import stats
from tqdm import tqdm

import itertools
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import JV_utils

SMALL_SIZE = 9
MEDIUM_SIZE = 13
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %%

def sim_Fv(rate_in, rate_out, t_stop=1000, refractory_period=2.5, N=100, 
           out_refrac=0, FDR=None):
    
    
    FDR_max = rate_out/(rate_in + rate_out)
    # if FDR wasn't specified, set it to its maximum possible value
    if FDR == None:
        FDR = FDR_max
    if FDR > FDR_max:
        raise Exception('Input FDR is not possible')
        
    fp = round((FDR*rate_in)/(1 - FDR))*t_stop
    
    F_v = np.zeros(N)
    FDR_rec = np.zeros(N)
    Rtot = np.zeros(N)
    
    rate_in = pq.Quantity(rate_in, 'Hz')
    rate_out = pq.Quantity(rate_out, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    refractory_period = pq.Quantity(refractory_period, 'ms')
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
        
    clu_in = StationaryPoissonProcess(rate=rate_in, t_stop=t_stop, 
                                      refractory_period=refractory_period)
    
    clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop)
    if out_refrac != 0:
        clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                           refractory_period=out_refrac)
    
    if FDR == FDR_max:
        desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
        for i in tqdm(range(N), desc=desc):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            spks_tot = np.concatenate((spks_in, spks_out))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR_rec[i] = len(spks_out)/len(spks_tot)
            Rtot[i] = len(spks_tot)/t_stop
        
        return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot)
        
       
    elif FDR != FDR_max:
        desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
        for i in tqdm(range(N), desc=desc):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            
            spks_out_sample = sample(list(spks_out), fp)
            spks_out_sample = np.array(spks_out_sample)
            
            spks_tot = np.concatenate((spks_in, spks_out_sample))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR_rec[i] = len(spks_out_sample)/len(spks_tot)
            Rtot[i] = len(spks_tot)/t_stop
            
        return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot)
        
    

# simulate two neurons with PSTHs being intermixed
def sim_Fv_PSTH(PSTH_in, PSTH_out, T=6, refractory_period=2.5, N=1000, 
                out_refrac=0, FPR=1):
    
    F_v = np.zeros(N)
    FDR = np.zeros(N)
    Rtot = np.zeros(N)
    
    refractory_period = pq.Quantity(refractory_period, 'ms')
    
    n = len(PSTH_in)
    f = n/T
    
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
    
    sig_in = AnalogSignal(PSTH_in, units='Hz', sampling_rate=f*pq.Hz)
    sig_out = AnalogSignal(PSTH_out, units='Hz', sampling_rate=f*pq.Hz)
    
    clu_in = NonStationaryPoissonProcess(rate_signal=sig_in,
                                         refractory_period=refractory_period)
    
    clu_out = NonStationaryPoissonProcess(rate_signal=sig_out)
    if out_refrac != 0:
        clu_out = NonStationaryPoissonProcess(rate_signal=sig_out,
                                              refractory_period=out_refrac)
    if FPR == 1:
        
        for i in tqdm(range(N)):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            spks_tot = np.concatenate((spks_in, spks_out))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR[i] = len(spks_out)/len(spks_tot)
            Rtot[i] = len(spks_tot)/T
            
        return np.mean(F_v), np.mean(FDR), np.mean(Rtot)
    
    if FPR != 1:
        for i in tqdm(range(N)):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            samples = round(FPR*len(spks_out))
            spks_out_sample = sample(list(spks_out), samples)
            spks_out_sample = np.array(spks_out_sample)
            
            spks_tot = np.concatenate((spks_in, spks_out_sample))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR[i] = len(spks_out_sample)/len(spks_tot)
            Rtot[i] = len(spks_tot)/T
            
        return np.mean(F_v), np.mean(FDR), np.mean(Rtot)
    
def sim_Fv_times(rate_in, rate_out, t_stop=1000, refractory_period=2.5, N=100, 
                 out_refrac=0, FDR=None): 
    
    refrac = refractory_period * 0.001
    t_stop_val = t_stop
    
    FDR_max = rate_out/(rate_in + rate_out)
    
    # if FDR wasn't specified, set it to its maximum possible value
    if FDR == None:
        FDR = FDR_max
    if FDR > FDR_max:
        raise Exception('Input FDR is not possible')
        
    if rate_in != 0:
        fp = round((FDR*rate_in)/(1 - FDR))*t_stop
    
    F_v = np.zeros(N)
    FDR_rec = np.zeros(N)
    Rtot = np.zeros(N)
    bad_time = np.zeros(N)
    
    rate_in = pq.Quantity(rate_in, 'Hz')
    rate_out = pq.Quantity(rate_out, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    refractory_period = pq.Quantity(refractory_period, 'ms')
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
        
    clu_in = StationaryPoissonProcess(rate=rate_in, t_stop=t_stop, 
                                      refractory_period=refractory_period)
    
    clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop)
    if out_refrac != 0:
        clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                           refractory_period=out_refrac)
    
    if FDR == FDR_max:
        desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
        for i in tqdm(range(N), desc=desc):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            spks_tot = np.concatenate((spks_in, spks_out))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR_rec[i] = len(spks_out)/len(spks_tot)
            Rtot[i] = len(spks_tot)/t_stop
            
            arr = []
            for spk in spks_tot:
                if spk-refrac < 0:
                    arr.append([0, spk+refrac])
                elif spk+refrac > t_stop_val:
                    arr.append([spk-refrac, t_stop_val])
                else:
                    arr.append([spk-refrac, spk+refrac])
                
            # Stores index of last element
            # in output array (modified arr[])
            index = 0
          
            # Traverse all input Intervals starting from
            # second interval
            for j in range(1, len(arr)):
          
                # If this is not first Interval and overlaps
                # with the previous one, Merge previous and
                # current Intervals
                if (arr[index][1] >= arr[j][0]):
                    arr[index][1] = max(arr[index][1], arr[j][1])
                else:
                    index = index + 1
                    arr[index] = arr[j]
            
            arr_bad_time = 0
            for interval in arr:
                arr_bad_time += interval[1] - interval[0]
            bad_time[i] = arr_bad_time/t_stop_val
        
        return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot), np.mean(bad_time)
        
       
    elif FDR != FDR_max:
        desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
        for i in tqdm(range(N), desc=desc):
            
            spks_in = clu_in.generate_spiketrain(as_array=True)
            spks_out = clu_out.generate_spiketrain(as_array=True)
            
            
            spks_out_sample = sample(list(spks_out), fp)
            spks_out_sample = np.array(spks_out_sample)
            
            spks_tot = np.concatenate((spks_in, spks_out_sample))
            spks_tot = np.sort(spks_tot)
            
            ISIs = np.diff(spks_tot)
            Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
            F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
            FDR_rec[i] = len(spks_out_sample)/len(spks_tot)
            Rtot[i] = len(spks_tot)/t_stop
            
        return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot)
    
def sim_Fv_overlap(rate_out, t_stop=1000, refractory_period=2.5, N=100, 
                 out_refrac=0, FDR=None): 
    
    refrac = refractory_period * 0.001
    t_stop_val = t_stop
    
    F_v = np.zeros(N)
    FDR_rec = np.zeros(N)
    Rtot = np.zeros(N)
    overlap = np.zeros(N)
    
    rate_out = pq.Quantity(rate_out, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    refractory_period = pq.Quantity(refractory_period, 'ms')
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
    
    clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop)
    if out_refrac != 0:
        clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                           refractory_period=out_refrac)
    
    def getOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    desc = "R_out = " + str(rate_out)
    for i in tqdm(range(N), desc=desc):
    
        spks_out = clu_out.generate_spiketrain(as_array=True)
        
        spks_tot = spks_out
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        FDR_rec[i] = len(spks_out)/len(spks_tot) if len(spks_tot) !=0 else 0
        Rtot[i] = len(spks_tot)/t_stop
        
        arr = []
        for spk in spks_tot:
            if spk-refrac < 0:
                arr.append([0, spk+refrac])
            elif spk+refrac > t_stop_val:
                arr.append([spk-refrac, t_stop_val])
            else:
                arr.append([spk-refrac, spk+refrac])
       
        overlap_trial = 0
        for a, b in itertools.combinations(arr, 2):
            overlap_trial += getOverlap(a, b)
            
        overlap[i] = overlap_trial/t_stop_val
           
    
    return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot), np.mean(overlap)

def sim_Fv_overlap2(rate_in, rate_out, t_stop=1000, refractory_period=2.5, N=100, 
                 out_refrac=0): 
    
    refrac = refractory_period * 0.001
    t_stop_val = t_stop
        
    
    F_v = np.zeros(N)
    FDR_rec = np.zeros(N)
    Rtot = np.zeros(N)
    overlap = np.zeros(N)
    
    rate_in = pq.Quantity(rate_in, 'Hz')
    rate_out = pq.Quantity(rate_out, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    refractory_period = pq.Quantity(refractory_period, 'ms')
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
        
    clu_in = StationaryPoissonProcess(rate=rate_in, t_stop=t_stop, 
                                      refractory_period=refractory_period)
    
    clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop)
    if out_refrac != 0:
        clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                           refractory_period=out_refrac)
        
    def getOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    
    desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
    for i in tqdm(range(N), desc=desc):
        
        spks_in = clu_in.generate_spiketrain(as_array=True)
        spks_out = clu_out.generate_spiketrain(as_array=True)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        FDR_rec[i] = len(spks_out)/len(spks_tot)
        Rtot[i] = len(spks_tot)/t_stop
        
        arr_in = []
        for spk in spks_in:
            if spk-refrac < 0:
                arr_in.append([0, spk+refrac])
            elif spk+refrac > t_stop_val:
                arr_in.append([spk-refrac, t_stop_val])
            else:
                arr_in.append([spk-refrac, spk+refrac])
        
        arr_out = []
        for spk in spks_out:
            if spk-refrac < 0:
                arr_out.append([0, spk+refrac])
            elif spk+refrac > t_stop_val:
                arr_out.append([spk-refrac, t_stop_val])
            else:
                arr_out.append([spk-refrac, spk+refrac])
            
        overlap_trial = 0
        for a, b in itertools.product(arr_in, arr_out):
            overlap_trial += getOverlap(a, b)
            
        overlap[i] = overlap_trial/t_stop_val
    
    return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot), np.mean(overlap)

def sim_Fv_overlap2(rate_in, rate_out, t_stop=1000, refractory_period=2.5, N=100, 
                 out_refrac=0): 
    
    refrac = refractory_period * 0.001
    t_stop_val = t_stop
        
    
    F_v = np.zeros(N)
    FDR_rec = np.zeros(N)
    Rtot = np.zeros(N)
    overlap = np.zeros(N)
    
    rate_in = pq.Quantity(rate_in, 'Hz')
    rate_out = pq.Quantity(rate_out, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    refractory_period = pq.Quantity(refractory_period, 'ms')
    if out_refrac != 0:
        out_refrac = pq.Quantity(out_refrac, 'ms')
        
    clu_in = StationaryPoissonProcess(rate=rate_in, t_stop=t_stop, 
                                      refractory_period=refractory_period)
    
    clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop)
    if out_refrac != 0:
        clu_out = StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                           refractory_period=out_refrac)
        
    def getOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    
    desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
    for i in tqdm(range(N), desc=desc):
        
        spks_in = clu_in.generate_spiketrain(as_array=True)
        spks_out = clu_out.generate_spiketrain(as_array=True)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        FDR_rec[i] = len(spks_out)/len(spks_tot)
        Rtot[i] = len(spks_tot)/t_stop
        
        arr_in = []
        for spk in spks_in:
            if spk-refrac < 0:
                arr_in.append([0, spk+refrac])
            elif spk+refrac > t_stop_val:
                arr_in.append([spk-refrac, t_stop_val])
            else:
                arr_in.append([spk-refrac, spk+refrac])
        
        arr_out = []
        for spk in spks_out:
            if spk-refrac < 0:
                arr_out.append([0, spk+refrac])
            elif spk+refrac > t_stop_val:
                arr_out.append([spk-refrac, t_stop_val])
            else:
                arr_out.append([spk-refrac, spk+refrac])
            
        overlap_trial = 0
        for a, b in itertools.product(arr_in, arr_out):
            overlap_trial += getOverlap(a, b)
            
        overlap[i] = overlap_trial/t_stop_val
    
    return np.mean(F_v), np.mean(FDR_rec), np.mean(Rtot), np.mean(overlap)
    
    
    
    
    
    

# # %%

# # R_in = np.linspace(0.01, 30, num=10)
# # R_out = np.linspace(0, 30, num=10)
# # F_v = np.zeros((len(R_in), len(R_out)))

# # for index, x in np.ndenumerate(F_v):
# #     F_v[index] = sim_Fv(R_in[index[0]], R_out[index[1]], N=10)
    
# # %%

# Rtot = np.linspace(0, 20, num=21)

# R_in = []
# R_out = []

# # how many unique combinations of R_in and R_out per Rtot
# n = 11
# for idx, el in enumerate(Rtot):
#     fractions = np.linspace(0,1,n)
#     R_in_slot = []
#     R_out_slot = []
#     for fraction in fractions:
#         R_in_slot.append(fraction*el)
#     for fraction in reversed(fractions):
#         R_out_slot.append(fraction*el)
#     R_in.append(R_in_slot)
#     R_out.append(R_out_slot)
    
# R_in = [item for sublist in R_in for item in sublist]
# R_out = [item for sublist in R_out for item in sublist]

# Rtot_final = []
# for idx, el in enumerate(Rtot):
#     Rtot_final.append(n*[el])
    
# Rtot = [item for sublist in Rtot_final for item in sublist]

# F_v = np.zeros(len(R_in))
# fp = np.zeros(len(R_in))

# for index, x in enumerate(F_v):
#     F_v[index] = sim_Fv(R_in[index], R_out[index], N=20)
#     fp[index] = R_out[index]/Rtot[index] if Rtot[index] != 0 else 0
    

    

# # Rtot, F_v = np.meshgrid(Rtot, F_v)
# # fp = np.zeros(Rtot.shape)
# # for index, x in np.ndenumerate(F_v):
# #     fp[index] = R_out[index[1]]/Rtot[index] if Rtot[index] != 0 else 0
    




# # %%

# # Rtot = np.zeros(F_v.shape)
# # fp = np.zeros(F_v.shape)
# # for index, x in np.ndenumerate(F_v):
# #     Rtot[index] = R_in[index[0]] + R_out[index[1]]
# #     fp[index] = R_out[index[1]]/Rtot[index]

# # %%
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = Rtot
# Y = F_v
# Z = fp

# # Plot the surface.
# surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
# # ax.set_zlim(0, 1)
# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')
# ax.set_zlabel('fp')


# plt.show()
# plt.title('fp vs. Rtot and F_v surface plot (rotated 1)', fontsize=14)
# ax.locator_params(nbins=5, axis='x')

# ax.view_init(azim=90, elev=0)

# # %%
# fig, ax = plt.subplots()

# # Make data.
# X = Rtot
# Y = F_v
# Z = fp

# # Plot the surface.
# surf = ax.tricontourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False,levels=n)

# m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
# m.set_array(Z)
# m.set_clim(0, 1)
# cbar = plt.colorbar(m)
# cbar.set_label('fp', rotation=270)
# # ax.set_zlim(0, 1)
# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')

# plt.title('fp vs. Rtot and F_v (filled contour)', fontsize=14)

# plt.show()

# # %%

# def plot_slices(Rtot, F_v, fp, sparsity=1):
#     uniq = list(set(Rtot))
#     uniq = uniq[::int(1/sparsity)]
#     for val in uniq:
#         idxs = [i for i, j in enumerate(Rtot) if j == val]
#         fig, ax = plt.subplots()
#         plt.plot(F_v[idxs], fp[idxs],lw=2)
#         plt.xlabel('F_v')
#         plt.ylabel('fp')
#         plt.title('R_tot = ' + str(round(val, 2)) + ' Hz', fontsize=14)
#         tviol = 0.0025
#         Rviol = val*F_v[idxs]
        
#         a = -1/2
#         b = val
#         c = -Rviol/(2*tviol)
        
#         predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
#         fp_test = predRout/val
        
#         plt.plot(F_v[idxs], fp_test,lw=2)
        
#         a = 1
#         b = -1
#         c = F_v[idxs]/(2*tviol*val)
#         predfp = (-b - (b**2 - 4*a*c)**(1/2))/(2*a)
        
        
#         plt.plot(F_v[idxs], predfp,lw=2)
#         if not np.isnan(fp_test).any():
#             e_label = 'Economo Equation, R = ' + str(round(stats.pearsonr(fp[idxs], fp_test)[0], 3))
#         else:
#             e_label = 'Economo Equation'
#         if not np.isnan(predfp).any():
#             k_label = 'Kleinfeld Equation, R = ' + str(round(stats.pearsonr(fp[idxs], predfp)[0], 3))
#         else:
#             k_label = 'Kleinfeld Equation'
#         plt.legend(['Simulation', e_label, k_label],fontsize=11)
        
# plot_slices(Rtot, F_v, fp)
# # %%

# Rs = []
# Rtots = []
# uniq = list(set(Rtot))
# for val in uniq:
#     idxs = [i for i, j in enumerate(Rtot) if j == val]
#     tviol = 0.0025
#     Rviol = val*F_v[idxs]
    
#     a = -1/2
#     b = val
#     c = -Rviol/(2*tviol)
    
#     predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
#     fp_test = predRout/val
    
#     if not np.isnan(fp_test).any():
#         Rs.append(stats.pearsonr(fp[idxs], fp_test)[0])
#         Rtots.append(val)
        
# plt.plot(Rtots,Rs,lw=2)

# plt.xlabel('Rtot (Hz)')
# plt.ylabel("Pearson's r")
# plt.title('Economo equation correlation with simulation', fontsize=14)

    
    
    



# # %%

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = Rtot
# Y = F_v
# Z = fp

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')
# ax.set_zlabel('fp')


# plt.show()

# # %% slice off right side

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Rtot_cut = np.copy(Rtot)
# fp_cut = np.copy(fp)
# F_v_cut = np.copy(F_v)

# Rtot_cut = Rtot_cut[Rtot > 150]
# fp_cut = fp_cut[Rtot > 150]
# F_v_cut = F_v_cut[Rtot > 150]


# # for index, x in np.ndenumerate(Rtot):
# #     if x > 150:
# #         Rtot_cut = np.delete(Rtot_cut, index)
# #         fp_cut = np.delete(fp_cut, index)
# #         F_v_cut = np.delete(F_v_cut, index)
    
# # Make data.
# X = Rtot_cut
# Y = F_v_cut
# Z = fp_cut

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')
# ax.set_zlabel('fp')

# plt.show()


# # %% slicerino 

# F_v_test = 0.005

# good_idx = []
# for index, x in np.ndenumerate(F_v):
#     if abs(x-F_v_test) < 0.003:
#         good_idx.append(index)
        
# Rtot_slice = []
# fp_slice = []
# for idx in good_idx:
#     Rtot_slice.append(Rtot[idx])
#     fp_slice.append(fp[idx])
    
# plt.plot(Rtot_slice, fp_slice)
# plt.xlabel('Rtot (Hz)')
# plt.ylabel('fp')
# plt.title('F_v = ' + str(F_v_test))
# plt.ylim(0, 1)

# Rtot_test = np.linspace(0, 50)
# tviol = 0.0025
# Rviol = Rtot_test*F_v_test

# a = -1/2
# b = Rtot_test
# c = -Rviol/(2*tviol)

# predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
# fp_test = predRout/Rtot_test

# plt.plot(Rtot_test, fp_test)

# a = 1
# b = -1
# c = F_v_test/(2*tviol*Rtot_test)
# predfp = (-b - (b**2 - 4*a*c)**(1/2))/(2*a)

# plt.plot(Rtot_test, predfp)
# plt.legend(['Simulation', 'Economo Equation', 'Hill Equation'])


# # %%
# good_idx = []
# for index, x in np.ndenumerate(F_v):
#     if abs(x-0.1) < 0.005:
#         good_idx.append(index)
        
# Rtot_slice = []
# fp_slice = []
# for idx in good_idx:
#     Rtot_slice.append(Rtot[idx])
#     fp_slice.append(fp[idx])
    
# plt.plot(Rtot_slice, fp_slice)
# plt.xlabel('Rtot (Hz)')
# plt.ylabel('fp')
# plt.title('F_v = 0.1')
# plt.ylim(0, 1)
        
# # %%
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Make data.
# X = Rtot
# Y = F_v
# Z = fp

# ax.scatter(X, Y, Z)

# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')
# ax.set_zlabel('fp')

# plt.show()

# # %%

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = Rtot
# Y = F_v
# Z = fp



# ax.set_xlabel('Rtot')
# ax.set_ylabel('F_v')
# ax.set_zlabel('fp')

# plt.show()

# xx = np.linspace(0, 300, num=30)

# zz = np.linspace(0, 1, num=30)

# xx, zz = np.meshgrid(xx, zz, indexing='ij')
# yy = (xx*0.0025)/2

# ax.plot_surface(xx, yy, zz, linewidth=0, alpha=0.5, antialiased=False)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# ax.view_init(azim=-90, elev=90)
