# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:47:11 2022

@author: jpv88
"""

import neuronsim

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from random import sample
from scipy.stats import levy, cauchy, entropy, norm, rv_histogram
from skopt import gp_minimize

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
mat_contents = sio.loadmat('hidehiko_ISIviol')
ISI_viol = mat_contents['ISI_viol']

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

def plot_cauchy(loc, scale):
    FDRs = []
    done = 0
    while done == 0:
        FDRs_temp = cauchy.rvs(loc=loc, scale=scale, size=1000)
        FDRs.extend(FDRs_temp[(FDRs_temp >= 0) & (FDRs_temp <= 1)])
        if len(FDRs) >= 10000:
            done = 1
            
    plt.hist(FDRs, bins=20)
    plt.xlabel('True FDR')
    plt.ylabel('Density (A.U.)')
    plt.title('Hidehiko Session 1 fitted Cauchy to predicted FDRs', 
              fontsize=18)
    print(np.mean(FDRs))
    
        
# match predicted FDR distributions
def f_eq(x):
    print(x)
    mean, std = x
    N = len(PSTHs)
    
    FDRs = []
    done = 0
    while done == 0:
        FDRs_temp = cauchy.rvs(loc=mean, scale=std, size=N)
        FDRs.extend(FDRs_temp[(FDRs_temp >= 0) & (FDRs_temp <= 1)])
        if len(FDRs) >= N:
            done = 1
        
    FDRs = sample(FDRs, N)
    
    Fvs = []
    for i in range(N):
        PSTH = PSTHs[i,:]
        PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
        Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=386, FDR=FDRs[i], T=3))
        print(i)
    
    pred_FDRs = []
    sim_FDRs = []
    
    for i in range(N):
        PSTH = PSTHs[i,:]
        PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
        out_unit = PSTH_out/np.linalg.norm(PSTH_out)
        pred_FDRs.append(FDR_master(ISI_viol[i], PSTH, out_unit, N=float('inf')))
        sim_FDRs.append(FDR_master(Fvs[i], PSTH, out_unit, N=float('inf')))
        
    pred_FDRs = np.array(pred_FDRs)
    sim_FDRs = np.array(sim_FDRs)
    pred_FDRs[np.isnan(pred_FDRs)] = 1
    sim_FDRs[np.isnan(sim_FDRs)] = 1
    
    pred_FDRs = np.histogram(pred_FDRs, bins=np.arange(0, 1.05, 0.05), 
                             density=False)[0]
    sim_FDRs = np.histogram(sim_FDRs, bins=np.arange(0, 1.05, 0.05), 
                            density=False)[0]
    
    pred_FDRs = [el + 1 for el in pred_FDRs]
    sim_FDRs = [el + 1 for el in sim_FDRs]
    
    return(entropy(sim_FDRs, pred_FDRs))

# match predicted FDR distributions
def f_Fv(x):
    print(x)
    mean, std = x
    N = len(PSTHs)
    
    FDRs = []
    done = 0
    while done == 0:
        FDRs_temp = cauchy.rvs(loc=mean, scale=std, size=N)
        FDRs.extend(FDRs_temp[(FDRs_temp >= 0) & (FDRs_temp <= 1)])
        if len(FDRs) >= N:
            done = 1
        
    FDRs = sample(FDRs, N)
    
    Fvs = []
    for i in range(N):
        PSTH = PSTHs[i,:]
        PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
        Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=214, FDR=FDRs[i], T=3))
        print(i)
    
    ISI_viol_dist = np.histogram(ISI_viol, 
                                 bins=np.arange(0, 1.001, 0.001), 
                                 density=False)[0]
    sim_Fv_dist = np.histogram(Fvs, 
                               bins=np.arange(0, 1.001, 0.001), 
                               density=False)[0]
    
    ISI_viol_dist = [el + 1 for el in ISI_viol_dist]
    sim_Fv_dist = [el + 1 for el in sim_Fv_dist]
    
    return(entropy(sim_Fv_dist, ISI_viol_dist))

# %%
    
res = gp_minimize(f_Fv, [(0, 0.5), (0, 0.5)], acq_func="EI", n_calls=30, 
                  n_initial_points=2) 

# %%

Fvs = []
pred_FDRs = []
sim_FDRs = []

for _ in range(10):
    
    N = len(PSTHs)
    
    FDRs = []
    done = 0
    while done == 0:
        FDRs_temp = cauchy.rvs(loc=5.607647803124527e-05, 
                               scale=0.04553339446708389, 
                               size=N)
        FDRs.extend(FDRs_temp[(FDRs_temp >= 0) & (FDRs_temp <= 1)])
        if len(FDRs) >= N:
            done = 1
    

    temp_Fvs = []
    for i in range(N):
        PSTH = PSTHs[i,:]
        PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
        temp_Fvs.append(neuronsim.sim_Fv_PSTH2(PSTH, PSTH_out, N=386, FDR=FDRs[i]))
        print(i)
    
    temp_pred_FDRs = []
    temp_sim_FDRs = []
    for i in range(N):
        PSTH = PSTHs[i,:]
        PSTH_out = np.mean(np.delete(PSTHs, i, axis=0), axis=0)
        out_unit = PSTH_out/np.linalg.norm(PSTH_out)
        temp_pred_FDRs.append(FDR_master(ISI_viol[i], PSTH, out_unit, N=float('inf')))
        temp_sim_FDRs.append(FDR_master(temp_Fvs[i], PSTH, out_unit, N=float('inf')))
        
    temp_pred_FDRs = np.array(temp_pred_FDRs)
    temp_sim_FDRs = np.array(temp_sim_FDRs)
    temp_pred_FDRs[np.isnan(temp_pred_FDRs)] = 1
    temp_sim_FDRs[np.isnan(temp_sim_FDRs)] = 1
    
    Fvs.append(temp_Fvs)
    pred_FDRs.append(temp_pred_FDRs)
    sim_FDRs.append(temp_sim_FDRs)
    
# %%

from scipy.spatial import distance

pred_FDRs_dist = [np.histogram(el, bins=np.arange(0, 1.05, 0.05), density=False)[0] for el in pred_FDRs]
sim_FDRs_dist = [np.histogram(el, bins=np.arange(0, 1.05, 0.05), density=False)[0] for el in sim_FDRs]
Fvs_dist = [np.histogram(el, bins=np.arange(0, 0.1, 0.001), density=False)[0] for el in Fvs]
test = distance.jensenshannon(sim_FDRs_dist[0], sim_FDRs_dist[1])

import itertools

sim_js_distance = []
for a, b in itertools.combinations(sim_FDRs_dist, 2):
    sim_js_distance.append(distance.jensenshannon(a, b))
    
Fv_js_distance = []
for a, b in itertools.combinations(Fvs_dist, 2):
    Fv_js_distance.append(distance.jensenshannon(a, b))
    
# %%

from scipy.interpolate import bisplrep
from scipy import interpolate

iters = [[0.009113132145136961, 0.02394573341153517],
         [0.194071369805847, 0.17765673984296834],
         [0.03295360251651281, 0.022309288354069957],
         [0.2, 0.0],
         [0.19342556295978475, 0.03988921394423471],
         [0.1303424916265658, 0.11196863320037394],
         [0.01029608540876227, 0.2],
         [0.04878782583796082, 0.0],
         [0.02853750630985489, 0.06648751243647193],
         [0.0, 0.05346126735201247],
         [0.0, 0.037700437549754086],
         [0.026828213600310697, 0.040149075106124414],
         [0.0, 0.1222943949711866],
         [0.08220631476782816, 0.2],
         [0.011786989238788216, 0.04145659176254908],
         [0.0, 0.08035581975125475],
         [0.05362002151134298, 0.03735893627494534],
         [0.05931005476333785, 0.1754451241189227],
         [0.016397153356387, 0.04057631190959812],
         [0.026558380380410487, 0.032497970574038464],
         [0.05791151066047656, 0.10319041667168477],
         [0.07873019657846175, 0.06823074481172535],
         [5.607647803124527e-05, 0.04553339446708389],
         [0.001819979757513735, 0.002015122777021783],
         [0.0, 0.15627779286582422],
         [0.026602623399809257, 0.051208476011813844],
         [0.2, 0.09978341495398048],
         [0.06328520800515387, 0.13643734675451033],
         [0.0760826439369125, 0.01530455311302138],
         [0.016263877896465318, 0.030040071062414794]]

iters_mean = [x[0] for x in iters]
iters_std = [x[1] for x in iters]

func_vals = [0.06308345, 1.09251828, 0.05147244, 2.00407503, 1.44572194,
       0.60012055, 0.52708146, 0.4050131 , 0.09242926, 0.02933279,
       0.02811074, 0.02584443, 0.27800801, 0.62333084, 0.02530081,
       0.12337904, 0.09713905, 0.54233249, 0.02698587, 0.02280913,
       0.30284601, 0.29320292, 0.02209349, 0.91796699, 0.37053341,
       0.02654697, 1.16349104, 0.34886082, 0.354732  , 0.0275843 ]

tck = bisplrep(iters_mean, iters_std, func_vals)

xnew, ynew = np.mgrid[0:0.2:0.001, 0:0.2:0.001]
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

from matplotlib import cm

surf = ax.plot_surface(xnew, ynew, znew, linewidth=0,
                       cmap=cm.coolwarm, antialiased=False)
ax.scatter(iters_mean, iters_std, func_vals)

ax.set_xlabel('loc (mean)')
ax.set_ylabel('scale (std)')
ax.set_zlabel('KL divergence')

# %%

from scipy.interpolate import bisplrep
from scipy import interpolate

iters = [[0.06288425916407127, 0.29007182082025745],
         [0.32894425544351397, 0.3705870531611795],
         [0.05220597234125177, 0.41970467027776415],
         [0.08069556680215025, 0.10838597494027749],
         [0.14820447620851296, 0.29514390925572614],
         [0.0, 0.0],
         [0.49846423598892536, 0.18208120984484966],
         [0.46438377926070806, 0.5],
         [0.482967371059048, 0.25173546993844487],
         [0.08952274149373274, 0.5],
         [0.20375832311141961, 0.029014648912361104],
         [0.26219818499751185, 0.35393160312210176],
         [0.38866440067207386, 0.004634287831054441],
         [0.17461898852104332, 0.029623841322425615],
         [0.23027719633291396, 0.4889028895411222],
         [0.2951207103361906, 0.03115634359964716],
         [0.12637836872688155, 0.0],
         [0.06897188646076444, 0.0],
         [0.0970620500957267, 0.2194889618951266],
         [0.24829118813164663, 0.0],
         [0.08421060864866826, 0.2904253558626006],
         [0.19436875092557343, 0.22958701740023818],
         [0.16249383358984612, 0.5],
         [0.3855260078048398, 0.5],
         [0.108598646554308, 0.2094452087142537],
         [0.0, 0.5],
         [0.1616613036142491, 0.1555935540161122],
         [0.5, 0.0],
         [0.24378758233179773, 0.14523381646662323],
         [0.3476342035435761, 0.16311572759642998]]


iters_mean = [x[0] for x in iters]
iters_std = [x[1] for x in iters]

func_vals = [0.02880403, 0.06385662, 0.03926325, 0.02805459, 0.03307887,
       0.31335362, 0.10112985, 0.10180691, 0.10761232, 0.0459981 ,
       0.03618457, 0.05761165, 0.06850799, 0.03767419, 0.06324014,
       0.04339304, 0.05429437, 0.09898217, 0.03204437, 0.0443559 ,
       0.03271726, 0.02857312, 0.05472037, 0.0861642 , 0.02835925,
       0.04977756, 0.02529785, 0.11958709, 0.03817703, 0.05733079]


tck = bisplrep(iters_mean, iters_std, func_vals)

xnew, ynew = np.mgrid[0:0.5:0.001, 0:0.5:0.001]
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

from matplotlib import cm

surf = ax.plot_surface(xnew, ynew, znew, linewidth=0,
                       cmap=cm.coolwarm, antialiased=False)
ax.scatter(iters_mean, iters_std, func_vals)

ax.set_xlabel('loc (mean)')
ax.set_ylabel('scale (std)')
ax.set_zlabel('KL divergence')

# %%

x = np.linspace(0, 1, 1000)

y = []
for val in x:
    y.append(cauchy.pdf(val, loc=5.607647803124527e-05, 
                        scale=0.04553339446708389))

plt.plot(x, y)

# %%

fig, ax = plt.subplots()
plt.hist(sim_FDRs, bins=20)
plt.title('Predicted FDRs with simulated ISI_viol', fontsize=18)

fig, ax = plt.subplots()
plt.hist(pred_FDRs, bins=20)
plt.title('Predicted FDRs with real ISI_viol', fontsize=18)

    
        
            
    