# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:18:30 2022

@author: jpv88
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import neuronsim
from scipy.signal import savgol_filter
import matplotlib.cm as cm
import math

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

Rtot = np.loadtxt('Rtot.csv')
F_v = np.loadtxt('F_v.csv')
ranges = np.loadtxt('ranges.csv', delimiter=',')
ranges = ranges - 1
ranges = ranges.astype(int)
sessions = np.loadtxt('sessions.csv', delimiter=',')


# %%


import csv
  
with open('trials.csv', 'r') as read_obj:
  
    # Return a reader object which will
    # iterate over lines in the given csvfile
    csv_reader = csv.reader(read_obj)
  
    # convert string to list
    trials = list(csv_reader)

for i, test_list in enumerate(trials):
    trials[i] = [int(el)-1 for el in test_list if el]
    
with open('spikes.csv', 'r') as read_obj:
  
    # Return a reader object which will
    # iterate over lines in the given csvfile
    csv_reader = csv.reader(read_obj)
  
    # convert string to list
    spikes = list(csv_reader)

for i, test_list in enumerate(spikes):
    spikes[i] = [float(el) for el in test_list if el]
    
# %%


def spikes_to_firing_rates(spikes, n, T=6, N=100):
    delta = T/N
    bins = np.zeros(N)
    for i in range(N):
        for j in spikes:
            if (j >= i*delta) and (j < (i+1)*delta):
                bins[i] += 1
                
    bins = bins/(delta*n)
    # bins = savgol_filter(bins, 11, 4)
    return bins

def norm_neg_one_to_one(data):
    mindata = min(data)
    maxdata = max(data)
    normed = [(2*(el-mindata)/(maxdata-mindata)) - 1 for el in data]
    return normed

def point_dist_to_line(x, y, a, b, c):
    num = abs(a*x + b*y + c)
    den = (a**2 + b**2)**(1/2)
    return num/den

def lin_reg(x, y):
    x = np.array(x)
    x = x.reshape(-1, 1)
    y = np.array(y)
    y = y.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    y_pred = reg.coef_*x + reg.intercept_
    print('R^2 = ' + str(reg.score(x, y)))
    return y_pred, reg

def scale_zero_to_x(data, x=1):
    min_data = min(data)
    max_data = max(data)
    range_data = max_data - min_data
    return([((el - min_data)/range_data)*x for el in data])

# x is length of the x axis
def norm_area(data, area, x):
    dx = x/len(data)
    data_area = np.trapz(data, dx=dx)
    return([el/(data_area/area) for el in data])

def sort_list_by_list(idx_list, sort_list):
    
    if len(idx_list) != len(sort_list):
        raise Exception('Lists must be of equal length')
        
    zipped_lists = zip(idx_list, sort_list)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list = [element for _, element in sorted_zipped_lists]
    
    return sorted_list
        
    
    
    
# %%

sessions_idx = np.unique(sessions)

t_viol = 0.0025
sessions_track = []
F_v = []
F_v_sim = []
corr = []
CV = []
CV1 = []
CV2 = []
PSTHs1 = []
PSTHs2 = []
Rtots1 = []
Rtots2 = []

# iterate through sessions
for idx in sessions_idx[:5]:
    session_mask = (sessions == idx)
    session_ranges = ranges[session_mask]
    session_spikes = [i for (i, v) in zip(spikes, session_mask) if v]
    session_trials = [i for (i, v) in zip(trials, session_mask) if v]
    
    good_pairs = []
    for i, unit_range in enumerate(session_ranges):
        for j, unit_range2 in enumerate(session_ranges[i+1:,:]):
            if all(unit_range == unit_range2):
                good_pairs.append([i, j+i+1])
          
    
    
    # iterate through pairs in a session
    for pair in good_pairs:
        neuron1 = session_spikes[pair[0]]
        neuron2 = session_spikes[pair[1]]
        neuron1_tri = session_trials[pair[0]]
        neuron2_tri = session_trials[pair[1]]
        
        pair_range = session_ranges[pair[0]]
        pair_range = range(pair_range[0], pair_range[1])
        
        # iterate through trials in a pair
        ISI_viol = 0
        spks1_tot = 0
        spks2_tot = 0
        spks1_all = []
        spks2_all = []
        ISIs_all = []
        ISIs1_all = []
        ISIs2_all = []
        CVpertrial = []
        for i in tqdm(pair_range):
            idx_1 = [(el == i) for el in neuron1_tri]
            idx_2 = [(el == i) for el in neuron2_tri]
            spks1 = [i for (i, v) in zip(neuron1, idx_1) if v]
            spks2 = [i for (i, v) in zip(neuron2, idx_2) if v]
            if (len(spks1) == 0) and (len(spks2) == 0):
                continue
            spks_tot = spks1 + spks2
            spks_tot.sort()
            ISIs1 = np.diff(spks1)
            ISIs2 = np.diff(spks2)
            if ISIs1.size != 0:
                ISIs1_all.extend(ISIs1)
            if ISIs2.size != 0:
                ISIs2_all.extend(ISIs2)
            ISIs = np.diff(spks_tot)
            if ISIs.size != 0:
                ISIs_all.extend(ISIs)
            # current_CV = np.std(ISIs)/np.mean(ISIs)
            # if not math.isnan(current_CV):
            #     CVpertrial.append(current_CV)
              
            ISI_viol += sum(np.diff(spks_tot) < t_viol)
            spks1_tot += len(spks1)
            spks2_tot += len(spks2)
            spks1_all.extend(spks1)
            spks2_all.extend(spks2)
        
        CV.append(np.std(ISIs_all)/np.mean(ISIs_all))
        CV1.append(np.std(ISIs1_all)/np.mean(ISIs1_all))
        CV2.append(np.std(ISIs2_all)/np.mean(ISIs2_all))
        tot = spks1_tot + spks2_tot
        F_v.append(ISI_viol/tot)
        sessions_track.append(idx)
        
        num_tri = len(pair_range)
        Rtot1 = spks1_tot/(num_tri*6)
        Rtot2 = spks2_tot/(num_tri*6)
        
        Rtots1.append(Rtot1)
        Rtots2.append(Rtot2)
        
        spks1_PSTH = spikes_to_firing_rates(spks1_all, num_tri)
        spks2_PSTH = spikes_to_firing_rates(spks2_all, num_tri)
        
        PSTHs1.append(spks1_PSTH)
        PSTHs2.append(spks2_PSTH)
        
        corr.append(stats.pearsonr(spks1_PSTH, spks2_PSTH)[0])
        
        F_v_sim.append(neuronsim.sim_Fv(Rtot1, Rtot2, out_refrac=2.5))
        
# %%

corr_sort = sorted(corr)

PSTHs1_sort = sort_list_by_list(corr, PSTHs1)
PSTHs2_sort = sort_list_by_list(corr, PSTHs2)

PSTHs1_norm = [norm_area(el, 100, 6) for el in PSTHs1_sort]
PSTHs2_norm = [norm_area(el, 100, 6) for el in PSTHs2_sort]

cross_corr = [np.dot(el0, el1) for el0, el1 in zip(PSTHs1_norm, PSTHs2_norm)]
cross_corr_sort = sorted(cross_corr)

PSTHs1_norm_sort = sort_list_by_list(cross_corr, PSTHs1_norm)
PSTHs2_norm_sort = sort_list_by_list(cross_corr, PSTHs2_norm)

# %%

F_v_PSTH = []
for i in range(len(PSTHs1)):
    F_v_PSTH.append(neuronsim.sim_Fv_PSTH(PSTHs1_norm[i], PSTHs2_norm[i]))
    
F_v_PSTH_sort = sort_list_by_list(cross_corr, F_v_PSTH)

# %%

corr_norm = [np.corrcoef(x, y)[0,1] for x, y in zip(PSTHs1_norm, PSTHs2_norm)] 

# %%

plt.scatter(cross_corr_sort, F_v_PSTH_sort, s=10)

y_pred, reg = lin_reg(cross_corr_sort, F_v_PSTH_sort)
plt.plot(cross_corr_sort, y_pred, c='r')

plt.xlabel('X-Corr (AU)')
plt.ylabel('$F_{v}$ (sim, AU)')    
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
    
# %%

plt.scatter(corr_sort, F_v_PSTH, s=10)

def func(x, a, b):
    return (a*(np.exp(x) - np.exp(-1*x))/2) + b

popt, pcov = curve_fit(func, np.array(corr_sort), np.array(F_v_PSTH))

@np.vectorize
def func(x, a, b):
    return (a*(np.exp(x) - np.exp(-1*x))/2) + b

x = np.linspace(-1, 1, 100)
y_pred = func(x, popt[0], popt[1])
plt.plot(x, y_pred, 'r')

plt.xlabel('Correlation (PCC)')
plt.ylabel('$F_{v}$ (sim, AU)')    
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()


# %%

F_v = np.array(F_v)
F_v = F_v.reshape(-1, 1)
F_v_sim = np.array(F_v_sim)
F_v_sim = F_v_sim.reshape(-1, 1)
reg = LinearRegression().fit(F_v_sim, F_v)
# corr_adjust = norm_neg_one_to_one(corr)

# colors = cm.coolwarm(corr)
fig, ax = plt.subplots(nrows=1, ncols=1)
sc = plt.scatter(F_v_sim, F_v, c=corr, cmap=cm.coolwarm, vmin=-1, vmax=1, s=10)
# sc = plt.scatter(F_v_sim, F_v, c=colors, vmin=-1, vmax=1)
# for x, y, c in zip(F_v_sim, F_v, colors):
#     sc = plt.scatter(x, y, color=c, vmin=-1, vmax=1)

sm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
sm.set_clim(vmin=-1, vmax=1)
cb = plt.colorbar(sm)
cb.ax.get_yaxis().labelpad = 20
cb.ax.set_ylabel('Correlation (PCC)', rotation=270)
cb.ax.tick_params(labelsize=10) 

# plt.scatter(F_v_sim, F_v, s=4)

dim = max(np.vstack((F_v, F_v_sim))) + 0.005
x = np.linspace(0, dim, 2)
y = reg.coef_*x + reg.intercept_
plt.plot(x, y, c='k')

x = np.linspace(0, dim, 2)
y = x
plt.plot(x, y, c='k', ls='--')

plt.xlabel('$F_{v}$ (sim)')
plt.ylabel('$F_{v}$ (data)')
plt.xlim(0, dim)
plt.ylim(0, dim)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# %%
dists = []
for x, y in zip(F_v_sim, F_v):
    if y >= x:
        dists.append(point_dist_to_line(x, y, -1, 1, 0))
    else:
        dists.append(-1*point_dist_to_line(x, y, -1, 1, 0))
        
dists = [el[0] for el in dists]
fig, ax = plt.subplots(nrows=1, ncols=1)
corr_abs = [abs(el) for el in corr]
plt.scatter(corr, dists, s=10)

plt.xlabel('Correlation (PCC)')
plt.ylabel('Signed Perpendicular Distance')    
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

def func(x, a, b, c):
    return ((np.exp(a*x) - np.exp(b*-1*x))/2) + c

popt, pcov = curve_fit(func, np.array(corr), np.array(dists))

@np.vectorize
def func(x, a, b, c):
    return ((np.exp(a*x) - np.exp(b*-1*x))/2) + c

x = np.linspace(-1, 1, 100)
y_pred = func(x, popt[0], popt[1], popt[2])
plt.plot(x, y_pred, 'r')

    

plt.hlines(0, -1, 1, ls='--', colors='k')
plt.title('Error vs. Correlation (Signed Perpendicular Distance)', fontsize=14)
fig.tight_layout()



# %%

res = []
for i, el in enumerate(F_v_sim):
    res.append(F_v[i]- el)

res = [el.item() for el in res]

plt.scatter(corr, res, s=10)

# corr_test = np.array(corr)
# corr_test = corr_test.reshape(-1, 1)
# res = np.array(res)
# res = res.reshape(-1, 1)
# reg = LinearRegression().fit(corr_test, res)

# dim = 1
# x = np.linspace(-1, dim, 2)
# y = reg.coef_*x + reg.intercept_

# plt.plot(x, y[0], c='r')
# plt.xlabel('Correlation (PCC)')
# plt.ylabel('Residual')    

# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)

plt.xlabel('Correlation (PCC)')
plt.ylabel('Residual')    
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

def func(x, a, b, c):
    return ((np.exp(a*x) - np.exp(b*-1*x))/2) + c

popt, pcov = curve_fit(func, np.array(corr), np.array(res))

@np.vectorize
def func(x, a, b, c):
    return ((np.exp(a*x) - np.exp(b*-1*x))/2) + c

x = np.linspace(-1, 1, 100)
y_pred = func(x, popt[0], popt[1], popt[2])
plt.plot(x, y_pred, 'r')

plt.hlines(0, -1, 1, ls='--', colors='k')
plt.title('Error vs. Correlation (Residual)', fontsize=14)
fig.tight_layout()
    

# %%
CVavg = [np.mean([x, y]) for x, y in zip(CV1, CV2)]

fig, ax = plt.subplots(nrows=1, ncols=1)
CVmax = max(CVavg)
sc = plt.scatter(F_v_sim, F_v, c=CVavg, cmap=cm.Reds, vmin=1, vmax=CVmax)
# sc = plt.scatter(F_v_sim, F_v, c=colors, vmin=-1, vmax=1)
# for x, y, c in zip(F_v_sim, F_v, colors):
#     sc = plt.scatter(x, y, color=c, vmin=-1, vmax=1)

sm = plt.cm.ScalarMappable(cmap=cm.Reds)
sm.set_clim(vmin=1, vmax=CVmax)
plt.colorbar(sm)

# %%

y_pred, reg = lin_reg(corr, CVavg)
print(stats.pearsonr(corr,CVavg)[0])
plt.scatter(corr, CVavg, s=10)
plt.plot(corr, y_pred, 'r')
plt.xlim(-1, 1)
plt.xlabel('Correlation (PCC)')
plt.ylabel('Poisson Inhomogeneity (CV)')
plt.vlines(0, 1, 3, ls='--', colors='k')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# %%
# CVdist = [abs(el-1) for el in CV]
# fig, ax = plt.subplots(nrows=1, ncols=1)
# CVmax = max(CVdist)
# sc = plt.scatter(F_v_sim, F_v, c=CVdist, cmap=cm.Reds, vmin=0, vmax=CVmax)
# # sc = plt.scatter(F_v_sim, F_v, c=colors, vmin=-1, vmax=1)
# # for x, y, c in zip(F_v_sim, F_v, colors):
# #     sc = plt.scatter(x, y, color=c, vmin=-1, vmax=1)

# sm = plt.cm.ScalarMappable(cmap=cm.Reds)
# sm.set_clim(vmin=0, vmax=CVmax)
# plt.colorbar(sm)


# %%

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    fp = predRout/Rtot
    
    if isinstance(fp, complex):
        fp = 1

    return fp

fps = economo_eq(Rtot, F_v)

# %%  



plt.boxplot(np.column_stack((Rtot, F_v, fps)))



