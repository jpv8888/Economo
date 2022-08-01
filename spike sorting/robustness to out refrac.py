# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:14:55 2022

@author: jpv88
"""

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

import neuronsim
import JV_utils

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = -1/2
    b = Rtot
    c = -Rviol/(2*tviol)
    
    predRout = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    FDR = predRout/Rtot
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

F_v_sim = []
F_v_eq = []
FDR = []
Rtot = []

true_FDRs = []
FDR_eqs = []

refracs = np.arange(0, 2.6, 0.1)

for i in range(1, 10):
    for j in range(1, 10):
        FDR_eq = []
        FDR = []
        for k in refracs:
            F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv(i, j, 
                                                             out_refrac=k, 
                                                             t_stop=1000) 
            FDR_eq.append(economo_eq(i+j, F_v_temp).item())
            FDR.append(FDR_temp) 
            
        true_FDRs.append(np.mean(FDR))
        FDR_eqs.append(FDR_eq/np.mean(FDR))
    
plt.plot(refracs, F_v_sim)

# %%

FDR_mean = np.zeros(26)

for i in range(len(FDR_mean)):
    FDR_mean[i]
    
# %% 

FDR_eq_actual = []

for i, el in enumerate(FDR_eqs):
    FDR_eq_actual.append(el*true_FDRs[i])

max_err = []

for el in FDR_eqs:
    max_err.append(min(el) - 1)

FDR_pred = [(1 - (1 - 4*el)**(0.5))/2 for el in max_err]

# plt.scatter(max_err, true_FDRs)

x = [0, 1]
y = [1- el for el in x]
plt.plot(x, y, ls='--', c='k')

FDR_eq_ratio = [el[-1]/el[0] for el in FDR_eq_actual]
FDR_ratio_pred = [1 - el for el in true_FDRs]

plt.scatter(true_FDRs, FDR_eq_ratio, s=20)
plt.text(0.05, 0.3, '$FDR_{ratio}$ = 1 - $FDR_{true}$', fontsize=14)
plt.xlabel('$FDR_{true}$', fontsize=14)
plt.ylabel('FDR Ratio ($FDR_{eq, min}$/$FDR_{eq, max}$)', fontsize=14)

R2 = str(round(r2_score(FDR_eq_ratio, FDR_ratio_pred), 2))
plt.text(0.05, 0.2, '$R^{2}$ = ' + R2, fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

true_lbs = []
pred_lbs = []

for el in FDR_eq_actual:
    true_lbs.append(el[-1])
    pred_lbs.append(el[0] - el[0]**2)
    
plt.scatter(true_lbs, pred_lbs, s=20)

x = [0.08, 0.31]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

plt.xlabel('True Lower Bound (FDR)', fontsize=14)
plt.ylabel('Predicted Lower Bound (FDR)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

R2 = str(round(r2_score(true_lbs, pred_lbs), 2))
plt.text(0.1, 0.25, '$R^{2}$ = ' + R2, fontsize=14)
plt.tight_layout()



# %%

idx = 8
FDR_eq_temp = FDR_eq_actual[idx]
true_FDR_temp = [true_FDRs[idx]] * len(FDR_eq_temp)
FDR_eq_err = [(el1 - el2) for el1, el2 in zip(FDR_eq_temp, true_FDR_temp)]

plt.xlabel('# Neurons', fontsize=14)
plt.ylabel('Equation Error', fontsize=14)
plt.title("$FDR_{true}$ = 90%", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.scatter(2.5/refracs, FDR_eq_err)
plt.hlines(0, 0, 25, ls='--', color='k')
plt.tight_layout()
# %%

@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

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

F_v_sim = []
F_v_eq = []
FDR = []
Rtot = []

true_FDRs = []
FDR_eqs = []

refracs = [0, 2.5]

# R_tot = range(5, 10)
# FDR_range = np.arange(0.1, 1, 0.1)
R_in_range = list(range(10, 20))
R_out_range = list(range(5, 8))
R_tot_actual = []

# for i in R_in_range:
#     for j in R_out_range:
#         R_tot_actual.append(i + j)

# for val in R_tot:
#     for j in FDR_range:
#         R_in_range.append(int(val*(1 - j)))
#         R_out_range.append(int(val*j))
#         R_tot_actual.append(int(val*(1 - j)) + int(val*j))

# for i, val in enumerate(R_tot_actual):
#     FDR_eq = []
#     FDR = []
#     R_in = R_in_range[i]
#     R_out = R_out_range[i]
#     for k in refracs:
#         F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv(R_in, R_out, 
#                                                          out_refrac=k, 
#                                                          t_stop=1000) 
#         if k == 0:
#             FDR_eq.append(economo_eq(i+j, F_v_temp).item())
#         elif k == 2.5:
#             FDR_eq.append(kleinfeld_eq(i+j, F_v_temp).item())
#         FDR.append(FDR_temp) 
        
#     true_FDRs.append(np.mean(FDR))
#     FDR_eqs.append(FDR_eq)
    
F_v_sim_temp2 = []
for R_in in R_in_range:
    for R_out in R_out_range:
        FDR_eq = []
        FDR = []
        F_v_sim_temp1 = []
        R_tot_actual.append(R_in + R_out)
        for k in refracs:
            F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv(R_in, R_out, 
                                                             out_refrac=k, 
                                                             t_stop=1000) 
            FDR_eq_temp = []
            FDR_eq_temp.append(economo_eq(R_in + R_out, F_v_temp).item())
            FDR_eq_temp.append(kleinfeld_eq(R_in + R_out, F_v_temp).item())
            FDR_eq.append(FDR_eq_temp)
            FDR.append(FDR_temp) 
            F_v_sim_temp1.append(F_v_temp)
            
        true_FDRs.append(np.mean(FDR))
        FDR_eqs.append(FDR_eq)
        F_v_sim_temp2.append(F_v_sim_temp1)
        
# %%


F_v = np.arange(0, 0.038, 0.001)
R_tot = [30] * len(F_v)

economo = economo_eq(R_tot, F_v)
kleinfeld = kleinfeld_eq(R_tot, F_v)

fig, ax = plt.subplots()
ax.plot(F_v, economo, lw=3)
ax.plot(F_v, kleinfeld, lw=3)
ax.fill_between(F_v, economo, kleinfeld, alpha=0.25)

plt.xlabel('$F_{v}$', fontsize=14)
plt.ylabel('FDR', fontsize=14)

plt.title("$R_{tot}$ = 30 Hz", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()


# %%

mean_Fv = [np.mean(el) for el in F_v_sim_temp2]
F_v_sorted = JV_utils.sort_list_by_list(mean_Fv, F_v_sim_temp2)

fig, ax = plt.subplots()
ax.plot([el[0] for el in F_v_sorted])
ax.plot([el[1] for el in F_v_sorted])

# %%

economo_eq_tau0 = [el[0][0] for el in FDR_eqs]
economo_eq_tau1 = [el[1][0] for el in FDR_eqs]

fig, ax = plt.subplots()
ax.scatter(true_FDRs, economo_eq_tau0)
ax.scatter(true_FDRs, economo_eq_tau1)
x = [0.2, 0.5]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

# %%

kleinfeld_eq_tau0 = [el[0][1] for el in FDR_eqs]
kleinfeld_eq_tau1 = [el[1][1] for el in FDR_eqs]

fig, ax = plt.subplots()
ax.scatter(true_FDRs, kleinfeld_eq_tau0)
ax.scatter(true_FDRs, kleinfeld_eq_tau1)
x = [0.2, 0.5]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

# %%
fig, ax = plt.subplots()
ax.scatter(true_FDRs, [el[1][1] for el in FDR_eqs])
x = [0.2, 0.5]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')

        
# %%


FDR_eq_actual = []

for i, el in enumerate(FDR_eqs):
    FDR_eq_actual.append(el)

max_err = []

for el in FDR_eqs:
    max_err.append(min(el) - 1)

FDR_pred = [(1 - (1 - 4*el)**(0.5))/2 for el in max_err]

# plt.scatter(max_err, true_FDRs)

x = [0, 1]
y = [1- el for el in x]
plt.plot(x, y, ls='--', c='k')

FDR_eq_ratio = [el[-1]/el[0] for el in FDR_eq_actual]
FDR_ratio_pred = [1 - el for el in true_FDRs]

plt.scatter(true_FDRs, FDR_eq_ratio, s=20, c=R_tot_actual)
plt.colorbar()
plt.text(0.05, 0.3, '$FDR_{ratio}$ = 1 - $FDR_{true}$', fontsize=14)
plt.xlabel('$FDR_{true}$', fontsize=14)
plt.ylabel('FDR Ratio ($FDR_{eq, min}$/$FDR_{eq, max}$)', fontsize=14)

R2 = str(round(r2_score(FDR_eq_ratio, FDR_ratio_pred), 2))
plt.text(0.05, 0.2, '$R^{2}$ = ' + R2, fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

@np.vectorize
def one_neuron_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = 2*tviol
    b = -2*tviol*Rtot
    c = Rviol
    
    predRout = (-b - (b**2 - 4*a*c)**(1/2))/(2*a)
    
    if predRout > Rtot - predRout:
        FDR = predRout/Rtot
    else:
        FDR = (Rtot - predRout)/Rtot
    
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

FDR_eq = []
FDR = []
for i in range(7, 10):
    for j in range(7, 10):

        F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv(i, j, 
                                                         out_refrac=2.5, 
                                                         t_stop=1000) 
        FDR_eq.append(kleinfeld_eq(i+j, F_v_temp).item())
        FDR.append(FDR_temp)
        
# %%

plt.scatter(FDR, FDR_eq)

x = [0, 1]
y = [el for el in x]
plt.plot(x, y, ls='--', c='k')
    

# %%
# n = list(range(1, 25))
#fn = [el/(el+1) for el in n]

# plt.plot(n, fn)