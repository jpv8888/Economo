# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:00:19 2022

@author: jpv88
"""
import JV_utils
import os
import pickle
import numpy as np
import neuronsim

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

JV_utils.pickle_load_dir()

PSTHs1 = None
PSTHs2 = None

def pickle_load_file(file):
    with open(file, 'rb') as pickle_file:
        var_name = file[:-7]
        globals()[var_name] = pickle.load(pickle_file)

def pickle_load_dir(directory=os.getcwd()):
    
    files = os.listdir(directory)
    
    for file in files:
        if file[-7:] == '.pickle':
            pickle_load_file(file)
            
pickle_load_dir()

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

# %%
 
PSTHs1_norm = [el/vector_mag(el) for el in PSTHs1]
PSTHs2_norm = [el/vector_mag(el) for el in PSTHs2]

PSTHs1_means = [np.mean(el) for el in PSTHs1]
PSTHs2_means = [np.mean(el) for el in PSTHs2]

mags1 = [vector_mag(el) for el in PSTHs1]
mags2 = [vector_mag(el) for el in PSTHs2]

F_v_eq = []
F_v_sim = []
for i in range(len(PSTHs1) - 600):
    print(i)
    R_in = mags1[i]
    R_out = mags2[i]
    # R_tot = [el1 + el2 for el1, el2 in zip(PSTHs1[i], PSTHs2[i])]
    # R_tot = vector_mag(R_tot)
    
    # R_in = PSTHs1_means[i]
    # R_out = PSTHs2_means[i]
    D = np.dot(PSTHs1_norm[i], PSTHs2_norm[i])
    # D = np.dot(PSTHs1[i], PSTHs2[i])
    Dprime = np.dot(PSTHs2_norm[i], PSTHs2_norm[i])
    # Dprime = np.dot(PSTHs2[i], PSTHs2[i])
    F_v_eq.append((R_in*R_out*D + 0.5*(R_out**2)*Dprime)*(2*0.0025)/(100*(PSTHs1_means[i]+PSTHs2_means[i])))
    # F_v_eq.append((D + 0.5*Dprime)*(2*0.0025)/(R_tot))
    F_v_sim.append(neuronsim.sim_Fv_PSTH(PSTHs1[i], PSTHs2[i])[0])
    
# %%

PSTHs1_norm = [el/vector_mag(el) for el in PSTHs1]
PSTHs2_norm = [el/vector_mag(el) for el in PSTHs2]

PSTHs1_means = [np.mean(el) for el in PSTHs1]
PSTHs2_means = [np.mean(el) for el in PSTHs2]

mags1 = [vector_mag(el) for el in PSTHs1]
mags2 = [vector_mag(el) for el in PSTHs2]

F_v_eq = []
F_v_sim = []
for i in range(len(PSTHs1) - 600):
    print(i)
    R_in = mags1[i]
    R_out = mags2[i]
    # R_tot = [el1 + el2 for el1, el2 in zip(PSTHs1[i], PSTHs2[i])]
    # R_tot = vector_mag(R_tot)
    
    # R_in = PSTHs1_means[i]
    # R_out = PSTHs2_means[i]
    D = np.dot(PSTHs1_norm[i], PSTHs2_norm[i])
    # D = np.dot(PSTHs1[i], PSTHs2[i])
    Dprime = np.dot(PSTHs2_norm[i], PSTHs2_norm[i])
    # Dprime = np.dot(PSTHs2[i], PSTHs2[i])
    F_v_eq.append((R_in*R_out*D + 0.5*(R_out**2)*Dprime)*(2*0.0025)/(100*(PSTHs1_means[i]+PSTHs2_means[i])))
    # F_v_eq.append((D + 0.5*Dprime)*(2*0.0025)/(R_tot))
    F_v_sim.append(neuronsim.sim_Fv_PSTH(PSTHs1[i], PSTHs2[i])[0])

# %%
PSTHs1_norm = [el/vector_mag(el) for el in PSTHs1]
PSTHs2_norm = [el/vector_mag(el) for el in PSTHs2]

PSTHs1_means = [np.mean(el) for el in PSTHs1]
PSTHs2_means = [np.mean(el) for el in PSTHs2]

mags1 = [vector_mag(el) for el in PSTHs1]
mags2 = [vector_mag(el) for el in PSTHs2]

def triple_dot(a, b, c):
    triple_sum = 0
    for i in range(len(a)):
        triple_sum += a[i] + b[i] + c[i]
    
    return triple_sum
    
        
F_v_eq = []
F_v_sim = []
for i in range(len(PSTHs1) - 600):
    print(i)
    R_in = mags1[i]
    R_out = mags2[i]
    # R_tot = [el1 + el2 for el1, el2 in zip(PSTHs1[i], PSTHs2[i])]
    # R_tot = vector_mag(R_tot)
    
    # R_in = PSTHs1_means[i]
    # R_out = PSTHs2_means[i]
    D = np.dot(PSTHs1_norm[i], PSTHs2_norm[i])
    # D = np.dot(PSTHs1[i], PSTHs2[i])
    Dprime = np.dot(PSTHs2_norm[i], PSTHs2_norm[i])
    D3 = triple_dot(PSTHs2[i], PSTHs2[i], PSTHs2[i])
    D4 = triple_dot(PSTHs1[i], PSTHs2[i], PSTHs2[i])
    # Dprime = np.dot(PSTHs2[i], PSTHs2[i])
    term1 = R_in*R_out*D
    term2 = 0.5*(R_out**2)*Dprime
    term3 = D3*-0.0025
    term4 = D4*2*-0.0025
    F_v_eq.append((term1 + term2 + term3 + term4)*(2*0.0025)/(100*(PSTHs1_means[i]+PSTHs2_means[i])))
    # F_v_eq.append((D + 0.5*Dprime)*(2*0.0025)/(R_tot))
    F_v_sim.append(neuronsim.sim_Fv_PSTH(PSTHs1[i], PSTHs2[i])[0])
    
# %%
y_pred, reg = JV_utils.lin_reg(F_v_sim, F_v_eq)

plt.scatter(F_v_sim, F_v_eq, s=10)

F_v_sim_plot = sorted(F_v_sim)
F_v_sim_plot = [F_v_sim_plot[0], F_v_sim_plot[-1]]
F_v_sim_plot = np.array(F_v_sim_plot)
F_v_sim_plot = F_v_sim_plot.reshape(-1, 1)

y_pred = reg.coef_*F_v_sim_plot + reg.intercept_

plt.plot(F_v_sim_plot, y_pred, 'r', lw=2)
plt.plot(F_v_sim_plot, F_v_sim_plot, 'k', ls='--', lw=2)

plt.xlabel('$F_{v}$ (sim)', fontsize=14)
plt.ylabel('$F_{v}$ (eq)', fontsize=14)
plt.title("Equation $F_{v}$ vs. Simulated $F_{v}$", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

dots = [np.dot(el1, el2) for el1, el2 in zip(PSTHs1, PSTHs2)]
err = [el1 - el2 for el1, el2 in zip(F_v_eq, F_v_sim)]

plt.scatter(dots[:221], F_v_sim)

# %%

PSTHs1_norm = [JV_utils.norm_area(el, 100, 6) for el in PSTHs1]
PSTHs2_norm = [JV_utils.norm_area(el, 100, 6) for el in PSTHs2]

F_v_sim = []
FDR = []
Rtot = []
for i in range(len(PSTHs1) - 600):
    F_v_temp, FDR_temp, Rtot_temp = neuronsim.sim_Fv_PSTH(PSTHs1_norm[i],
                                                          PSTHs2_norm[i],
                                                          refractory_period=2.5,
                                                          out_refrac=0)
    F_v_sim.append(F_v_temp)
    FDR.append(FDR_temp) 
    Rtot.append(Rtot_temp)
    
# %%

dots = [np.dot(el1, el2) for el1, el2 in zip(PSTHs1_norm, PSTHs2_norm)]
out_dots = [np.dot(el1, el2) for el1, el2 in zip(PSTHs2_norm, PSTHs2_norm)]

plt.scatter(dots[:121], F_v_sim, s=10)

dots2 = np.column_stack((np.array(dots[:121]), np.array(out_dots[:121])))

y_pred, reg = JV_utils.lin_reg(dots2, F_v_sim)

dots_plot = sorted(dots[:121])
dots_plot = [dots_plot[0], dots_plot[-1]]
dots_plot = np.array(dots_plot)
dots_plot = dots_plot.reshape(-1, 1)

y_pred = reg.coef_*dots_plot + reg.intercept_

plt.plot(dots_plot, y_pred, 'r', lw=2)

plt.xlabel('$PSTH_{1}$ · $PSTH_{2}$ $(Hz^{2})$', fontsize=14)
plt.ylabel('$F_{v}$ (sim)', fontsize=14)
plt.title("$τ_{in}$ = 0.5 ms, $τ_{out}$ = 5 ms", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.text(15000, 0.016, '$R^{2}$ = 0.99', fontsize=14)
plt.tight_layout()

# %%

xx, yy = np.meshgrid(range(10000, 70000, 10000), range(30000, 80000, 10000))
z = xx*reg.coef_[0][0] + yy*reg.coef_[0][1] + reg.intercept_

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(dots[:121], out_dots[:121], F_v_sim)
ax.plot_surface(xx, yy, z, alpha=0.5)
plt.xlabel('In · Out $(Hz^{2})$', fontsize=14)
plt.ylabel('Out · Out $(Hz^{2})$', fontsize=14)
ax.set_zlabel('$F_{v}$')
plt.title("$τ_{in}$ = 5 ms, $τ_{out}$ = 0 ms", fontsize=18)
plt.tight_layout()


# %%

idxs = [27, 41, 46]

plt.plot(PSTHs1_norm[46])
plt.plot(PSTHs2_norm[46])

plt.xlabel('AU')
plt.ylabel('AU')
plt.legend(['In', 'Out'])

# %%

@np.vectorize
def economo_Fv(R_in, R_out, tau):
    num = (R_in*R_out + 0.5*(R_out**2))*2*tau
    den = R_in + R_out
    return num/den

F_v_sim = []
F_v_eq = []
for i in range(1, 10):
    for j in range(1, 10):
        F_v_sim.append(neuronsim.sim_Fv(i, j))
        F_v_eq.append(economo_Fv(i, j, 0.0025))
        
# %% 

sim_data = F_v_sim
F_v_sim = [el[0] for el in F_v_sim]

# %%

plt.scatter(F_v_sim, F_v_eq)

def func(X, a):
    x, y = X
    num = (x*y + a*(y**2))*2*0.0025
    den = x + y
    return num/den

x = []
y = []
for i in range(1, 10):
    for j in range(1, 10):
        x.append(i)
        y.append(j)

popt, pcov = curve_fit(func, (x, y), F_v_sim)

# %%

Rin = [10] * 100
Rout = [5] * 100

Rin = [el/vector_mag(Rin) for el in Rin]
Rout = [el/vector_mag(Rout) for el in Rout]

# %%
R_test = list(range(90, 100))
F_v_sim = []
FDR = []
Rtot = []
bad_time = []
for i in R_test:
    F_v_temp, FDR_temp, Rtot_temp, bad_time_temp = neuronsim.sim_Fv_times(0, i)
    F_v_sim.append(F_v_temp)
    FDR.append(FDR_temp)
    Rtot.append(Rtot_temp)
    bad_time.append(bad_time_temp)

# %%
y_pred, reg = JV_utils.lin_reg(R_test, bad_time)
# plt.plot(R_test, y_pred)
plt.scatter(R_test, bad_time, s=10)

y_ideal = JV_utils.lin_eq(0.005, 0, R_test)
plt.plot(R_test, y_ideal, ls='--')

# %%

overlaps = []
rates = range(0, 20)
for i in rates:
    overlaps.append(neuronsim.sim_Fv_overlap(i, t_stop=100)[3])
    
# %%
def func(x, a):
    return a*(x**2)

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*(x**2)

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 2.5 ms", fontsize=18)
plt.text(1, 0.004, '1.253e-5 * $R_{out}^{2}$', fontsize=14)
plt.text(1, 0.0035, '2 * $τ^{2}$ = 1.25e-5', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

overlaps2 = []
rates = range(0, 20)
for i in rates:
    overlaps2.append(neuronsim.sim_Fv_overlap(i, t_stop=100, refractory_period=5)[3])
    
# %%

def func(x, a):
    return a*(x**2)

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps2))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*(x**2)

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps2, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 5 ms", fontsize=18)
plt.text(1, 0.015, '5.008e-5 * $R_{out}^{2}$', fontsize=14)
plt.text(1, 0.013, '2 * $τ^{2}$ = 5e-5', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

overlaps3 = []
rates = range(0, 20)
for i in rates:
    overlaps3.append(neuronsim.sim_Fv_overlap(i, t_stop=100, refractory_period=1)[3])
    
# %%

def func(x, a):
    return a*(x**2)

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps3))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*(x**2)

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps3, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 1 ms", fontsize=18)
plt.text(1, 0.0006, '2.009e-6 * $R_{out}^{2}$', fontsize=14)
plt.text(1, 0.000525, '2 * $τ^{2}$ = 2e-6', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

overlaps4 = []
rates = range(0, 20)
for i in rates:
    overlaps4.append(neuronsim.sim_Fv_overlap2(5, i, t_stop=100, refractory_period=2.5)[3])
    
# %%

def func(x, a):
    return a*x

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps4))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*x

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps4, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 2.5 ms", fontsize=18)
plt.text(1, 0.002, '1.252e-4 * $R_{out}$', fontsize=14)
plt.text(1, 0.0018, '4 * $τ^{2}$ * $R_{in}$ = 1.25e-4', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%
overlaps5 = []
rates = range(0, 20)
for i in rates:
    overlaps4.append(neuronsim.sim_Fv_overlap2(5, i, t_stop=100, refractory_period=1)[3])
    
# %%

def func(x, a):
    return a*x

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps4))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*x

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps4, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 1 ms", fontsize=18)
plt.text(1, 0.00035, '2.009e-5 * $R_{out}$', fontsize=14)
plt.text(1, 0.00031, '4 * $τ^{2}$ * $R_{in}$ = 2e-5', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%
overlaps6 = []
rates = range(0, 20)
for i in rates:
    overlaps6.append(neuronsim.sim_Fv_overlap2(10, i, t_stop=100, refractory_period=2.5)[3])


# %%

def func(x, a):
    return a*x

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps6))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*x

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps6, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 2.5 ms", fontsize=18)
plt.text(1, 0.004, '2.5005e-4 * $R_{out}$', fontsize=14)
plt.text(1, 0.0035, '4 * $τ^{2}$ * $R_{in}$ = 2.5e-4', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# %%

overlaps7 = []
rates = range(0, 20)
for i in rates:
    overlaps7.append(neuronsim.sim_Fv_overlap2(10, i, t_stop=100, refractory_period=1)[3])
    
# %%

def func(x, a):
    return a*x

popt, pcov = curve_fit(func, np.array(rates), np.array(overlaps7))
# quad_fit = [(0.0025**2)*(el**2) for el in rates]

@np.vectorize
def func(x, a):
    return a*x

quad_fit = func(rates, popt[0])

plt.scatter(rates, overlaps7, s=10)
plt.plot(rates, quad_fit)

plt.xlabel('$R_{out}$ (Hz)', fontsize=14)
plt.ylabel('Overlap Time', fontsize=14)
plt.title("τ = 1 ms", fontsize=18)
plt.text(1, 0.0007, '4.02e-5 * $R_{out}$', fontsize=14)
plt.text(1, 0.00063, '4 * $τ^{2}$ * $R_{in}$ = 4e-5', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()


# %%

#Python 3 
from math import sqrt 
 
def cbrt(x): 
	return x**(1/3) 
 
#get the root of ax^3+bx^2+cx+d=0 
def cubicRoot(a, b, c, d): 
	if not a==0: 
		x=-(b**3)/(27*(a**3))+(b*c)/(6*(a**2))-d/(2*a) 
		y=x**2+(c/(3*a)-b/(9*(a**2)))**3 
		return cbrt(x-sqrt(y))+cbrt(x+sqrt(y))-b/(3*a) 
	#dealing with quadtric equation 0x^3+bx^2+cx+d=0 
	elif not b==0: 
		br=c**2-4*b*d 
		rt=(-c+sqrt(br))/(2*b), (-c-sqrt(br))/(2*b)  
		return rt if not br==0 else -c/(2*b) 
	#linear equation cx+d=0 
	elif not c==0: 
		return -d/c 
	else: 
		#regardless of x we have d=0 
		if d==0: 
			return "all numbers fit" 
		else: 
			return "no number fit" 
        
@np.vectorize
def economo_eq(Rtot, F_v, tviol=0.0025):

    Rviol = Rtot*F_v
    
    a = tviol
    b = -0.5 - Rtot*2*tviol
    c = Rtot
    d = -Rviol/(2*tviol)
    
    predRout = cubicRoot(a, b, c, d)
    
    FDR = predRout/Rtot
    
    if isinstance(FDR, complex):
        FDR = 1

    return FDR

@np.vectorize
def economo_Fv_new(R_in, R_out, tau):
    Rtot = R_in + R_out
    num = ((R_out**3)*2*(tau**2)) + (-tau - Rtot*4*(tau**2))*(R_out**2) + (2*tau*Rtot*R_out)
    den = R_in + R_out
    return num/den
        











    