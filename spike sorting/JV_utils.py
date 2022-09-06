# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:11:37 2022

@author: jpv88

"""
import numpy as np
import pickle
import os

from scipy.stats import t
from sklearn.linear_model import LinearRegression

# little n is number of trials, big N is number of bins
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

def norm_zero_to_one(data):
    mindata = min(data)
    maxdata = max(data)
    normed = [(el-mindata)/(maxdata-mindata) for el in data]
    return normed
    
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
    if isinstance(x, list):
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

def pickle_load_file(file):
    with open(file, 'rb') as pickle_file:
        var_name = file[:-7]
        globals()[var_name] = pickle.load(pickle_file)
   
def pickle_load_dir(directory=os.getcwd()):
    
    files = os.listdir(directory)
    
    for file in files:
        if file[-7:] == '.pickle':
            pickle_load_file(file)

@np.vectorize
def lin_eq(m, b, x):
    return m*x + b

def ci_95(sample):
    N = len(sample)
    df = N - 1
    std = np.std(sample)
    mean = np.mean(sample)
    low = t.ppf(0.025, df, loc=mean, scale=std)
    high = t.ppf(0.975, df, loc=mean, scale=std)
    bounds = [low, high]
    
    return bounds
    
    
    
    
    
    
    
    