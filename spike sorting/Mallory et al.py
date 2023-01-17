# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:42:16 2023

@author: jpv88
"""

import scipy.io as sio

data_dir = 'C://Users//jpv88/Downloads//'
mat_fname = 'Freely_moving_data_with_inertial_sensor.mat'

mat_contents = sio.loadmat(data_dir + mat_fname)