# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:01:53 2022

@author: jpv88
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 20, 0.0001)
y = np.sin(x)

plt.plot(x, y)