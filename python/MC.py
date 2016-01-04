# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:24:34 2015
Monte Carlo method to estimate pi
@author: wanghuaq
"""

import numpy as np
import random

n = 10 ** 6
r = 1.0
data = [random.random()**2 + random.random()**2 for x in range(n)]
num = np.sum(np.array(data) < r**2)

p = 4.0 * num / n
print("pi is roughly: " , p)

#data2 = [[random.random(), random.random()] for x in range(n)]
#num2 = np.sum(np.array([x**2 + y**2 for (x,y) in data2]) < r**2)
#
#p2 = 4.0 * num2 / n
#print("pi is roughly: " , p2)

