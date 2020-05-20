# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 01:15:44 2019

@author: Aayushi Agarwal
"""

import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
for i in range(50):
    x_temp = i + 1
    u = np.random.uniform(-1, 1)
    y_temp = i + 1 + u
    x.append(x_temp)
    y.append(y_temp)
    
# converting the summation into y*psuedo_inv_of_x gives us the most optimal w0 and w1 values
x_temp = np.linalg.inv(np.matmul(np.array([np.ones(50), x]), np.transpose(np.array([np.ones(50), x]))))
x_transpose = np.transpose(np.array([np.ones(50), x]))
x_psuedo_inv = np.matmul(x_transpose, x_temp)
w = np.matmul(np.array(y) , x_psuedo_inv)

check = np.array([np.ones(50), x]

#print(check)

yn = np.polyval([w[1], w[0]], np.array(x))
print("Polyval",yn)
fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(x, y, c = 'red')
plt.plot(x, yn, c = 'blue')
plt.ylabel('Y - values')
plt.xlabel('X - values')
plt.title('Linear Least Squares')
plt.show()