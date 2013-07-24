# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:07:21 2013

@author: shishir
@email: shishir.py@gmail.com
@filename: graddes.py

This program is a demonstration of linear regression 
with optimization done by gradient descent algorithm.

The program shows the contours of cost function and the 
path taken by the optimization.
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt

m = 100
alpha = 0.08
iters = 10000
#np.random.seed(0)
centered = False

x = 10 * np.random.rand(m)
x1 = np.ones(m)

X = np.vstack((x1, x))
X = X.T

beta = [[5],[3]]  #fix the function that generates the data

Y = np.dot(X,beta) + 0.2 * np.random.rand(m,1)

converged = False
    
b = np.zeros((2,1))

diff = []
i = 0
updates = []

# preparing for the contors
x = np.arange(-1,10,0.05)
y = np.arange(-1,10,0.05)

XX,  YY = np.meshgrid(x,y)

tmp = []

print X.shape, XX.shape, YY.shape

#pdb.set_trace()
for i, val in enumerate(X):
    tmp.append(val[0]*XX + val[1] * YY - Y[i])

#pdb.set_trace()   
tmp = np.array(tmp)
tmp = tmp**2/m
print tmp.shape
Z = np.sum(tmp, axis = 0)
print Z.shape

#contor levels
v1 = np.linspace(0.0004, 10, num = 10)
v2 = np.logspace(-3, 4, num = 5, base = 2)
v = np.concatenate((v1,v2))


plt.figure()
CS = plt.contour(XX,YY,Z,v)
plt.clabel(CS, inline = 1, fontsize = 10)
plt.grid('on')


while not converged:
    i += 1
    
    #pdb.set_trace()
    b1 = b.copy()
    b = b - 1.0/(2*m) * alpha * np.dot(X.T, (np.dot(X,b) - Y))
    
    diff.append(np.linalg.norm(b - b1))
    updates.append(b1.ravel())  #add to updates list
    if diff[-1] < 0.00002:
        converged = True
    elif i >= iters:
        converged = True
        
    if i % 100 == 0:
        print "iteration", i
        
print "convereged at", i
print b
#pdb.set_trace()
updates = np.array(updates)
print type(updates)
print updates.shape

print np.max(updates, axis = 0)
print np.min(updates, axis = 0)

plt.scatter(updates[:,0], updates[:,1], marker = "x")
plt.plot(updates[:,0], updates[:,1], 'g--')