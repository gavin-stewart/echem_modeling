#!/bin/python
# The code for plotting harmonics of I.

#Add top level package to python path
import os, sys, os.path
topLevel = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))



import solutionTools
import numpy as np
import matplotlib.pyplot as plt

def plotHarmonic(n, freq, I, pot, trim, color):
	harmonic = solutionTools.extractHarmonic(n, freq, I)
	plt.fill_between(pot[trim:-trim], harmonic[trim:-trim], color=color)

# Code to reproduce Fig 6 of Morris et al 2015
n = 9.1e5
t=np.linspace(0,7,n)
E_0 = 0
dE = 80e-3
freq = 72
k_0Vals = [10000,10000,1000,100]
Ru = 0
Cdl = 0
Cdl1 = 0
Cdl2 = 0
Cdl3 = 0
EStart = -0.2
ERev = 0.5
temp = 293
nu = 0.1
area = 1
coverage = 1e-11
pot = np.linspace(-0.2,0.5,n)
colors = ["blue", "red", "black", "green"]



plotTrim = 1e4  # High-frequency components on the ends make the plot ugly
I = []
for k_0,i in zip(k_0Vals, range(len(k_0Vals))):
	Itmp, amt = solutionTools.solveIDimensional(t, E_0, dE, freq, k_0, 
	Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, 
	coverage, False)
	I.append(Itmp)

for i, color in zip(I, colors):
	plotHarmonic(20, freq*7, i, pot, plotTrim, color)
	plt.hold(True)

plt.show()



