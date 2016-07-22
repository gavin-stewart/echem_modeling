# A script for generating high resolution datasets to be compared against lower resolution datasets to guess
# at the minimum numbe rof points needed to have a good resolution.

import dataFileIO as dfio
import numpy as np
import solutionTools

# Data supplied by Martin.
EStart = -0.85
ERev = -0.1
freq = 8.95925020 #Rescaled from Martin's data
dE = 150e-3
nu = 27.94e-3
temp = 25+273.15
area = 0.03
Ru = 100
Cdl = 0.000133878548046
coverage = 1.93972908557 * 6.5e-12
Cdl1 = 0.000653657774506
Cdl2 = 0.000245772700637
Cdl3 = 1.10053945995e-6
E_0 = -0.413798197974
k_0 = 4000 #Rescaled from Martin's paper

tRev = (ERev - EStart) / nu		
tEnd = 2*tRev
n = 1e7

t = np.linspace(0, tEnd, n)

fileName = "/users/gavart/Private/python/electrochemistry/files/highresSimData.npz"

print "Beginning to generate the data. . ."

I = solutionTools.solveIDimensional(t, E_0, dE, freq, k_0, Ru, Cdl,Cdl1, Cdl2,
Cdl3, EStart, ERev, temp, nu, area, coverage, True)[0]

print "Done"
print "Write data to file. . ."
dfio.writeTimeCurrentDataBinary(fileName, t, I, True)
print "Done"
