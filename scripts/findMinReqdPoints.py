# Investigates the minimum number of points needed to accurately represent the current data.

#Get the top level directory of the project, and add it to the path.
import getTopLevel

import numpy as np
from numpy.fft import rfft, irfft
import tools.io as io
import solutionTools as st
import matplotlib.pyplot as plt

# Data supplied by Martin.
EStart = -0.85
ERev = -0.1
freq = 8.95925020
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
k_0 = 4000.0

tRev = (ERev - EStart) / nu		
tEnd = 2*tRev

nVals = [1e6, 5e5, 1e5, 5e4, 1e4]
IVals = []
harmonics = range(1, 11)

fileName = getTopLevel.makeFilePath("highresSimData.npz")

def downSample(data, k):
	"""Returns a list containing every k-th point in the supplied data"""
	l = len(data)
	indices = range(0,l,k)
	return data[indices]

def l2ScaledNorm(x):
	return np.sum(np.square(x) / len(x))


# Read in the high-resolution data
print "Loading high-resolution data. . ."
tHR, IHR = io.readTimeCurrentDataBinary(fileName)
print "Done loading data"

del tHR #To save space
#Generate the data
print "Generating data for lower numbers of points"
for n in nVals:
	print "\tAttempting to find current with {0} steps. . .".format(n)
	t = np.linspace(0,tEnd,n)
         time_step = t[1] - t[0]
         num_time_pts = len(t)
	I, amt = st.solve_reaction_dimensional(time_step, num_time_pts, E_0, dE, freq, k_0, Ru, Cdl, Cdl1, 
	Cdl2, Cdl3, EStart, ERev, temp, nu, area, coverage, True)
	IVals.append(I)
	print "\tDone"

# Test fit in the time domain
print "Testing accuracy in the time domain for. . ."
maxErr = l2ScaledNorm(IHR)
print "Scaled l2 norm of the high-res data was {0}".format(maxErr)
for n,I in zip(nVals, IVals):
	print "\tn = {0}".format(n)
	dsRate = int(np.floor((len(IHR)) / (len(I))))
	IHRDS = downSample(IHR, dsRate)
	err = l2ScaledNorm(IHRDS- I)
	if err > maxErr * 0.01:
		print "\tError {0} was greater than the maximum allowed".format(err)
	else:
		print "\tError {0} was less than 1% of the l2 norm of the true value.".format(err)


# Test fit in the frequency domain
print "Testing accuracy in the frequency domain. . ."
fourierHR = rfft(IHR)
maxErrHarm = []
hrHarmonics = []
harm10Err = []
for h in harmonics:
	hrHarmonics.append(irfft(fourierHR * st.short_centered_kaiser_window(0.75*freq*tEnd, h*freq* tEnd, len(fourierHR))))
	maxErrHarm.append(l2ScaledNorm(hrHarmonics[-1]))
	print "Scaled l2 norm of the high-res data for harmonic #{0} was {1}".format(h, maxErrHarm[-1])
for n,I in zip(nVals, IVals):
	trim = int(np.ceil(n/100))
	print "\t n = {0}".format(n)
	fourier = rfft(I)
	for h, maxError in zip(harmonics, maxErrHarm):
		harmHR = hrHarmonics[h-1]
		harm = irfft(fourier * st.short_centered_kaiser_window(0.75*freq * tEnd, h*freq * tEnd, len(fourier)))
		dsRate = int(np.floor( (len(harmHR)) / (len(harm))))
		harmHR = downSample(harmHR, dsRate)
		err = l2ScaledNorm((harmHR- harm)[trim:-trim])

		if h == 10:
			harm10Err.append(err)

		if err > maxError * 0.01:
			print "\tError {0} was greater than the maximum allowed for harmonic #{1}".format(err, h)
		else:
			print "\tError {0} was less than 1% of the l2 norm of the true value for harmonic #{1}".format(err, h)
plt.plot(np.log(nVals), harm10Err)
plt.show()
