#!/bin/python

# Add top level directory to pythonpath
import sys,getTopLevel

# Plot and display the harmonics for the dispersion data.

import matplotlib.pyplot as plt
import tools.dataFileIO as dfio
import tools.solutionTools as st
import tools.gridTools as gt
import scipy.signal
import numpy as np
import matplotlib.collections
import itertools
import matplotlib.cm as cm
import os.path

colors = itertools.cycle(cm.rainbow(np.linspace(0,1,4)))

numEvals = 40

fileName = getTopLevel.makeFilePath("/simulationParameters.json")

baseData = dfio.readParametersFromJSON(fileName, "Martin's experiment")
freq = baseData["freq"]
EMean = baseData["E_0"]
kMean = baseData["k_0"]
ESDVals = [1e-3, 1e-2, 1e-1]
kSDVals = [1,2,3]
trim = slice(5000, -5000)


E_0Bins = lambda n: gt.hermgaussParam(n, EMean, ESD, False)
k_0Bins = lambda n: gt.hermgaussParam(n, kMean, kSD, True)

IEnvLim = 0

plotsE0 = plt.figure(1)
plotsk0	= plt.figure(2)
plotsboth = plt.figure(3)

plotsIE0 = plotsE0.add_subplot(211)
plotsHarmE0 = plotsE0.add_subplot(212)
plotsIk0 = plotsk0.add_subplot(211)
plotsHarmk0 = plotsk0.add_subplot(212)
plotsIboth = plotsboth.add_subplot(211)
plotsHarmboth = plotsboth.add_subplot(212)

def plotCurrentAndHarmonics(I, label, currentAx, harmAx):
	global IEnvLim
	plotColor = next(colors)
	IEnvUpper = st.interpolatedUpperEnvelope(t, I-ICap)
	IEnvLower = st.interpolatedLowerEnvelope(t, I-ICap)
	IEnvMax = max(np.amax(abs(IEnvUpper)), np.amax(abs(IEnvLower)))
	if IEnvMax > IEnvLim: 
		IEnvLim = IEnvMax
	envelopes = matplotlib.collections.LineCollection([list(zip(t, IEnvUpper)), list(zip(t, IEnvLower))], label=label, color=plotColor)
	currentAx.add_collection(envelopes)
	harm10 = st.extractHarmonic(10, freq * t[-1], I)[trim]
	harm10 = abs(scipy.signal.hilbert(harm10))
	harmAx.plot(harm10, label=label, color=plotColor)

def setupAndSavePlots(fig, currentAx, harmAx, name):
	leg = harmAx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), mode="expand", borderaxespad=0., ncol=2)
	currentAx.set_xlim([t[0], t[-1]])
	currentAx.set_ylim([-1.25 * IEnvLim, 1.25 * IEnvLim])
	fig.savefig(os.path.join(figLoc, name), bbox_extra_artists=(leg,), bbox_inches='tight', pad_inches=0.5)

# Read in where the plots should be saved
figLoc = raw_input("Enter the directory where the figures should be saved: ")

# No dispersion.
fileName = getTopLevel.makeFilePath("dispersion/HermGauss1pts.npz")
t,INoDisp = dfio.readTimeCurrentDataBinary(fileName)
#Generate a zero-capacitance model 
baseData["k_0"] = 0
ICap, _ = st.solveIFromJSON(t, baseData)
baseData["k_0"] = kMean
baseData["type"] = "disp-dimensional-bins"


# Dispersion in E_0 only.
kSD = 0
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIE0, plotsHarmE0)
for ESD in ESDVals:
	baseData["bins"] = gt.productGrid(E_0Bins, numEvals, k_0Bins, 1)
	I, _ = st.solveIFromJSON(t, baseData)
	label = "E_0 disp " + str(ESD)
	plotCurrentAndHarmonics(I, label, plotsIE0, plotsHarmE0) 



# Dispersion in k_0 only.
ESD = 0
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIk0, plotsHarmk0)
for kSD in kSDVals:
	baseData["bins"] = gt.productGrid(E_0Bins, 1, k_0Bins, numEvals)
	I, _ = st.solveIFromJSON(t, baseData)
	label = "k_0 disp " + str(kSD)
	plotCurrentAndHarmonics(I, label, plotsIk0, plotsHarmk0)
 


# Dispersion in both E_0 and k_0.
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIboth, plotsHarmboth)
for ESD, kSD in zip(ESDVals, kSDVals):
	baseData["bins"] = gt.productGrid(E_0Bins, 1, k_0Bins, numEvals)
	I, _ = st.solveIFromJSON(t, baseData)
	label = "E_0 disp " + str(ESD) + " k_0 disp " + str(kSD)
	plotCurrentAndHarmonics(I, label, plotsIboth, plotsHarmboth)

#Save plots
setupAndSavePlots(plotsE0, plotsIE0, plotsHarmE0, "E0Dispersion.pdf")

print "Saved E0 dispersion plots"

setupAndSavePlots(plotsk0, plotsIk0, plotsHarmk0, "k0Dispersion.pdf")
print "Saved k0 dispersion plots"

setupAndSavePlots(plotsboth, plotsIboth, plotsHarmboth, "bothDispersion.pdf")
print "Saved both dispersion plots"
