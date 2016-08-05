#!/bin/python

# Add top level directory to pythonpath
import sys,getTopLevel

# Plot and display the harmonics for the dispersion data.

import matplotlib.pyplot as plt
import tools.fileio as io
import tools.solution_tools as st
import tools.grid as gt
import scipy.signal
import numpy as np
import matplotlib.collections
import itertools
import matplotlib.cm as cm
import os.path

colors = itertools.cycle(cm.rainbow(np.linspace(0,1,4)))

numEvals = 40

fileName = getTopLevel.makeFilePath("/simulationParameters.json")

baseData = io.read_json_params(fileName, "Martin's experiment")
freq = baseData["freq"]
EMean = baseData["eq_pot"]
kMean = baseData["eq_rate"]
ESDVals = [1e-3, 1e-2, 1e-1]
kSDVals = [1,2,3]
trim = slice(5000, -5000)


E_0Bins = lambda n: gt.hermgauss_param(n, EMean, ESD, False)
k_0Bins = lambda n: gt.hermgauss_param(n, kMean, kSD, True)

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
	IEnvUpper = st.interpolated_upper_envelope(t, I-ICap)
	IEnvLower = st.interpolated_lower_envelope(t, I-ICap)
	IEnvMax = max(np.amax(abs(IEnvUpper)), np.amax(abs(IEnvLower)))
	if IEnvMax > IEnvLim: 
		IEnvLim = IEnvMax
	envelopes = matplotlib.collections.LineCollection([list(zip(t, IEnvUpper)), list(zip(t, IEnvLower))], label=label, color=plotColor)
	currentAx.add_collection(envelopes)
	harm10 = st.extract_harmonic(10, freq * t[-1], I)[trim]
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
ESD = 0
kSD = 0
tEnd = 2 * (baseData["pot_rev"] - baseData["pot_start"]) / baseData["nu"]
num_time_pts = int(np.ceil(tEnd * baseData["freq"] * 200)
time_step = tEnd / num_time_pts
t = np.linspace(0, tEnd, num_time_pts) 
baseData["bins"] = gt.product_grid(E_0Bins, 1, k_0Bins, 1)
INoDisp, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
#Generate a zero-capacitance model 
baseData["eq_rate"] = 0
ICap, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
baseData["eq_rate"] = kMean
baseData["type"] = "disp-dimensional-bins"


# Dispersion in E_0 only.
kSD = 0
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIE0, plotsHarmE0)
for ESD in ESDVals:
	baseData["bins"] = gt.product_grid(E_0Bins, numEvals, k_0Bins, 1)
	I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
	label = "E_0 disp " + str(ESD)
	plotCurrentAndHarmonics(I, label, plotsIE0, plotsHarmE0) 



# Dispersion in k_0 only.
ESD = 0
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIk0, plotsHarmk0)
for kSD in kSDVals:
	baseData["bins"] = gt.product_grid(E_0Bins, 1, k_0Bins, numEvals)
	I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
	label = "k_0 disp " + str(kSD)
	plotCurrentAndHarmonics(I, label, plotsIk0, plotsHarmk0)
 


# Dispersion in both E_0 and k_0.
plotCurrentAndHarmonics(INoDisp, "No disp", plotsIboth, plotsHarmboth)
for ESD, kSD in zip(ESDVals, kSDVals):
	baseData["bins"] = gt.product_grid(E_0Bins, 1, k_0Bins, numEvals)
	I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
	label = "E_0 disp " + str(ESD) + " k_0 disp " + str(kSD)
	plotCurrentAndHarmonics(I, label, plotsIboth, plotsHarmboth)

#Save plots
setupAndSavePlots(plotsE0, plotsIE0, plotsHarmE0, "E0Dispersion.pdf")

print "Saved E0 dispersion plots"

setupAndSavePlots(plotsk0, plotsIk0, plotsHarmk0, "k0Dispersion.pdf")
print "Saved k0 dispersion plots"

setupAndSavePlots(plotsboth, plotsIboth, plotsHarmboth, "bothDispersion.pdf")
print "Saved both dispersion plots"
