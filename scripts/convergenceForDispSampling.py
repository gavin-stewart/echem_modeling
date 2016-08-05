#!/bin/python
# Creates a plot of the convergence of various methods of calculating current with kinetic and thermodynamic dispersion.

#Add top level package to python path
import getTopLevel

import tools.fileio as io
import tools.solutionTools as st
import tools.gridTools as gt
import numpy as np
import os.path
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

PTS_PER_WAVE = 200

numEvaluations = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70]
benchmarkNumEvals = 80

fileName = getTopLevel.makeTopLevelPath("files/simulationParameters.json")
dataName = "Martin's experiment"
baseData = io.read_json_params(fileName, dataName)
baseData["type"] = "disp-dimensional-bins"
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("eq_pot", None)
E_0SD = 1e-1
k_0Mean = baseData.pop("eq_rate", None)
k_0SD = 3
freq = baseData["freq"]

endTime = (baseData["pot_rev"] - baseData["pot_start"]) / baseData["nu"]
numPts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
trim = int(np.floor(numPts / 100))


def setupForEqSpParam(numPts):
	E_0Bins = lambda n: gt.unifSpacedParam(n, E_0Mean - 5 * E_0SD, E_0Mean + 5 * E_0SD,E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.unifSpacedParam(n, -10, 10, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def setupForEqSpProb(numPts):
	E_0Bins = lambda n: gt.unifSpacedProb(n, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.unifSpacedProb(n, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def setupForLegGaussParam(numPts):
	E_0Bins = lambda n: gt.leggaussParam(n, E_0Mean -5 * E_0SD, E_0Mean + 5 * E_0SD, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.leggaussParam(n,-10, 10, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def setupForLegGaussProb(numPts):
	E_0Bins = lambda n: gt.leggaussProb(n, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.leggaussProb(n, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def setupForHermGauss(numPts):
	E_0Bins = lambda n: gt.hermgaussParam(n, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.hermgaussParam(n, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def l2Norm(a):
	return np.sum(np.square(a))
 
def genFileName(name):
	return getTopLevel.makeTopLevelPath("files/dispersion/" + name + ".npz")

#names = ["EqSpParam", "EqSpProb", "LegGaussParam", "LegGaussProb", "HermGauss"]
names = ["HermGauss"]
setupFunctions = {"EqSpParam" : setupForEqSpParam, "EqSpProb" : setupForEqSpProb, "LegGaussParam" : setupForLegGaussParam, "LegGaussProb" : setupForLegGaussProb, "HermGauss" : setupForHermGauss}

# Load benchmark data, or generate it

simFileName = genFileName("benchmark")
print simFileName
if os.path.exists(simFileName):
	t, IHR = io.read_time_current_data_bin(simFileName)
else:
	t = np.linspace(0, endTime, numPts)
         time_step = t[1] - t[0]
         num_time_pts = len(t)
	setupForHermGauss(benchmarkNumEvals)
	IHR, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
	del _
	io.write_time_current_bin_cmp(simFileName, t, IHR)
harmHR = st.extract_harmonic(10, freq*endTime, IHR)
harmNorm = l2Norm(harmHR[trim:-trim])
print "Benchmark data loaded"
print "L2 norm of the 10th harmonic was {0}".format(harmNorm)

for name in names:
	print "Beginning processing for {0}".format(name)
	err = []
	harmErr = []

	t = np.linspace(0, endTime, numPts)
	for numSamples in numEvaluations:
		# First, do equally spaced points
		simFileName = genFileName(name+str(numSamples)+"pts")
		if os.path.exists(simFileName):
			_, I = io.read_time_current_data_bin(simFileName)
		else:
			setupFunctions[name](numSamples)
			I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
			io.write_time_current_bin_cmp(simFileName, t, I)
		harm = st.extract_harmonic(10, freq*endTime, I)
		harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
		print "Data for {0} samples loaded".format(numSamples)
		print "\tError was {0}.".format(harmErr[-1])

	plt.title("Convergence in the 10th harmonic for {0}".format(name))
	plt.loglog(numEvaluations, harmErr)
	plt.savefig(getTopLevel.makeTopLevelpath("files/convPlots/harm10Conv{0}.pdf".format(name)))
	plt.close()
