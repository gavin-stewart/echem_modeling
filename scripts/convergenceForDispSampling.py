# Creates a plot of the convergence of various methods of calculating current with kinetic and thermodynamic dispersion.

import tools.dataFileIO as dfio
import tools.solutionTools as st
import tools.gridTools as gt
import numpy as np
import os.path
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

PTS_PER_WAVE = 200

numEvaluations = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
benchmarkNumEvals = 60

fileName = "./files/simulationParameters.json"
dataName = "Martin's experiment"
baseData = dfio.readParametersFromJSON(fileName, dataName)
baseData["type"] = "disp-dimensional-bins"
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("E_0", None)
E_0SD = 1e-2
k_0Mean = baseData.pop("k_0", None)
k_0SD = 3
freq = baseData["freq"]

endTime = (baseData["ERev"] - baseData["EStart"]) / baseData["nu"]
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
	return "./files/dispersion/" + name + ".npz"

#names = ["EqSpParam", "EqSpProb", "LegGaussParam", "LegGaussProb", "HermGauss"]
names = ["HermGauss"]
setupFunctions = {"EqSpParam" : setupForEqSpParam, "EqSpProb" : setupForEqSpProb, "LegGaussParam" : setupForLegGaussParam, "LegGaussProb" : setupForLegGaussProb, "HermGauss" : setupForHermGauss}

# Load benchmark data, or generate it

simFileName = genFileName("benchmark")
if os.path.exists(simFileName):
	t, IHR = dfio.readTimeCurrentDataBinary(simFileName)
else:
	t = np.linspace(0, endTime, numPts)
	setupForHermGauss(benchmarkNumEvals)
	IHR, _ = st.solveIFromJSON(t, baseData)
	del _
	dfio.writeTimeCurrentDataBinary(simFileName, t, IHR)
harmHR = st.extractHarmonic(10, freq*endTime, IHR)
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
			_, I = dfio.readTimeCurrentDataBinary(simFileName)
		else:
			setupFunctions[name](numSamples)
			I, _ = st.solveIFromJSON(t, baseData)
			dfio.writeTimeCurrentDataBinary(simFileName, t, I)
		harm = st.extractHarmonic(10, freq*endTime, I)
		harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
		print "Data for {0} samples loaded".format(numSamples)

	plt.title("Convergence in the 10th harmonic for {0}".format(name))
	plt.loglog(numEvaluations, harmErr)
	plt.savefig("./files/convPlots/harm10Conv{0}.pdf".format(name))
	plt.close()
