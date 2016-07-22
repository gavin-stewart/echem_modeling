#!/usr/bin/python
# Creates a plot of the convergence of sparse grid vs product grid methods of calculating current with kinetic and thermodynamic dispersion.

import ..tools.dataFileIO as dfio
import ..tools.solutionTools as st
import ..tools.gridTools as gt
import numpy as np
import os.path
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

PTS_PER_WAVE = 200

numEvaluations = [1, 2, 3, 4, 5,6,7,8,9, 10, 15, 20, 30, 40, 50, 60, 70]
benchmarkNumEvals = 70

fileName = "./files/simulationParameters.json"
dataName = "Martin's experiment"
baseData = dfio.readParametersFromJSON(fileName, dataName)
baseData["type"] = "disp-dimensional-bins"
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("E_0", None)
E_0SD = 5e-3
k_0Mean = baseData.pop("k_0", None)
k_0SD = 2
freq = baseData["freq"]

endTime = (baseData["ERev"] - baseData["EStart"]) / baseData["nu"]
numPts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
trim = int(np.floor(numPts / 100))

ptSeq = [1,3,5,9,17,33]


def setupForHermGaussProduct(numPts):
	E_0Bins = lambda n: gt.hermgaussParam(n, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.hermgaussParam(n, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, numPts, k_0Bins, numPts)

def setupForHermGaussSparse(level):
	ERule = lambda n: gt.hermgaussParam(n, E_0Mean, E_0SD, False)
	KRule = lambda n: gt.hermgaussParam(n, k_0Mean, k_0SD, True)

	baseData["bins"] = gt.sparseGrid(ERule, KRule, level, ptSeq, ptSeq)
	print baseData["bins"]

def l2Norm(a):
	return np.sum(np.square(a))
 
def genFileName(name):
	return "./files/dispersion/" + name + ".npz"

names = ["HermGaussSparse", "HermGauss"]
setupFunctions = {"HermGaussSparse" : setupForHermGaussSparse, "HermGauss" : setupForHermGaussProduct}

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
print "Beginning processing for product grid"
err = []
harmErr = []

t = np.linspace(0, endTime, numPts)
for numSamples in numEvaluations:
	# First, do product grids
	simFileName = genFileName("HermGauss"+str(numSamples)+"pts")
	if os.path.exists(simFileName):
		_, I = dfio.readTimeCurrentDataBinary(simFileName)
	else:
		setupForHermGaussProduct(numSamples)
		I, _ = st.solveIFromJSON(t, baseData)
		dfio.writeTimeCurrentDataBinary(simFileName, t, I)
	harm = st.extractHarmonic(10, freq*endTime, I)
	harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
	del I
	del _
	del harm
	print "Data for {0} samples loaded".format(numSamples)

plt.title("Convergence in the 10th harmonic")
plt.loglog(numEvaluations, harmErr)

numPts = []
harmErr = []
print "Beginning processing for sparse grid"
for level in range(1, len(ptSeq)):
	numPts.append(np.sum(np.array(ptSeq[:level]) * np.array(gt.reverse(ptSeq[:level]))) + np.sum(np.array(ptSeq[:level-1]) * np.array(gt.reverse(ptSeq[:level-1]))))
	simFileName = genFileName("HermGaussSparse"+str(level)+"level")
	if os.path.exists(simFileName):
		_,I = dfio.readTimeCurrentDataBinary(simFileName)
	else:
		setupForHermGaussSparse(level)
		I, _ = st.solveIFromJSON(t, baseData)
		dfio.writeTimeCurrentDataBinary(simFileName, t, I)
	harm = st.extractHarmonic(10, freq*endTime, I)
	harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
	print "Data for level {0} loaded.".format(level)
plt.loglog(numPts, harmErr)
plt.savefig("./files/convPlots/sparseVsProduct.pdf")
plt.show()

