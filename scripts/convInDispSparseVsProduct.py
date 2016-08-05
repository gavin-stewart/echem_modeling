#!/usr/bin/python
# Creates a plot of the convergence of sparse grid vs product grid methods of calculating current with kinetic and thermodynamic dispersion.

import ..tools.io as io
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
baseData = io.read_json_params(fileName, dataName)
baseData["type"] = "disp-dimensional-bins"
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("eq_pot", None)
E_0SD = 5e-3
k_0Mean = baseData.pop("eq_rate", None)
k_0SD = 2
freq = baseData["freq"]

endTime = (baseData["pot_rev"] - baseData["pot_start"]) / baseData["nu"]
num_time_pts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
time_step = endTime / (num_time_pts - 1)
trim = int(np.floor(num_time_pts / 100))

ptSeq = [1,3,5,9,17,33]


def setupForHermGaussProduct(num_time_pts):
	E_0Bins = lambda n: gt.hermgaussParam(n, E_0Mean, E_0SD, False)
	k_0Bins = lambda n: gt.hermgaussParam(n, k_0Mean, k_0SD, True)
	
	baseData["bins"] =  gt.productGrid(E_0Bins, num_time_pts, k_0Bins, num_time_pts)

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
	t, IHR = io.read_time_current_data_bin(simFileName)
else:
	t = np.linspace(0, endTime, num_time_pts)
	setupForHermGauss(benchmarkNumEvals)
	IHR, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
	del _
	io.write_time_current_bin_cmp(simFileName, t, IHR)
harmHR = st.extract_harmonic(10, freq*endTime, IHR)
harmNorm = l2Norm(harmHR[trim:-trim])
print "Benchmark data loaded"
print "Beginning processing for product grid"
err = []
harmErr = []

t = np.linspace(0, endTime, num_time_pts)
for numSamples in numEvaluations:
	# First, do product grids
	simFileName = genFileName("HermGauss"+str(numSamples)+"pts")
	if os.path.exists(simFileName):
		_, I = io.read_time_current_data_bin(simFileName)
	else:
		setupForHermGaussProduct(numSamples)
		I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
		io.write_time_current_bin_cmp(simFileName, t, I)
	harm = st.extract_harmonic(10, freq*endTime, I)
	harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
	del I
	del _
	del harm
	print "Data for {0} samples loaded".format(numSamples)

plt.title("Convergence in the 10th harmonic")
plt.loglog(numEvaluations, harmErr)

num_time_pts = []
harmErr = []
print "Beginning processing for sparse grid"
for level in range(1, len(ptSeq)):
	num_time_pts.append(np.sum(np.array(ptSeq[:level]) * np.array(gt.reverse(ptSeq[:level]))) + np.sum(np.array(ptSeq[:level-1]) * np.array(gt.reverse(ptSeq[:level-1]))))
	simFileName = genFileName("HermGaussSparse"+str(level)+"level")
	if os.path.exists(simFileName):
		_,I = io.read_time_current_data_bin(simFileName)
	else:
		setupForHermGaussSparse(level)
		I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
		io.write_time_current_bin_cmp(simFileName, t, I)
	harm = st.extract_harmonic(10, freq*endTime, I)
	harmErr.append(l2Norm(harmHR[trim:-trim] - harm[trim:-trim]) / harmNorm)
	print "Data for level {0} loaded.".format(level)
plt.loglog(num_time_pts, harmErr)
plt.savefig("./files/convPlots/sparseVsProduct.pdf")
plt.show()

