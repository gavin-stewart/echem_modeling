#!/bin/python
# Creates a plot of the convergence of various methods of calculating current with kinetic and thermodynamic dispersion.

import electrochemistry.tools.fileio as io
import electrochemistry.tools.solution_tools as st
import electrochemistry.tools.grid as gt
import numpy as np
import os.path
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

PTS_PER_WAVE = 200

numEvaluations = [1, 5, 10, 15, 20, 30, 40, 50, 60]
benchmarkNumEvals = 85
HARMONIC_RANGE = range(5, 13)

fileName = io.get_file_resource_path("simulationParameters.json")
dataName = "Dispersion points plot"
baseData = io.read_json_params(fileName, dataName)
baseData["type"] = "disp-dimensional-bins"
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("eq_pot", None)
E_0SD = 1e-1
k_0Mean = baseData.pop("eq_rate", None)
k_0SD = np.sqrt(np.log(1 + 8**2)) #Coefficient of variation of 8.
freq = baseData["freq"]

endTime = (baseData["pot_rev"] - baseData["pot_start"]) / baseData["nu"]
numPts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
trim = int(np.floor(numPts / 100))


def setupForEqSpParam(numPts):
    E_0Bins = lambda n: gt.unif_spaced_param(n, E_0Mean - 5 * E_0SD, E_0Mean + 5 * E_0SD,E_0Mean, E_0SD, False)
    k_0Bins = lambda n: gt.unif_spaced_param(n, -10, 10, k_0Mean, k_0SD, True)

    baseData["bins"] =  gt.product_grid(E_0Bins, numPts, k_0Bins, numPts)

def setupForEqSpProb(numPts):
    E_0Bins = lambda n: gt.unif_spaced_prob(n, E_0Mean, E_0SD, False)
    k_0Bins = lambda n: gt.unif_spaced_prob(n, k_0Mean, k_0SD, True)

    baseData["bins"] =  gt.product_grid(E_0Bins, numPts, k_0Bins, numPts)

def setupForLegGaussParam(numPts):
    E_0Bins = lambda n: gt.leggauss_param(n, E_0Mean -5 * E_0SD, E_0Mean + 5 * E_0SD, E_0Mean, E_0SD, False)
    k_0Bins = lambda n: gt.leggauss_param(n,-10, 10, k_0Mean, k_0SD, True)

    baseData["bins"] =  gt.product_grid(E_0Bins, numPts, k_0Bins, numPts)

def setupForLegGaussProb(numPts):
    E_0Bins = lambda n: gt.leggauss_prob(n, E_0Mean, E_0SD, False)
    k_0Bins = lambda n: gt.leggauss_prob(n, k_0Mean, k_0SD, True)

    baseData["bins"] =  gt.product_grid(E_0Bins, numPts, k_0Bins, numPts)

def setupForHermGauss(numPts):
    E_0Bins = lambda n: gt.hermgauss_param(n, E_0Mean, E_0SD, False)
    k_0Bins = lambda n: gt.hermgauss_param(n, k_0Mean, k_0SD, True)

    baseData["bins"] =  gt.product_grid(E_0Bins, numPts, k_0Bins, numPts)

def l2Norm(a):
    return np.sqrt(np.sum(np.square(a)))

def genFileName(name):
    return io.get_file_resource_path("dispersion/" + name + ".npz")

def trim_data(data, trim_amount):
    return data[trim_amount:-trim_amount]

#names = ["EqSpParam", "EqSpProb", "LegGaussParam", "LegGaussProb", "HermGauss"]
names = ["HermGauss"]
setupFunctions = {"EqSpParam" : setupForEqSpParam, "EqSpProb" : setupForEqSpProb, "LegGaussParam" : setupForLegGaussParam, "LegGaussProb" : setupForLegGaussProb, "HermGauss" : setupForHermGauss}

# Load benchmark data, or generate it

simFileName = genFileName("benchmark")
if os.path.exists(simFileName):
    t, IHR = io.read_time_current_data_bin(simFileName)
    time_step = t[1] - t[0]
    num_time_pts = len(t)
else:
    t = np.linspace(0, endTime, numPts)
    time_step = t[1] - t[0]
    num_time_pts = len(t)
    setupForHermGauss(benchmarkNumEvals)
    IHR, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
    del _
    io.write_time_current_bin_cmp(simFileName, t, IHR)
harmHR = {}
harmNorm = {}
for harm_num, harmonic in zip(HARMONIC_RANGE, 
                              st.extract_harmonic(HARMONIC_RANGE, 
                                                  freq * endTime, IHR)):
    harmHR[harm_num] = trim_data(harmonic, trim)
    harmNorm[harm_num] = l2Norm(harmHR[harmonic])
print "Benchmark data loaded"

for name in names:
    print "Beginning processing for {0}".format(name)
    harmErr = {}
    for harmonic in HARMONIC_RANGE:
        harmErr[harmonic] = []

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
        for harm_num, harmonic in zip(HARMONIC_RANGE,
                                      st.extract_harmonic(HARMONIC_RANGE,
                                                          freq * endTime, I)):
            harmonic = trim_data(harmonic, trim)
            harmErr[harm_num].append(l2Norm(harmHR[harm_num] - harmonic) / harmNorm[harm_num])
        print "Data for {0} samples loaded".format(numSamples)

    plt.title("Convergence harmonics for {0}".format(name))
    for harmonic in HARMONIC_RANGE:
        plt.loglog(numEvaluations, harmErr[harmonic], label="Harmonic {0}".format(harmonic))
    plt.legend()
    plt.show()
    #plt.savefig(io.get_file_resource_path("convPlots/harm10Conv{0}.pdf".format(name)))
    #plt.close()
