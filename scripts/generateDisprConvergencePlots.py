#!/bin/python
# Creates a plot of the convergence of various methods of calculating current
# with kinetic and thermodynamic dispersion.

#Add top level package to python path
import getTopLevel
import pickle
import tools.io as io
import tools.solutionTools as st
import tools.gridTools as gt
import numpy as np
import os.path
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
import time

PTS_PER_WAVE = 200

ERRRELSQR = np.square(0.01)

MAX_HARM = 12

fileName = getTopLevel.makePathFromTop("files/simulationParameters.json")
dataName = "Dispersion points plot"
baseData = io.readParametersFromJSON(fileName, dataName)
# Remove E_0, k_0 from the data and store them as means for the distributions.
E_0Mean = baseData.pop("eq_pot", None)
E_0SDs = np.linspace(0, 15e-2, 16)
k_0Mean = baseData.pop("eq_rate", None)
# Choose the shape parameter for the log-normal distribution underlying k_0 to
# give a desired coefficient of variation.
k_0CVs = np.linspace(0, 7, 15)
k_0SDs = np.sqrt(np.log(1 + np.square(k_0CVs)))
freq = baseData["freq"]

endTime = (baseData["pot_rev"] - baseData["pot_start"]) / baseData["nu"]
num_time_pts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
time_step = endTime / num_time_pts
trim = int(np.floor(num_time_pts / 100))
t = np.linspace(0, endTime, num_time_pts)

E_0BinFunFactory = lambda E_0SD: lambda n: gt.hermgaussParam(n, E_0Mean,
                                                           E_0SD, False)
k_0BinFunFactory = lambda k_0SD: lambda n: gt.hermgaussParam(n, k_0Mean,
                                                            k_0SD, True)

baseLineNumPts = 50

dataPickleFile = getTopLevel.makePathFromTop("files/dispersionPlotsData.dat")
benchmark = None
benchmarkHarmonics = None

E_0MinReqdPts = {}
k_0MinReqdPts = {}

def l2Norm(a):
	return np.sum(np.square(a))

def getData(E_0SD, k_0SD, num_time_ptsE_0, num_time_ptsk_0):
    E_0BinFun = E_0BinFunFactory(E_0SD)
    k_0BinFun = k_0BinFunFactory(k_0SD)
    baseData["bins"] = gt.productGrid(E_0BinFun, num_time_ptsE_0, k_0BinFun, num_time_ptsk_0)
    I, _ = st.solve_reaction_from_json(time_step, num_time_pts, baseData)
    return I

def computeBenchmarkAndHarmonics(E_0SD, k_0SD):
    global benchmark
    global benchmarkHarmonics
    benchmark = getData(E_0SD, k_0SD, baseLineNumPts, baseLineNumPts)
    benchmarkHarmonics = []
    for h in range(1, MAX_HARM+1):
        benchmarkHarmonics.append(st.extract_harmonic(h, freq * endTime, benchmark))

def closeToBenchmark(Itest):
    if l2Norm(Itest - benchmark) >= ERRRELSQR * l2Norm(benchmark):
        return False
    for h in range(1, MAX_HARM+1):
        harm = st.extract_harmonic(h, freq * endTime, Itest)
        if l2Norm(harm - benchmarkHarmonics[h-1]) >= \
           ERRRELSQR * l2Norm(benchmarkHarmonics[h-1]):
            return False
    return True

if not os.path.exists(dataPickleFile):
    data = {}
else:
    with open(dataPickleFile) as f:
        data = pickle.load(f)
for E_0SD in E_0SDs:
    for k_0SD in k_0SDs:
        if (E_0SD, k_0SD) in data:
            print "Data exists of E_0SD = {0}, k_0SD = {1}".format(E_0SD, k_0SD)
            ptsE, ptsK = data[(E_0SD, k_0SD)]
            E_0MinReqdPts[E_0SD] = ptsE
            k_0MinReqdPts[k_0SD] = ptsK
        else:
            print "Beginning processing for E_0SD = {0}, k_0SD = {1}".format(E_0SD, k_0SD)
            tStart = time.time()
            computeBenchmarkAndHarmonics(E_0SD, k_0SD)
            def bisect2d(nle, nue, nlk, nuk, best):
                """Algorithm adapted from:
                http://articles.leetcode.com/searching-2d-sorted-matrix-part-ii
                """
                print "bisect2d({0}, {1}, {2}, {3}, {4})".format(nle, nue, nlk, nuk, best)
                if best[0] * best[1] <= nle * nlk:
                    # No point in continuing, we can't find a better solution
                    return best
                if nle > nue or nlk > nuk:
                    # No points satisfy this condition.
                    return best
                # Trim down the upper limits if they are high enough that
                # we can't improve on best
                if nle * nuk > best[0] * best[1]:
                    return bisect2d(nle, nue, nlk,
                                   (best[0] * best[1]) / nle, best)
                elif nue * nlk > best[0] * best[1]:
                    return bisect2d(nle, (best[0] * best[1]) / nlk,
                                    nlk, nuk, best)

                if nle == nue and nlk == nuk:
                    #Previous check confirms that this is better than best,
                    #if it is a solution
                    sol = getData(E_0SD, k_0SD, nue, nuk)
                    if closeToBenchmark(sol):
                       best = (nue, nuk)
                       return best
                if nue - nle >= nuk - nlk:
                    nre = (nue + nle) / 2
                    # Run a binary search on num_time_ptsE_0 = nre
                    nukBin = nuk
                    nlkBin = nlk
                    while nukBin - nlkBin > 1:
                        nrkBin = (nukBin + nlkBin) / 2
                        nrkSol = getData(E_0SD, k_0SD, nre, nrkBin)
                        if closeToBenchmark(nrkSol):
                            nukBin = nrkBin
                        else:
                            nlkBin = nrkBin
                    # We must check that nlkBin does not also satisfy the
                    # condition.

                    if best[0] * best[1] > nre * nlkBin and \
                     closeToBenchmark(getData(E_0SD, k_0SD, nre, nlkBin)):
                        best = (nre, nlkBin)
                    elif best[0] * best[1] > nre * nukBin and \
                     closeToBenchmark(getData(E_0SD, k_0SD, nre, nukBin)):
                        best = (nre, nukBin)
                    best = bisect2d(nle, nre - 1, nlkBin, nuk, best)
                    best = bisect2d(nre + 1, nue, nlk, nukBin, best)
                else:
                    nrk = (nlk + nuk)/2
                    # Run a binary search on num_time_ptsk_0 = nrk
                    nueBin = nue
                    nleBin = nle
                    nreBin = nleBin
                    while nueBin - nleBin > 1:
                        nreBin = (nueBin + nleBin) / 2
                        nreSol = getData(E_0SD, k_0SD, nreBin, nrk)
                        if closeToBenchmark(nreSol):
                            nueBin = nreBin
                        else:
                            nleBin = nreBin
                    # We must check that nleBin does not also satisfy the
                    # condition.
                    if best[0] * best[1] > nleBin * nrk and \
                     closeToBenchmark(getData(E_0SD, k_0SD, nleBin, nrk)):
                        best = (nleBin, nrk)
                    elif best[0] * best[1] > nueBin * nrk and \
                      closeToBenchmark(getData(E_0SD, k_0SD, nueBin, nrk)):
                        best = (nueBin, nrk)
                    best = bisect2d(nle, nueBin, nrk + 1, nuk, best)
                    best = bisect2d(nleBin, nue, nlk, nrk - 1, best)

                return best
            ptsE, ptsK = bisect2d(E_0MinReqdPts.get(E_0SD, 1), baseLineNumPts,
                                  k_0MinReqdPts.get(k_0SD, 1), baseLineNumPts,
                                  (baseLineNumPts, baseLineNumPts))
            assert closeToBenchmark(getData(E_0SD, k_0SD, ptsE, ptsK))
            E_0MinReqdPts[E_0SD] = ptsE
            k_0MinReqdPts[k_0SD] = ptsK
            data[(E_0SD, k_0SD)] = (ptsE, ptsK)
            with open(dataPickleFile, 'w+') as f:
                pickle.dump(data, f)
            tEnd = time.time()
            print "Done with processing for E_0SD = " + str(E_0SD) + " k_0SD = " + str(k_0SD)
            print "Result was {0}".format(data[(E_0SD, k_0SD)])
            print "Processing took {0} seconds.".format(tEnd - tStart)


# Now, do the plotting.

figureE0Only = plt.figure(1)
num_time_ptsE0Only = [data[(E_0SD, 0.0)][0] for E_0SD in E_0SDs]
plt.plot(E_0SDs, num_time_ptsE0Only)
plt.xlabel('E_0 dispersion (V)')
plt.ylabel('Number of points')
title = 'Points required to achieve an error of < 1% with only E_0 dispersion'
plt.title(title)

figurek0Only = plt.figure(2)
num_time_ptsk0Only = [data[(0.0, k_0SD)][1] for k_0SD in k_0SDs]
plt.plot(k_0CVs, num_time_ptsk0Only)
plt.xlabel('k_0 dispersion (CV)')
plt.ylabel('Number of points')
title = 'Points required to achieve an error of < 1% with only k_0 dispersion'
plt.title(title)

figureBoth = plt.figure(3)
# Reshape data into an array
dataArray = []
for k_0SD, k_0CV in zip(k_0SDs, k_0CVs):
    row = []
    for E_0SD in E_0SDs:
        dataVal = data[E_0SD, k_0SD]
        row.append(dataVal[0] * dataVal[1])
    dataArray.append(row)

dataArray = np.array(dataArray)
ptsMin = 0
ptsMax = np.max(dataArray)
Em, km = np.meshgrid(E_0SDs, k_0CVs)
#TODO: Need to see if this produces the desired output.
plt.pcolormesh(Em, km, dataArray, cmap="Oranges", vmin = ptsMin, vmax = ptsMax)
plt.title("Number of points required at difference kinetic/thermodynamic dispersions")
plt.axis([Em.min(), Em.max(), km.min(), km.max()])
plt.xlabel("E_0 dispersion (V)")
plt.ylabel("k_0 dispersion (CV)")
plt.colorbar()


# Uncomment to plot
plt.show(block=True)
