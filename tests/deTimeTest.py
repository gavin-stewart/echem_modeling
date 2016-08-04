# Time how long it takes to solve the DE for one value of E_0, k_0.

import tools.io as io
import tools.solutionTools as st
import numpy as np
import os.path
from scipy.stats.distributions import norm
import time
from profilehooks import profile
import matplotlib.pyplot as plt

PTS_PER_WAVE = 200

def timeit(func):
	def timed(*args, **kw):
		ts=time.time()
		result = func(*args, **kw)
		te = time.time()
		
		print '%r (%r, %r) %2.2f sec' % \
		 (func.__name__, args, kw, te-ts)
		return result
	return timed

fileName = "./files/simulationParameters.json"
dataName = "Martin's experiment"
baseData = io.readParametersFromJSON(fileName, dataName)

endTime = (baseData["ERev"] - baseData["EStart"]) /baseData["nu"]
numPts = int(np.ceil(PTS_PER_WAVE * baseData["freq"] * 2 * endTime))
trim = int(np.floor(numPts / 100))

t = np.linspace(0, endTime, numPts)

@profile
def solveODE(numRep = 100, plot = False):
	for i in xrange(numRep):
		_,__ = st.solveIFromJSON(t, baseData)
		if(i==0 and plot):
			plt.plot(t, _)
baseData["k_0"] = 1e11
baseData["E_0"] = -0.1
solveODE(100)
baseData["E_0"] = -0.2
solveODE(100)
baseData["E_0"] = -0.3
solveODE(100)
baseData["E_0"] = -0.4
solveODE(100)
baseData["E_0"] = -0.5
solveODE(100)
baseData["E_0"] = -0.6
solveODE(100)
baseData["E_0"] = -0.7
solveODE(100)
plt.show()

