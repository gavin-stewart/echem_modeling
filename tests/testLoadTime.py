"""
Evaluates the time/memory trade-off associated with constructing/loading 50 current traces with random parameters.
"""

NUM_TRACE = 50
import time
import getTopLevel
import os.path
import tools.solutionTools as st
import tools.dataFileIO as dfio

loadPath = getTopLevel.makeFilePath('loadTimeTest')


def timeit(func):
	def timed(*args, **kwargs):
		ts = time.time()
		result = func(*args, **kwargs)
		te = time.time()
		
		print '%f (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te-ts)
	return timed

timeEnd = 0.75 / 27.94e-3

import numpy as np
def genRandParams():
	"""Return a random parameter set as a dictionary"""
	data = {"temp" : 293.15, "nu" : 27.94e-3, "area" : 0.03, "coverage" : 1.26e-11, "reverse" : True, "EStart" : -0.85, "ERev" : -0.1, "freq" : 8.959, "de" : 150e-3}
	data["E_0"] = np.random.norm(loc=-0.4, scale = 1e-2)
	data["k_0"] = np.random.lognormal(mean=4000.0, sigma=2)
	data["Ru"] = np.random.lognormal(mean=100.0, sigma=1e-2)
	data["Cdl"] = np.random.lognormal(mean=1e-3, sigma=1e-4)
	data["Cdl1"] = np.random.normal(loc=5e-3, scale=1e-4)
	data["Cdl2"] = np.random.normal(loc=2.5e-3, scale=5e-5)
	data["Cdl3"] = np.random.normal(loc=1e-6, scale=1e-7)
	return data

@timeit
def generateFromScratch():
	for jsonData in data:
		I, _ = st.solveIFromJSON(t, jsonData)
 

@timeit
def readFromFileCompressed():
	for i in range(NUM_TRACE):
		t,I = dfio.readTimeCurrentDataBinary(os.path.join(loadPath, "data"+str(i)+".npz"))

@timeit
def readFromFileUncompressed():
	for i in range(NUM_TRACE):
		t,I = dfio.readTimeCurrentDataBinary(os.path.join(loadPath, "data"+str(i)+".npy"))



# Initialize data to be a list of dictionaries.
data = []
t = np.linspace(0, timeEnd, int(np.ceil(timeEnd * 8.959 * 200)))
for i in range(NUM_TRACE):
	dataCurr = getRandParams()
	data.append(dataCurr)
	I, _ = solveIFromJSON(t, dataCurr)
	#Save compressed and uncompressed copies.
	dfio.writeTimeCurrentDataBinaryCompressed(os.path.join(loadPath, "data"+str(i)+".npz", t, I))		
	dfio.writeTimeCurrentDataBinaryUncompressed(os.path.join(loadPath, "data"+str(i)+".npy", t, I))



