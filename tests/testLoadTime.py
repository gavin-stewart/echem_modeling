"""Evaluates the time/memory trade-off associated with constructing/loading 50
current traces with random parameters.
"""

NUM_TRACE = 50
import time
import os.path, os, shutil
import tools.solution_tools as st
import tools.fileio as io

loadPath = './tests/files/temp/'

if not os.path.exists(loadPath):
	os.makedirs(loadPath)

if not os.path.exists(os.path.join(loadPath, 'compressed')):
	os.makedirs(os.path.join(loadPath, 'compressed'))


if not os.path.exists(os.path.join(loadPath, 'uncompressed')):
	os.makedirs(os.path.join(loadPath, 'uncompressed'))

def timeit(func):
	def timed(*args, **kwargs):
		ts = time.time()
		result = func(*args, **kwargs)
		te = time.time()
		
		print '%s (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te-ts)
	return timed

timeEnd = 2 * 0.75 / 27.94e-3

import numpy as np
def genRandParams():
	"""Return a random parameter set as a dictionary"""
	data = {"temp" : 293.15, "nu" : 27.94e-3, "area" : 0.03,
                 "coverage" : 1.26e-11, "reverse" : True, "pot_start" : -0.85,
                 "pot_rev" : -0.1, "freq" : 8.959, "ac_amplitude" : 150e-3,
                 "type" : "dimensional"}
	data["eq_pot"] = np.random.normal(loc=-0.4, scale = 1e-2)
	data["eq_rate"] = np.random.lognormal(mean=np.log(4000.0), sigma=2)
	data["resistance"] = np.random.lognormal(mean=np.log(100.0), sigma=1e-2)
	data["cdl"] = np.random.lognormal(mean=1e-3, sigma=1e-4)
	data["cdl1"] = np.random.normal(loc=5e-3, scale=1e-4)
	data["cdl2"] = np.random.normal(loc=2.5e-3, scale=5e-5)
	data["cdl3"] = np.random.normal(loc=1e-6, scale=1e-7)
	return data

@timeit
def generateFromScratch():
	for jsonData in data:
		I, _ = st.solve_reaction_from_json(time_step, num_time_pts, jsonData)
 

@timeit
def readFromFileCompressed():
	for i in range(NUM_TRACE):
		t,I = io.read_time_current_data_bin(os.path.join(loadPath, "compressed/data"+str(i)+".npz"))

@timeit
def readFromFileUncompressed():
	for i in range(NUM_TRACE):
		t,I = io.read_time_current_data_bin(os.path.join(loadPath, "uncompressed/data"+str(i)+".npz"))



# Initialize data to be a list of dictionaries.
data = []
t = np.linspace(0, timeEnd, int(np.ceil(timeEnd * 8.959 * 200)))
num_time_pts = int(np.ceil(timeEnd * 8.959 * 200))
time_step = timeEnd / (num_time_pts - 1)

for i in range(NUM_TRACE):
	dataCurr = genRandParams()
	data.append(dataCurr)
	I, _ = st.solve_reaction_from_json(time_step, num_time_pts, dataCurr)
	#Save compressed and uncompressed copies.
	io.write_time_current_bin_cmp(os.path.join(loadPath, "compressed/data"+str(i)+".npz"), t, I, overwrite=True)		
	io.write_time_current_data_bin(os.path.join(loadPath, "uncompressed/data"+str(i)+".npz"), t, I, overwrite=True)

generateFromScratch()

readFromFileUncompressed()

readFromFileCompressed()

# Get size of compressed, uncompressed files.
sizeCompressed = 0
sizeUncompressed = 0
for i in range(NUM_TRACE):
	sizeCompressed += os.path.getsize(os.path.join(loadPath, "compressed/data"+str(i)+".npz"))
	sizeUncompressed += os.path.getsize(os.path.join(loadPath, "uncompressed/data"+str(i)+".npz"))

print "Memory usage for uncompressed data was {0} bytes\nMemory usage for compressed data was {1} bytes".format(sizeUncompressed, sizeCompressed)

#Clean up
shutil.rmtree(loadPath)

