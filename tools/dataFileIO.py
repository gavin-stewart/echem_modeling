import numpy as np
import unittest
import os.path
import json

def readTimeCurrentData(fileName):
	"""Returns time and current data contained in the given file as two 
	numpy arrays.

	
	The function expects the file to contain two columns of data, separated 
	by whitespace.  The first column should contain time data, and the 
	second the current data.  It is up to the user to perform any necessary 
	unit conversions on these data."""

	# Redo to use preallocated arrays
	time = []
	current = []
	with open(fileName, 'r') as f:
		for line in f:
			split = line.split();
			if(len(split) != 2):
				raise ValueError("In file", fileName, 
				"the line:\n\t", line, "had ", len(split),
				 "whitespace separated columns (2 columns expected).")
			time.append(float(split[0]))
			current.append(float(split[1]))
	return np.array(time), np.array(current)

def writeTimeCurrentData(fileName, t, I, overwrite=False):
	"""Writes the given time and current data to the specified file in a human-readable format."""

	if len(t) != len(I):
		msg = "Expected time and current lists to have the same length.  Time length was {0} and current length was {1}.".format(len(t), len(I))
		raise ValueError(msg)

	if os.path.exists(fileName) and not overwrite:
		msg = "File exists and overwriting was not requested."
		raise ValueError(msg)

	with open(fileName, 'w') as f:
		for i in range(len(t)):
			f.write("{0}\t{1}\n".format(t[i], I[i]))

def writeTimeCurrentDataBinary(fileName, t, I, overwrite=False):
	"""Writes the given time and current data to the specified file in binary."""

	if len(t) != len(I):
		msg = "Expected time and current lists to have the same length.  Time length was {0} and current length was {1}.".format(len(t), len(I))
		raise ValueError(msg)

	if os.path.exists(fileName) and not overwrite:
		msg = "File exists and overwriting was not requested."
		raise ValueError(msg)

	data = { 'tStart' : t[0], 'tEnd' : t[-1], 'nt' : len(t), 'I' : I }
	np.savez_compressed(fileName, **data)

def readTimeCurrentDataBinary(fileName, overwrite=False):
	data = np.load(fileName)
	if 't' in data: #Old-style
		t = data['t']
	else: #New-style
		tStart = data['tStart']
		tEnd = data['tEnd']
		nt = data['nt']
		t = np.linspace(tStart, tEnd, nt)
	I = data['I']
	return t,I

def readParametersFromJSON(fileName, dataName, typeName=None):
	"""Returns a dictionary from the specified file named dataName of type 
	typeName if one is found, None otherwise."""
	if not os.path.exists(fileName):
		raise ValueError("File {0} does not exist.".format(fileName))
	with open(fileName, 'r') as f:
		loaded = json.load(f)
	for entry in loaded:
		if entry['name'] == dataName:
			if not typeName is None and entry['type'] != typeName:
				raise ValueError("Parameters {0} in file {1} are not nondimensional.  The parameters have type {2}".format(dataName, fileName, entry['type']))
			return entry
	return None


def readNondimensionalParametersFromJSON(fileName, dataName):
	"""Reads non-dimensionalized experiment parameters from the specified 
	datafile and returns a dictionary object if the experiment name was found
	or None otherwise."""
	return readParametersFromJSON(fileName, dataName, 'nondimensional')

def readDimensionalParametersFromJSON(fileName, dataName):
	"""Reads non-dimensionalized experiment parameters from the specified 
	datafile and returns a dictionary object if the experiment name was found
	or None otherwise."""
	return readParametersFromJSON(fileName, dataName, 'dimensional')

def writeParametersToJSON(fileName, parameters):
	"""Write experimental parameters to the specified datafiles.  If a set 
	of parameters of the same name already exists, it will be overwritten."""
	if not isinstance(parameters, dict):
		raise ValueError("Parameters must be a dictionary.")
	elif not ('name' in parameters and 'type' in parameters):
		raise ValueError("Parameters must contain keys 'name' and 'type'")

	if os.path.exists(fileName):
		data = json.load(fileName)
	else:
		data = []

	preexisting = False
	for n, entry in enumerate(data):
		if entry['name'] == parameters['name']:
			data[i] = parameters
			preexisting = True
			break
	if not preexisting:
		data.append(parameters)
	with open(fileName, "w") as f:
		json.dump(data, f)
			 

def writeNondimensionalParametersToJSON(fileName, dataName, eps_0, dEps, omega,
kappa, rho, gamma, gamma1, gamma2, gamma3, revTau, reverse):
	"""Write non-dimensionalized experiment parameters to the specified 
	datafile.  If experimental parameters of that name already exist, they 
	will be overwritten."""
	parameters = {'name' : dataName,'type' : 'nondimensional',\
	'eps_0': eps_0,'dEps' : dEps, 'omega' : omega, 'kappa' : kappa,\
	'rho' : rho, 'gamma' : gamma, 'gamma1' : gamma1,'gamma2' : gamma2,\
	'gamma3' : gamma3, 'revTau' : revTau, 'reverse' : reverse}
	writParametersToJSON(fileName, parameters)

def writeDimensionalParametersToJSON(fileName, dataName, E_0, dE, freq, k_0, 
Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, coverage, reverse):
	"""Write non-dimensionalized experiment parameters to the specified 
	datafile.  If experimental parameters of that name already exist, they 
	will be overwritten."""
	parameters = {'name' : dataName,'type' : 'dimensional',\
	'eps_0': eps_0,'dEps' : dEps, 'omega' : omega, 'kappa' : kappa,\
	'rho' : rho, 'Cdl' : Cdl, 'Cdl1' : Cdl1,'Cdl2' : Cdl2, 'Cdl3' : Cdl3,\
	'revTau' : revTau, 'reverse' : reverse}
	writParametersToJSON(fileName, parameters)


