# A set of tools for converting between dimensionalized andnondimensionalized
# data for the Bulter-Volmer kinetic model.

import numpy as np

FARADAY = 96485.33289 #C /mol.  Faraday's constant
R_GAS = 8.3144598 #J / mol / K. Ideal Gas constant

def scaleParameters(temp, nu, area, coverage):
	"""Returns the scale parameters E_0, T_0, and I_0 for nondimensionalization."""
	E_0 = R_GAS * temp / FARADAY
	T_0 = E_0 / nu
	I_0 = FARADAY * area * coverage / T_0
	return E_0, T_0, I_0

# Potential functions
def nondimPot(temp, E):
	return E / potScale(temp)

def dimPot(temp, eps):
	return eps * potScale(temp)

def potScale(temp):
	return R_GAS * temp / FARADAY

# Time functions

def timeScale(temp, nu):
	return potScale(temp) / nu

def timeToNondimVoltage(temp, nu, EStart, ERev,t, reverse=True):
	T0 = timeScale(temp, nu)
	E0 = potScale(temp)
	n = len(t)
	if reverse:
		return np.linspace(EStart/E0, EStart / E0 + 2*(ERev - EStart) / (nu * T0), n)
	else:
		return np.linspace(EStart/E0, EStart / E0 + (ERev - EStart) / (nu * T0), n)

# Frequency functions
def freqToDimOmega(freq):
	return 2*np.pi*freq

def dimOmegaToFreq(omega):
	return omega/(2*np.pi)

def nondimOmegaToFreq(temp, nu, omega):
	return omega * timeScale(temp, nu) / (2*np.pi)

def dimOmegaToNondimOmega(temp, nu, omega):
	return omega * timeScale(temp, nu)

def nondimOmegaToDimOmega(temp, nu, omega):
	return omega / timeScale(temp, nu)

def freqToNondimOmega(temp, nu, freq):
	return 2*np.pi*freq * timeScale(temp, nu)

# Rate functions
def nondimRate(temp, nu, k):
	return k * timeScale(temp, nu)

def dimRate(temp, nu, kappa):
	return kappa / timeScale(temp, nu)

#Resistance functions
def nondimResistance(temp, nu, area, coverage, R):
	return R * currentScale(temp, nu, area, coverage) / potScale(temp) 

def dimResistance(temp, nu, area, coverage, rho):
	return rho * potScale(temp) / currentScale(temp, nu, area, coverage)

#Capacitance functions

#Note that the factor of area enters the equations because experimental capacitances are often given
#per unit area, while the derived equation deals in the net capacitance of the system.
def nondimCapacitance(temp, nu, area, coverage, C):
	return C * potScale(temp) * area / (currentScale(temp, nu, area, coverage) * timeScale(temp, nu))

def dimCapacitance(temp, nu, area, coverage, gamma):
	return gamma * currentScale(temp, nu, area, coverage) * timeScale(temp, nu) / (potScale(temp) * area)

# Current functions
def nondimCurrent(temp, nu, area, coverage, I):
	return I / currentScale(temp, nu, area, coverage)

def dimCurrent(temp, nu, area, coverage, i):
	return i * currentScale(temp, nu, area, coverage)

def currentScale(temp, nu, area, coverage):
	return FARADAY * area * coverage / timeScale(temp, nu)

	
