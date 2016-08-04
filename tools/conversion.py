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

# TODO: Unsupported below this line.
class ExperimentParameters(object):
    """Container class for parameters relating to the experimental setup."""

    def __init__(
          self, resistance, temperature, 
          ramp_rate, e_start, e_rev, area, coverage):
        self.resistance = resistance
        self.temperature = temperature
        self.ramp_rate = ramp_rate
        self.e_start = e_start
        self.e_reverse = e_reverse
        self.area = area
        self.coverage = coverage

    def scale_parameters(self):
     """Returns the scale parameters E_0, T_0, and I_0 for 
     nondimensionalization.
     """
     E_0 = R_GAS * self.temperature / FARADAY
     T_0 = E_0 / self.ramp_rate
     I_0 = FARADAY * self.area * self.coverage / T_0
     return E_0, T_0, I_0

    def pot_scale(self):
        """Returns the reference potential."""
        return R_GAS * self.temperature / FARADAY

    def time_scale(self):
        """Returns the reference time length."""
        return self.potScale() / self.ramp_rate

class ACParameters(object):
    """Container class for parameters relating to the AC component of 
    potential.
    """

    def __init__(self, frequency, amplitude, dimensional=True):
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.amplitude = amplitude
        self.dimensional = dimensional

    def nondimensionalize(self, exp_params):
        """Returns a nondimensionalized set of AC parameters."""
        if dimensional:
            E_0, T_0, _ = exp_params.scale_parameters()
            return ACParameters(self.freq * T_0, amplitude / E_0, False)
        else:
            return self

    def dimensionalize(self, exp_params):
        """Returns a dimensionalized set of AC parameters."""
        if dimensional:
            return self
        else:
            E_0, T_0, _ = exp_params.scale_parameters()
            return ACParameters(self.freq / T_0, amplitude * E_0, True)

class ReactionParameters(object):
    """Container class for parameters governing the electrochemical reaction.
    """

    def __init__(self, rate, half_cell_potential):
        self.rate = rate
        self.half_cell_potential = half_cell_potential



