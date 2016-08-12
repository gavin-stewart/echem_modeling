"""A set of tools for converting between dimensionalized andnondimensionalized
data for the Bulter-Volmer kinetic model.
"""

import numpy as np

FARADAY = 96485.33289 #C /mol.  Faraday's constant
R_GAS = 8.3144598 #J / mol / K. Ideal Gas constant

# Potential functions
def nondim_pot(temp, dim_pot_val):
    """Return a nomdimensional potential."""
    return dim_pot_val / scale_pot(temp)

def dim_pot(temp, nondim_pot_val):
    """Return a dimensional potential"""
    return nondim_pot_val * scale_pot(temp)

def scale_pot(temp):
    """The reference potential used for nondimensionalization."""
    return R_GAS * temp / FARADAY

# Time functions

def scale_time(temp, ramp_rate):
    """The reference length of time used for nondimensionalization."""
    return scale_pot(temp) / ramp_rate

def nondim_time(temp, ramp_rate, pot_start, time):
    """Return a nondimensional time."""
    return (time * ramp_rate + pot_start) / scale_pot(temp)

# Frequency functions
def freq_to_nondim_omega(freq):
    """Convert frequency to nondimensional phase velocity."""
    return 2*np.pi*freq

def dim_omega_to_freq(omega):
    """Convert dimensional phase velocity to frequency."""
    return omega/(2*np.pi)

def nondim_omega_to_freq(temp, ramp_rate, omega):
    """Convert nondimensional phase velocity to frequency."""
    return omega * scale_time(temp, ramp_rate) / (2*np.pi)

def nondim_omega(temp, ramp_rate, omega):
    """Return nondimensionalized phase velocity."""
    return omega * scale_time(temp, ramp_rate)

def dim_omega(temp, ramp_rate, omega):
    """Return dimensionalized phase velocity."""
    return omega / scale_time(temp, ramp_rate)

def freq_to_dim_omega(temp, ramp_rate, freq):
    """Convert frequency to dimensional phase velocity."""
    return 2*np.pi*freq * scale_time(temp, ramp_rate)

# Rate functions
def nondim_rate(temp, ramp_rate, k):
    """Nondimensionalize a reaction rate."""
    return k * scale_time(temp, ramp_rate)

def dim_rate(temp, ramp_rate, kappa):
    """Dimensionalize a reaction rate."""
    return kappa / scale_time(temp, ramp_rate)

#Resistance functions
def nondim_resistance(temp, ramp_rate, area, coverage, resistance):
    """Nondimensionalize resistance."""
    return resistance * scale_current(temp, ramp_rate, area, coverage)\
         / scale_pot(temp)

def dim_resistance(temp, ramp_rate, area, coverage, resistance):
    """Dimensionalize resistance."""
    return resistance * scale_pot(temp)\
         / scale_current(temp, ramp_rate, area, coverage)

#Capacitance functions

#Note that the factor of area enters the equations because experimental
#capacitances are often given per unit area, while the derived equation
#deals in the net capacitance of the system.
def nondim_capacitance(temp, ramp_rate, area, coverage, capacitance):
    """Nondimensionalize capacitance."""
    return capacitance * scale_pot(temp) * area \
           / (scale_current(temp, ramp_rate, area, coverage) \
           * scale_time(temp, ramp_rate))

def dim_capacitance(temp, ramp_rate, area, coverage, capacitance):
    """Dimensionalize capacitance."""
    return capacitance * scale_current(temp, ramp_rate, area, coverage)\
         * scale_time(temp, ramp_rate) / (scale_pot(temp) * area)

# Current functions
def nondim_current(temp, ramp_rate, area, coverage, current):
    """Nondimensionalize a current."""
    return current / scale_current(temp, ramp_rate, area, coverage)

def dim_current(temp, ramp_rate, area, coverage, current):
    """Dimensionalize a current."""
    return current * scale_current(temp, ramp_rate, area, coverage)

def scale_current(temp, ramp_rate, area, coverage):
    """Return the reference current used for nondimensionalization."""
    return FARADAY * area * coverage / scale_time(temp, ramp_rate)

# TODO: Unsupported below this line.
class ExperimentParameters(object):
    """Container class for parameters relating to the experimental setup."""

    def __init__(
            self, resistance, temperature, ramp_rate, pot_start, pot_rev, area,
            coverage):
        self.resistance = resistance
        self.temperature = temperature
        self.ramp_rate = ramp_rate
        self.pot_start = pot_start
        self.pot_rev = pot_rev
        self.area = area
        self.coverage = coverage

    def scale_pot(self):
        """Returns the reference potential."""
        return R_GAS * self.temperature / FARADAY

    def scale_time(self):
        """Returns the reference time length."""
        return self.scale_pot() / self.ramp_rate

    def scale_current(self):
        """Returns the reference current."""
        return FARADAY * self.area * self.coverage / self.scale_time()

class ACParameters(object):
    """Container class for parameters relating to the AC component of
    potential.
    """

    def __init__(self, frequency, ac_amplitude, dimensional=True):
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.ac_amplitude = ac_amplitude
        self.dimensional = dimensional

    def nondimensionalize(self, exp_params):
        """Returns a nondimensionalized set of AC parameters."""
        if self.dimensional:
            ref_pot = exp_params.scale_pot()
            ref_time = exp_params.scale_time()
            return ACParameters(self.frequency * ref_time,
                                self.ac_amplitude / ref_pot, False)
        else:
            return self

    def dimensionalize(self, exp_params):
        """Returns a dimensionalized set of AC parameters."""
        if self.dimensional:
            return self
        else:
            ref_pot, ref_time, _ = exp_params.scale_parameters()
            return ACParameters(self.frequency / ref_time,
                                self.ac_amplitude * ref_pot, True)

class ReactionParameters(object):
    """Container class for parameters governing the electrochemical reaction.
    """

    def __init__(self, eq_rate, eq_pot):
        self.eq_rate = eq_rate
        self.eq_pot = eq_pot



