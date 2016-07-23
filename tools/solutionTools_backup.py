import numpy as np
import tools.conversion as conv
import scipy.signal
from scipy.optimize import brentq
import inspect
import scipy.stats.distributions
import numpy.polynomial.legendre
from scipy.weave import inline, blitz

solverFunctions = {}
binFunctions ={}

#TODO: Put C code in a separate file to tidy up.

fDef = """
// Python discrep
#define MAX_ITER 10000
#define NEWTONTOL 1e-5
#define TOL  1.48e-8
#define FTOL  1e-8

#define MIN(a,b) (a<b)?(a):(b)
#define MAX(a,b) (a>b)?(a):(b)

double discrep(double overPotNext, double rhof, double INext, double ICurr, double gammaf, double gamma1f, double gamma2f, double gamma3f, double rhoDivhf, double hf, int i, double* _theta, double dEpsNext, double kappaf) {
	double eta = overPotNext - rhof * INext;
	double cap = gammaf * (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
	double expon = exp(0.5 * eta);
	double kRed = kappaf / expon;
	double kOx = kappaf * expon;
	_theta[i+1] = (_theta[i] + hf * kOx) / (1 + hf*(kOx+kRed));
	return cap * (dEpsNext - rhoDivhf * (INext - ICurr)) + (1.0-_theta[i+1])*kOx - _theta[i+1] * kRed - INext;
}

double discrepStep(double overPotNext, double rhof, double INext, double ICurr, double gammaf, double gamma1f, double gamma2f, double gamma3f, double rhoDivhf, double hf, int i, double* _theta, double dEpsNext, double kappaf, double* fVal) {
	double eta = overPotNext - rhof * INext;
	double cap = gammaf * (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
	double expon = exp(0.5 * eta);
	double kRed = kappaf / expon;
	double kOx = kappaf * expon;
	double denom = (1 + hf*(kOx+kRed));
	_theta[i+1] = (_theta[i] + hf * kOx) / denom; 
	double dThetaNext = hf * (-kOx - _theta[i+1] * (kRed - kOx)) / denom; 
	*fVal = (cap * (dEpsNext - rhoDivhf * (INext - ICurr)) + (1.0-_theta[i+1])*kOx - _theta[i+1] * kRed - INext);

	double dCap = gammaf * (gamma1f + eta * (2 * gamma2f + eta * 3 * gamma3f));
	double fPrime = -cap * rhoDivhf + dCap * (dEpsNext - rhoDivhf * (INext - ICurr)) - rhof * 0.5 * (dThetaNext * (kOx + kRed) - (_theta[i+1] - 1) * kOx + _theta[i+1] * kRed) - 1;
	return -(*fVal) / fPrime;
}

"""

expr = """ // Python expr
int i;
double t = tau0;
double op, dOP;
return_val = 0;
int numIter;

for(i = 0; i < n-1; i++) {
	t += hf;
	if(t < revTau) {
		op = t + dEps * sin(omega * (t - tau0)) - eps_0;
		dOP = 1 + omega * dEps * cos(omega * (t-tau0));
	} else {
		op = 2 * revTau - t + dEps * sin(omega * (t - tau0)) - eps_0;
		dOP = -1 + omega * dEps * cos(omega * (t - tau0));
	}
	// This choice of initial conditions gives good performance for a wide range of k_0, E_0 values.
	double I1, I0 = I[i];
	if(I0 < 0) {
		I1 = I0 * (1+1e-4) + 1;
	} else {
		I1 = I0 * (1+1e-4) - 1;
	}
	double f1, f0 = discrep(op, rhof, I0, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);	

	
	f1 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);

	int bracketed = 0;
	double INext, fNext;
	for(numIter = 0; numIter < MAX_ITER; numIter++) {
		if((f1 > 0 && f0 > 0) || (f1< 0 && f0<0)) { //We haven't bounded a root.
			// Modified secant algorithm designed to overshoot such that f1 ~ -0.01 * f0 (with equality for linear functions).
			INext = ((f1 + 0.01*f0) * I0 - 1.01* I1 * f0) / (f1 - f0); 
			I0 = I1;
			f0 = f1;
			I1 = INext;
			f1 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
		} else { //We've bracketed a root!
			bracketed = 1;
			if(f1 < 0) { // From here on, we want f(I0) < 0 < f(I1)
				double tmp = I1;
				I1 = I0;
				I0 = tmp;
				tmp = f1;
				f1 = f0;
				f0 = tmp;
			}
			// Continue looping, but use bisection. 
			int useNext = 1;
			INext = (I1 + I0) * 0.5;
			for(;numIter < MAX_ITER;numIter++) { 
				fNext = discrep(op, rhof, INext, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);
				if(fNext < 0) {
					I0 = INext;
				} else {
					I1 = INext;
				}
				if(fabs(I1-I0) < TOL) { 
					I[i+1] = (I1+I0) * 0.5;
					break; // Root-finding done.
				}
				INext = (I1 + I0) * 0.5;
			}
			break; //One way or another, we've finished.
		}
	}
	if(numIter == MAX_ITER) {
		I[0] = I0;
		I[1] = I1;
		I[2] = f0;
		I[3] = f1;
		if(bracketed) {
			return_val = 1;
		} else {
			return_val = -1;
		}
		break; //Big loop
	}
}
"""


def solveI(tau, eps_0, dEps, omega, kappa, rho, gamma, gamma1, gamma2, gamma3,
	revTau, **kwargs):
	"""Solves for current using backwards Euler and returns the result as 
	a numpy array."""
	# Fixing alpha = 0.5 makes the analytic form, numerical evaluation nicer.
	n = len(tau)
	I = np.empty(n)
	theta = np.empty(n)
	I[0] = 0.0
	theta[0] = 0.0
	tau0 = float(tau[0]);
	h = float(tau[1] - tau[0]) #Assume equally spaced time points
	gammaf = float(gamma)
	gamma1f = float(gamma1)
	gamma2f = float(gamma2)
	gamma3f = float(gamma3)
	rhof = float(rho)
	rhoDivhf = float(rhof/h)
	hf = float(h)
	omega = float(omega)
	dEps = float(dEps)
	kappaf = float(kappa)
	revTau = float(revTau)
	eps_0 = float(eps_0)

	retFlag = inline(expr, ['n', 'rhof', 'theta', 'I', 'gammaf', 'gamma1f', 'gamma2f', 'gamma3f', 'rhoDivhf', 'hf', 'omega', 'dEps', 'kappaf', 'revTau', 'eps_0', 'tau0'], support_code=fDef)

	if retFlag == -1:
		msg = "Failed to converge; did not bound a root.  Values were (%10f, %.10f) and (%10f, %.10f)." %(I[0], I[2], I[1], I[3])
		raise RuntimeError(msg)
	elif retFlag > 0:
		msg = "Failed to converge; did bound a root. Values were (%10f, %.10f) and (%10f, %.10f)" %(I[0], I[2], I[1], I[3])
		raise RuntimeError(msg)
	return I,theta


solverFunctions["nondimensional"] = solveI

def solveIFromJSON(t, jsonData):
	"""Solves for the current and amount of species A using backwards Euler and returns the results as numpy arrays.

	This method is a wrapper around solveI and solveiDimensional which takes a dictionary (perhaps read from a json file) as input."""
	if not isinstance(jsonData, dict):
		raise ValueError("Expected a dictionary")

	try:
		typeVal = jsonData['type']
	except KeyError:
		raise ValueError("Expected jsonData to include a key 'type'")

	if typeVal not in  solverFunctions:
		raise ValueError("Unknown type {0}".format(jsonData["type"]))
	else:
		solver = solverFunctions[typeVal]
		return solver(t, **jsonData)

	
def solveIDimensional(t, E_0, dE, freq, k_0, Ru, Cdl, Cdl1, Cdl2, Cdl3, 
EStart, ERev, temp, nu, area, coverage, reverse, **kwargs):
	"""Solves for current using backwards Euler and returns the result as a numpy array.

	This method is a wrapper around the nondimensional solveI method."""
	epsStart = conv.nondimPot(temp, EStart)
	epsRev = conv.nondimPot(temp, ERev)
	dEps = conv.nondimPot(temp, dE)
	eps_0 = conv.nondimPot(temp, E_0)

	omega = conv.freqToNondimOmega(temp, nu, freq)

	kappa = conv.nondimRate(temp, nu, k_0)
	
	tau = conv.timeToNondimVoltage(temp, nu, EStart, ERev, t, reverse)	

	rho = conv.nondimResistance(temp, nu, area, coverage, Ru)
	
	# Note: the parameters Cdl1...3 are already nondimensional and should
	# NOT be multiplied by the scale potential E0
	E0 = conv.potScale(temp)
	gamma =  conv.nondimCapacitance(temp, nu, area, coverage, Cdl)
	gamma1 = Cdl1 
	gamma2 = Cdl2 
	gamma3 = Cdl3 

	i, theta = solveI(tau, eps_0, dEps, omega, kappa, rho, gamma, gamma1, gamma2, gamma3, epsRev)

	I = conv.dimCurrent(temp, nu, area, coverage, i)
	amtCovered = theta * area * coverage

	return I, amtCovered
solverFunctions["dimensional"] = solveIDimensional


def solveIWithDispersionDimensionalBins(t, dE, freq, Ru, Cdl, Cdl1, Cdl2, Cdl3,
EStart, ERev, temp, nu, area, coverage, bins, reverse, **kwargs):
	"""Solves for current when there is dispersion.
	
	The variable bins should be a list of tuples of the form (E_0Val, K_0Val, weight)."""
	IAvg = np.zeros(len(t))
	amtAvg = np.zeros(len(t))
	for vals in bins:
			E_0Val = vals[0]
			k_0Val = vals[1]
			weight = vals[2]
			I, amt = solveIDimensional(t, E_0Val, dE, freq, k_0Val,
			Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, 
			nu, area, coverage, reverse)
			IAvg += weight * I
			amtAvg += weight * amt
	return IAvg, amtAvg
solverFunctions["disp-dimensional-bins"] = solveIWithDispersionDimensionalBins

def widthAtHalfMaximum(I, time, nu, tRevIndex=None):
	"""Returns the width at half-maximum for a current I driven by a dc current with ramp nu."""
	if tRevIndex is None:
		tRev = len(time)-1

	maxI = -np.inf
	maxInd = 0
	for ind,IVal in zip(range(len(I)), I):
		if IVal > maxI:
			maxI = IVal
			maxInd = ind
		#Search only until current reverses
		if time[ind] > tRev:
			break
	
	halfMax = maxI * 0.5;
	# Binary search for location of left half-maximum
	lower = 0
	upper = maxInd
	while upper - lower > 1:
		midpt = (lower + upper)/2
		if I[midpt] > halfMax:
			upper = midpt
		elif I[midpt] < halfMax:
			lower = midpt
		else: #Exact equality
			lower = midpt
			upper = midpt
	if maxI - I[lower] > I[upper] - maxI:
		left = upper
	else:
		left = lower
	# Binary search for location of right half-maximum
	lower = maxInd
	upper = tRev
	while upper - lower > 1:
		midpt = (lower + upper)/2
		if I[midpt] > halfMax:
			lower = midpt
		elif I[midpt] < halfMax:
			upper = midpt
		else: #Exact equality
			lower = midpt
			upper = midpt
	if maxI - I[upper] < I[lower] - maxI:
		right = upper
	else:
		right = lower

	return (time[right] - time[left]) * nu

def shortCenteredKaiserWindow(halflength, center, N):
	"""Returns a Kaiser window of given half-length and center."""
	window = np.zeros(N)
	if center < 0 or center > N:
		raise ValueError("Center {0} is outside the range of values [0,{1}]".format(center, N))
	windowLeft = int(np.floor(center - halflength))
	windowRight = int(np.ceil(center + halflength))

	if windowLeft >= 0:
		windowLeftTrunc = windowLeft
		leftTrunc = 0
	else:
		windowLeftTrunc = 0
		leftTrunc = windowLeftTrunc - windowLeft

	if windowRight < N:
		windowRightTrunc = windowRight
		rightTrunc = leftTrunc + windowRightTrunc - windowLeftTrunc + 1 
	else:
		windowRightTrunc = N - 1
		rightTrunc = leftTrunc + windowRightTrunc - windowLeftTrunc + 1 
	window[windowLeftTrunc:windowRightTrunc+1] = scipy.signal.kaiser(windowRight-windowLeft+1, 14)[leftTrunc:rightTrunc]
	return window

def extractHarmonic(n, freq, data):
	"""Returns the nth harmonic of a given frequency extracted from real data.

	Note that a Kaiser window is used to extract the harmonic."""
	fourier = np.fft.rfft(data)
	windowHW = 0.75 * freq
	window = shortCenteredKaiserWindow(windowHW, n*freq, len(fourier))
	return np.fft.irfft(fourier * window)

def solveIDimensionalMC(t, numRuns, dE, freq, Ru, Cdl, Cdl1, Cdl2, Cdl3,
EStart, ERev, temp, nu, area, coverage, E_0Mean, E_0SD, k_0Mean, k_0SD, reverse, **kwargs):
	if E_0SD == 0:
		E_0Vals = np.repeat(E_0Mean, numRuns)
	else:
		E_0Vals = np.random.normal(E_0Mean, E_0SD, numRuns)
	if k_0SD == 0:
		k_0Vals = np.repeat(k_0Mean, numRuns)
	else:
		k_0Vals = k_0Mean * np.power(2,np.random.normal(0, k_0SD, numRuns))

	def solveIInner(E_0, k_0,):
		# Use the args passed to the function.
		return solveIDimensional(t, E_0, dE, freq, k_0, Ru, Cdl, Cdl1, Cdl2, Cdl3, 
		EStart, ERev, temp, nu, area, coverage, reverse)
	IAgg, amtAgg = reduce(lambda a,b: (a[0]+b[0], a[1]+b[1]), map(solveIInner, E_0Vals, k_0Vals))
	return IAgg/numRuns, amtAgg/numRuns
solverFunctions["disp-dimensional-MC"] = solveIDimensionalMC

def getSolverNames():
	return solverFunctions.keys()
