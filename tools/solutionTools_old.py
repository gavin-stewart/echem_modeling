import numpy as np
import tools.conversion as conv
import scipy.signal
from scipy.optimize import brentq
import inspect
import scipy.stats.distributions
import numpy.polynomial.legendre
from scipy.weave import inline, blitz

MAX_ITER = 100
solverFunctions = {}
binFunctions ={}

#TODO: Put C code in a separate file to tidy up.

fDef = """
// Python discrep
int MAX_ITER_BISECT = 100;
int MAX_ITER_NEWTON = 100;
double TOL = 1.48e-8;
double FTOL = 1e-8;

//Using global vars for a performance gain.  discrep must be called before discrepDeriv in order for this to work.
double eta, expon, cap, denom, kRed, kOx, dResVolDrop;

#define MIN(a,b) (a<b)?(a):(b)
#define MAX(a,b) (a>b)?(a):(b)

double discrep(double overPotNext, double rhof, double INext, double ICurr, double gammaf, double gamma1f, double gamma2f, double gamma3f, double rhoDivhf, double hf, int i, double* _theta, double dEpsNext, double kappaf) {
	eta = overPotNext - rhof * INext;
	cap = gammaf * (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
	expon = exp(0.5 * eta);
	kRed = kappaf / expon;
	kOx = kappaf * expon;
	denom = (1 + hf*(kOx+kRed));
	_theta[i+1] = (_theta[i] + hf * kOx) / denom;
	dResVolDrop = rhoDivhf* (INext - ICurr);
	return cap * (dEpsNext - dResVolDrop) + (1.0 - _theta[i+1]) * kOx - _theta[i+1] * kRed - INext;
}

double discrepDeriv(double overPotNext, double rhof, double INext, double ICurr, double gammaf, double gamma1f, double gamma2f, double gamma3f, double rhoDivhf, double hf, int i, double* _theta, double dEpsNext, double kappaf) {
	double dCap = -rhof * gammaf * (gamma1f + eta * (2*gamma2f + eta * 3 * gamma3f));
	double dThetaNext = hf * rhof * 0.5 * (-kOx - (-kOx + kRed) * _theta[i+1]) / denom;
	return dCap * (dEpsNext - dResVolDrop) - cap * rhoDivhf + 0.5 * rhof * ((_theta[i+1] - 1.0)*kOx - _theta[i+1] * kRed) - dThetaNext * (kOx + kRed) - 1;
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
		op = t + dEps * sin(omega * (t-tau0)) - eps_0;
		dOP = 1 + omega * dEps * cos(omega * (t-tau0));
	} else {
		op = revTau - 2*t + dEps * sin(omega * (t-tau0)) - eps_0;
		dOP = -1 + omega* dEps * cos(omega * t);
	}

	double I1, I0 = I[i];

	double f1,f0 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
	double fPrime = discrepDeriv(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);

	I1 = I0 + f0/fPrime;
	I0 -= f0/fPrime; 

	for(numIter = 0; numIter < MAX_ITER; numIter++) {
		I1 -= f1 / fPrime;
		f1 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
		fPrime = discrepDeriv(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf); 
		if(fabs(f1) < FTOL) {
			I[i+1] = I1;
			break;
		}
	} 
	if(numIter == MAX_ITER) { //Try again, but safely.
		I0 = I[i] - 5*kappaf*hf; //Entirely hueristic bound
		f0 = discrep(op, rhof, I0, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);

		I1 = I[i] + 5*kappaf*hf;
		f1 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);
		fPrime = discrepDeriv(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);

		int lastTouched = 1;
		int bracketed = 0;
		for(numIter = 0; numIter < MAX_ITER; numIter++) {
			double INext;
			if(lastTouched) {
				INext = I1 - f1 / fPrime;
			} else {
				INext = I0 - f0 / fPrime;
			}
			if(!bracketed && ((f1 > 0 && f0 > 0) || (f1< 0 && f0<0))) { //We haven't bounded a root.
				I0 = I1;
				f0 = f1;
				I1 = INext;
				f1 = discrep(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
				if(fabs(f1) < FTOL) {
					I[i+1] = I1;
					break;
				}
				fPrime = discrepDeriv(op, rhof, I1, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
			} else { //We've bracketed a root!
				if(!bracketed) { //First time here?  Better record this and ensure f0 < 0 < f1 !
					bracketed = 1;
					if(f1 < 0) { // From here on, we want f(I0) < 0 < f(I1)
						double tmp = I1;
						I1 = I0;
						I0 = tmp;
						tmp = f1;
						f1 = f0;
						f0 = tmp;
						lastTouched = 0;
					} else {
						lastTouched = 1;
					}
				}
				
				if(INext <= MIN(I0, I1) || INext >= MAX(I0, I1)) { //We've bounded a root, but not made the bracket smaller! Better switch to bisection.
					INext = (I1 + I0) / 2;
				}
				double fNext = discrep(op, rhof, INext, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf,i, theta, dOP, kappaf);
				double fPrime = discrepDeriv(op, rhof, INext, I[i], gammaf, gamma1f, gamma2f, gamma3f, rhoDivhf, hf, i, theta, dOP, kappaf);
				if(fNext < 0) {
					I0 = INext;
					f0 = fNext;
					if(f0 > -FTOL) {
						I[i+1] = I0;
						break;
					}
					lastTouched = 0; 
				} else {
					I1 = INext;
					f1 = fNext;
					if(f1 < FTOL) {
						I[i+1] = I1;
						break;
					}
					lastTouched = 1;
				}
				if(fabs(I1-I0) < TOL) {
					I[i+1] = (I1 + I0) * 0.5;
					break; // Root-finding done.
				}
			}
	} 
	if(numIter == MAX_ITER) { //Even the safe method failed.
		I[0] = I0;
		I[1] = I1;
		if(!bracketed) {
			return_val = -1;
		} else {
			return_val = fabs(I1-I0);
		}
		break; //Big loop
	}
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
	dEps * np.sin(omega*(tau-tau[0])) - eps_0
	gammaf = float(gamma)
	gamma1f = float(gamma1)
	gamma2f = float(gamma2)
	gamma3f = float(gamma3)
	rhof = float(rho)
	rhoDivhf = float(rho/h)
	hf = float(h)
	omega = float(omega)
	dEps = float(dEps)
	kappaf = float(kappa)
	revTau = float(revTau)
	eps_0 = float(eps_0)

	retFlag = inline(expr, ['n', 'rhof', 'theta', 'I', 'gammaf', 'gamma1f', 'gamma2f', 'gamma3f', 'rhoDivhf', 'hf', 'omega', 'dEps', 'kappaf', 'revTau', 'eps_0', 'tau0'], support_code=fDef)

	if retFlag == -1:
		msg = "Failed to converge after %d iterations; did not bound a root.  Values were %.8f and %.8f." %(MAX_ITER, I[0], I[1])
		raise RuntimeError(msg)
	elif retFlag > 0:
		msg = "Failed to converge after %d iterations; but did bound a root.  Distance at end was %.16f." %(MAX_ITER, retFlag)
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
