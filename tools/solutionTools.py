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

ftol = float(3e-8); 
maxIter = int(50);

#TODO: Put C code in a separate file to tidy up.

fDef = """
// Python discrep

struct Params {
	double overPotNext;
	double rhof;
	double INext;
	double* ICurr;
	double gammaf;
	double gamma1f;
	double gamma2f;
	double gamma3f;
	double rhoDivhf;
	double hf;
	double* thetaCurr;
	double dEpsNext;
	double kappaf;
};

double discrep(double overPotNext, double rhof, double INext, double ICurr, double gammaf, double gamma1f, double gamma2f, double gamma3f, double rhoDivhf, double hf, int i, double* _theta, double dEpsNext, double kappaf) {
	double eta = overPotNext - rhof * INext;
	double cap = gammaf * (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
	double expon = exp(0.5 * eta);
	double kRed = kappaf / expon;
	double kOx = kappaf * expon;
	_theta[i+1] = (_theta[i] + hf * kOx) / (1 + hf*(kOx+kRed));
	return cap * (dEpsNext - rhoDivhf * (INext - ICurr)) + (1.0-_theta[i+1])*kOx - _theta[i+1] * kRed - INext;
}

double discrepParam(double INext, void *params) {
	struct Params* p = (struct Params*)params;
	double eta = p->overPotNext - p->rhof * INext;
	double cap = p->gammaf * (1 + eta * (p->gamma1f + eta * (p->gamma2f + eta * p->gamma3f)));
	double expon = exp(0.5 * eta);
	double kRed = p->kappaf / expon;
	double kOx = p->kappaf * expon;
	*(p->thetaCurr + 1) = (*(p->thetaCurr) + p->hf * kOx) / (1 + p->hf*(kOx+kRed));
	return cap * (p->dEpsNext - p->rhoDivhf * (INext - *(p->ICurr))) + (1.0 - *(p->thetaCurr + 1) ) * kOx - *(p->thetaCurr + 1)  * kRed - INext;
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
	return (*fVal) / fPrime;
}

double discrepStepParam(double INext, double* fVal, void * args) {
	struct Params* p = (struct Params*)args;
	double eta = p->overPotNext - p->rhof * INext;
	double cap = p->gammaf * (1 + eta * (p->gamma1f + eta * (p->gamma2f + eta * p->gamma3f)));
	double expon = exp(0.5 * eta);
	double kRed = p->kappaf / expon;
	double kOx = p->kappaf * expon;
	double denom = (1 + p->hf*(kOx+kRed));
	double dThetaNext = p->hf * (-kOx - *(p->thetaCurr + 1) * (kRed - kOx)) / denom;
 
	*(p->thetaCurr + 1) = (*(p->thetaCurr) + p->hf * kOx) / (1 + p->hf*(kOx+kRed));
	*fVal = cap * (p->dEpsNext - p->rhoDivhf * (INext - *(p->ICurr))) + (1.0 - *(p->thetaCurr + 1) ) * kOx - *(p->thetaCurr + 1)  * kRed - INext;
	double dCap = p->gammaf * (p->gamma1f + eta * (2 * p->gamma2f + eta * 3 * p->gamma3f));
	double fPrime = -cap * p->rhoDivhf + dCap * (p->dEpsNext - p->rhoDivhf * (INext - *(p->ICurr))) - p->rhof * 0.5 * (dThetaNext * (kOx + kRed) - (*(p->thetaCurr + 1) - 1) * kOx + *(p->thetaCurr + 1) * kRed) - 1;
	return (*fVal) / fPrime;
}

"""

expr = """ // Python expr
int i; 
double t = tau0;
return_val = -1; //All good

if(fabs(rhof) > 1e-10) {
	// Set up the parameter struct.
	struct Params *parameters = (struct Params*)malloc(sizeof(struct Params));
	assert(parameters != NULL);

	parameters->rhof = rhof;
	parameters->gammaf = gammaf;
	parameters->gamma1f = gamma1f;
	parameters->gamma2f = gamma2f;
	parameters->gamma3f = gamma3f;
	parameters->rhoDivhf = rhof/hf;
	parameters->hf = hf;
	parameters->kappaf = kappaf;


	//Set up the I, theta pointers.
	parameters->thetaCurr = theta;
	parameters->ICurr = I;


	for(i = 0; i < n-1; i++) {
		t += hf;
		// Based on setting di/dtau = 0 and taking the Taylor series expansion of dTheta/dtau to third order.
		double IRadius = kappaf * hf; //Rough heuristic.
		if(t < revTau) {
			parameters->overPotNext = t + dEps * sin(omega * (t - tau0)) - eps_0;
			parameters->dEpsNext = 1 + omega * dEps * cos(omega * (t-tau0));
		} else {
			parameters->overPotNext = 2 * revTau - t + dEps * sin(omega * (t - tau0)) - eps_0;
			parameters->dEpsNext = -1 + omega * dEps * cos(omega * (t - tau0));
		}
 	
		I[i+1] = I[i];
		double IGuess = I[i] - IRadius;
	
		if(solverSecant(maxIter, (I+i+1), IGuess, &discrepParam, ftol, parameters) == -1) { //Failed to converge. . .
			return_val = i;
			break;
		}
		parameters->ICurr++;
		parameters->thetaCurr++;
	}
	free(parameters);
} else {
	double op, dOP;
	for(i = 0; i < n-1; i++) {
			t += hf;
			if(t < revTau) {
				op = t + dEps * sin(omega * (t - tau0)) - eps_0;
				dOP = 1 + omega * dEps * cos(omega * (t-tau0));
			} else {
				op = 2 * revTau - t + dEps * sin(omega * (t - tau0)) - eps_0;
				dOP = -1 + omega * dEps * cos(omega * (t - tau0));
			}
			double expon = exp(0.5 * op);
			double kOx = kappaf * expon;
			double kRed = kappaf / expon;
			theta[i+1] = (theta[i] + hf * kOx) / (1 + hf*(kRed + kOx));
			double dTheta = (1-theta[i+1]) * kOx - theta[i+1] * kRed;
			I[i+1] = gammaf * dOP + dTheta;
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
	
	retFlag = inline(expr, ['n', 'rhof', 'theta', 'I', 'gammaf', 'gamma1f', 
	'gamma2f', 'gamma3f', 'hf', 'omega', 'dEps', 'kappaf', 
	'revTau', 'eps_0', 'tau0', 'ftol', 'maxIter'], support_code=fDef, 
	headers=['"solvers.c"', "<stdlib.h>"], include_dirs=["/users/gavart/Private/python/electrochemistry/tools"], 
	define_macros=[("NEWTON_DEBUG", None)])

	if retFlag != -1:
		msg = "Failed to converge at %d.  Values were I[%d] = %10f, I[%d]=%.10f." %(retFlag+1, retFlag, I[retFlag], retFlag+1, I[retFlag+1])
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
