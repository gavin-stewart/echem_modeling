"""A collection of functions for constructing quadrature grids."""
import scipy.stats.distributions
import numpy as np
import itertools

def unifSpacedParam(numSamples, startPoint, endPoint, loc, scale, exp, distr=scipy.stats.distributions.norm):
	if numSamples == 1:
		#Silly, but worth a special case
		return [(startPoint + endPoint)/2],[ 1.0]
	keys = np.linspace(startPoint, endPoint, numSamples).tolist()
	keys.insert(0, -np.inf)
	keys.append(np.inf)
	if exp:
		pt = loc * np.exp(keys[1:-1]) * exp(-np.square(scale) / 2)
	else:
		pt = keys[1:-1]
	weights = []
	for key, keyPrev, keyNext in zip(keys[1:-1], keys[:-2], keys[2:]):
		if not exp:
			weights.append(distr.cdf((key + keyNext)/2, loc, scale) -\
			distr.cdf((key+keyPrev)/2, loc, scale))
		else:
			weights.append(distr.cdf((key + keyNext)/2, 0, scale) -\
			distr.cdf((key+keyPrev)/2, 0, scale))

	return pt, weights

def unifSpacedProb(numSamples, loc, scale, exp, distr=scipy.stats.distributions.norm):
	if numSamples== 1:
		#Again, silly, but it must be handled separately.
		return [loc], [1.0]
	binEnds = np.linspace(0,1,numSamples+1)
	binCenters = 0.5 * (binEnds[1:] + binEnds[:-1])
	pts = []
	weights = itertools.repeat(1.0/float(numSamples), numSamples)
	for k in binCenters:
		if exp:
			pts.append(loc * np.expdistr.ppf(k, 0, scale))
		else:
			pts.append(distr.ppf(k, loc, scale))
	return pts, weights

def leggaussParam(numSamples, startPoint, endPoint, loc, scale, exp, distr=scipy.stats.distributions.norm):
	GLPts, GLweights = np.polynomial.legendre.leggauss(numSamples)
	rescaledPts = 0.5 * (GLPts + 1) * (endPoint-startPoint) + startPoint
	if exp:
		pts = loc * np.exp(rescaledPts)* exp(-np.square(scale) / 2)
	else:
		pts = rescaledPts
	weights = [] 
	for pt, weight in zip(rescaledPts, GLweights):
		if exp:
			weights.append(weight * distr.pdf(pt, 0, scale))
		else:
			weights.append(weight * distr.pdf(pt, loc, scale))
	return pts, weights

def leggaussProb(numSamples, loc, scale, exp, distr=scipy.stats.distributions.norm):
	GLPts, GLWeights = np.polynomial.legendre.leggauss (numSamples)
	rescaled = (GLPts + 1) * 0.5
	pts = []
	weights = []
	for pt,weight in zip(rescaled, GLWeights):
		if exp:
			paramPt = loc * np.exp(distr.ppf(pt, 0, scale))* exp(-np.square(scale) / 2)
			weights.append(np.exp(-np.square(-distr.ppf(pt, 0, scale)) / (2*scale**2)) / (np.sqrt(2*np.pi) * scale) * weight)
		else:
			paramPt = distr.ppf(pt, loc, scale)
			weights.append(np.exp(-np.square(paramPt - loc) / (2*scale**2)) / (np.sqrt(2*np.pi) * scale) * weight)
		pts.append(paramPt)
	return pts, weights

def hermgaussParam(numSamples, mean, SD, exp=False):
	"""Computes Gauss-Hermite quadrature points and weights for a given number of samples.

	Note that this functions assumes that E_0 and log(k_0) are normally distributed."""

	GHPts, weights = np.polynomial.hermite.hermgauss(numSamples)
	if exp:
		pts = mean * np.exp(np.sqrt(2) * SD * GHPts)* np.exp(-np.square(SD) / 2)
	else:
		pts = np.sqrt(2) * SD * GHPts + mean
	weights = weights / np.sqrt(np.pi)
	return pts, weights


def productGrid(ERule, ENumPts, KRule, KNumPts):
	"""Returns the tensor product grid for the given quadrature rules and numbers of points."""
	return [(E_0, k_0, wE*wK) for E_0, wE in zip(*ERule(ENumPts))
	for k_0, wK in zip(*KRule(KNumPts))]

def sparseGrid(ERule, KRule, level, EPtsSeq, KPtsSeq):
	"""Returns the sparse grid of a given level for the given quadrature rules."""
	if len(EPtsSeq) < level:
		raise ValueError("The length of EPtsSeq, {0}, was too small for the level {1}".format(len(EPtsSeq), level))
	if len(KPtsSeq) < level:
		raise ValueError("The length of KPtsSeq, {0}, was too small for the level {1}".format(len(KPtsSeq), level))

	smolyakPts = []
	
	# Smolyak formula for 2D.
	# Positive-weight points
	for E_0Pts, wEVals, k_0Pts, wKVals in map(lambda x,y: ERule(x) + KRule(y), 
	EPtsSeq[:level+1], reverse(KPtsSeq[:level+1])):
		smolyakPts.extend([(E_0, k_0, wE*wK) for E_0, wE in zip(E_0Pts, wEVals) for k_0, wK in zip(k_0Pts, wKVals)])

	#Negative-weight points
	for E_0Pts, wEVals, k_0Pts, wKVals in map(lambda x,y: ERule(x) + KRule(y),
	EPtsSeq[:level], reverse(KPtsSeq[:level])):
		smolyakPts.extend([(E_0, k_0, -wE*wK) for E_0, wE in zip(E_0Pts, wEVals) for k_0, wK in zip(k_0Pts, wKVals)])
	return smolyakPts

def reverse(l):
	retL = []
	for x in l:
		retL.insert(0, x)
	return retL

