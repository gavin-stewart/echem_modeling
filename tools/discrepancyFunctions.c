#include <math.h>

typedef struct Params {
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
} Params;

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
	Params* p = (Params*)params;
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
	Params* p = (Params*)args;
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


