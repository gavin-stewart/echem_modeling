
#include <math.h>

#ifndef MIN
#define MIN(a,b) (a<b)?(a):(b)
#endif

#ifndef MAX
#define MAX(a,b) (a>b)?(a):(b)
#endif

int solverNewton(int maxIter, double* x, double (*fOverFPrime)(double, double*, void *), double ftol, void* args) {
	int i;
	double fVal;
	for(i = 0; i < maxIter; i++) {
		*x -= (*fOverFPrime)(*x, &fVal, args);
		if(fabs(fVal) < ftol) {
			return i+1;
		}
	}
	return -1;
}

int solverBisection(int maxIter, double* x, double a, double b, double (*f)(double, void *), double tol, void* args) {
	int i;
	double fLower, fUpper;
	double xLower, xUpper;
	xLower = a;
	xUpper = b;
	fLower = (*f)(a, args);
	fUpper = (*f)(b, args);
	if(fLower > fUpper) {
		double tmp = fLower;
		fLower = fUpper;
		fUpper = tmp;
		tmp = xLower;
		xLower = xUpper;
		xUpper = tmp;
	}
	//Error checking.
	if(fLower > 0 || fUpper < 0) {
		return -1;
	}
	for(i = 0; i < maxIter; i++) {
		*x = (xLower + xUpper) * 0.5;
		if(fabs(xLower - xUpper) < tol) {
			return i+1;
		}
		double fNext = (*f)(*x, args);
		if(fNext < 0) {
			xLower = *x;
		} else {
			xUpper = *x;
		}
	}
	return -1;
}


int solverFalsePosition(int maxIter, double* x, double a, double b, double (*f)(double, void *), double tol, void* args) {
	int i;
	double fLower, fUpper;
	double xLower, xUpper;
	xLower = a;
	xUpper = b;
	fLower = (*f)(a, args);
	fUpper = (*f)(b, args);
	if(fLower > fUpper) {
		double tmp = fLower;
		fLower = fUpper;
		fUpper = tmp;
		tmp = xLower;
		xLower = xUpper;
		xUpper = tmp;
	}
	//Error checking.
	if(fLower > 0 || fUpper < 0) {
		return -1;
	}
	int lastEnd = 0; // For Illinois
	*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
	for(i = 0; i < maxIter; i++) {
		if(fabs(xLower - xUpper) < tol) {
			return i+1;
		}
		double fNext = (*f)(*x, args);
		if(fNext < 0) {
			xLower = *x;
			fLower = fNext;
			if(lastEnd == -1) {
				*x = (2.0 * fUpper * xLower - fLower * xUpper)  / (2.0 * fUpper - fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
			}
			lastEnd = -1;	
		} else {
			xUpper = *x;
			fUpper = fNext;
			if(lastEnd == 1) {
				*x = (fUpper * xLower - 2.0 * fLower * xUpper)  / (fUpper - 2.0 * fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
			}
			lastEnd = 1;	
		}
	}
	return -1;
}

int solverFPBisHybrid(int maxIter, double* x, double a, double b, double (*f)(double, void *), double tol, void* args) {
	int i;
	double fLower, fUpper;
	double xLower, xUpper;
	xLower = a;
	xUpper = b;
	fLower = (*f)(a, args);
	fUpper = (*f)(b, args);
	if(fLower > fUpper) {
		double tmp = fLower;
		fLower = fUpper;
		fUpper = tmp;
		tmp = xLower;
		xLower = xUpper;
		xUpper = tmp;
	}
	//Error checking.
	if(fLower > 0 || fUpper < 0) {
		return -1;
	}
	int lastEnd = 0; // For Illinois
	*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
	for(i = 0; i < maxIter; i++) {
		double len = fabs(xLower - xUpper);
		if(*x < MIN(xLower, xUpper) + 0.0 * len || *x > MAX(xLower, xUpper) + 0.0 * len) { //Poor improvement; switch to bisection.
			*x = (xLower + xUpper) * 0.5;
		}
		if(len < tol) {
			return i+1;
		}
		double fNext = (*f)(*x, args);
		if(fNext < 0) {
			xLower = *x;
			fLower = fNext;
			if(lastEnd == -1) {
				*x = (2.0 * fUpper * xLower - fLower * xUpper)  / (2.0 * fUpper - fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
			}
			lastEnd = -1;	
		} else {
			xUpper = *x;
			fUpper = fNext;
			if(lastEnd == 1) {
				*x = (fUpper * xLower - 2.0 * fLower * xUpper)  / (fUpper - 2.0 * fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper)  / (fUpper - fLower);
			}
			lastEnd = 1;	
		}
	}
	return -1;
}

int solverNewtonBackstep(int maxIter, double* x, double (*fOverFPrime)(double, double *, void *), double (*fFun)(double, void *), double ftol, void *args) {
	int i;
	double fVal, fLast;
	for(i = 0; i < maxIter; i++) {
		double funStep = (*fOverFPrime)(*x, &fLast, args);
		double xNext = *x - funStep;
		fVal = (*fFun)(xNext, args);
		while(fabs(fVal) > fabs(fLast) && fabs(funStep) > 1e-16 * *x) { //Backstep until improvment.
			funStep *= 0.5;
			xNext = *x - funStep;
			fVal = (*fFun)(xNext, args);
		}
		*x = xNext;
		#ifdef NEWTON_DEBUG
		printf("\tNewton value: %f\n", fVal);
		#endif
		if(fabs(fVal) < ftol) {
			return i+1;
		}
	}
	return -1;
}

int solverNewtonWithBisection(int maxIter, double* x, double a, double b, double (*fOverFPrime)(double, double *, void *), double (*fFun)(double, void *), double tol, void *args) {
	int i;
	double fVal;
	double xLower, xUpper;
	if((*fFun)(a, args) > 0) {
		xUpper = a;
		xLower = b;
	} else {
		xLower = a;
		xUpper = b;
	}
	if((*fFun)(xUpper, args) < 0) { //a and b do not bound a root.
		return -1;
	}
	for(i=0; i < maxIter; i++) {
		double funStep = (*fOverFPrime)(*x, &fVal, args);
		*x -= funStep;
		if(*x <= MIN(xLower, xUpper) || *x >=MAX(xLower, xUpper)) {
			*x = (xLower + xUpper) * 0.5;
			fVal = (*fFun)(*x, args);
		}
		
		if(fVal < 0) {
			xLower = *x;
		} else {
			xUpper = *x;
		}
		if(fabs(xLower - xUpper) < tol) {
			return i+1;
		}
		#ifdef NEWTON_DEBUG
		printf("\tNewton value: %f\n", fVal);
		#endif
	}
	return -1;
}

int solverSecant(int maxIter, double *x, double xStart, double (*fFun)(double, void*), double tol, void *args) {
	int i;
	double fVal = fFun(*x, args);
	double fPrev = fFun(xStart, args);
	double xPrev = xStart;
	for(i = 0; i < maxIter; i++) {
		double xNext = (fVal * xPrev - fPrev * *x) / (fVal - fPrev);
		fPrev = fVal;
		xPrev = *x;
		*x = xNext;
		fVal = fFun(*x, args);
		if(fabs(*x - xPrev) < tol) {
			return i+1;
		}
	}
	return -1;
}

int oppositeSign(double x, double y) {
	return (x <=0 && y >= 0) || (x >= 0 && y <= 0);
}

int solverSecantToFP(int maxIter, double *x, double xStart, double (*fFun)(double, void*), double tol, double ftol, void *args) {
	int i;
	double fVal = fFun(*x, args);
	double fStart = fVal;
	double fPrev = fFun(xStart, args);
	double xPrev = xStart;
	int stepShortened;
	for(i=0; !oppositeSign(fVal, fPrev) && i < maxIter; i++) {
		stepShortened = 0;
		double xStep = fVal * (*x - xPrev) / (fVal - fPrev);
		double xNext = *x - xStep;
		double fNext = fFun(xNext, args);
		while(fabs(fNext) >= 0.5 * (fabs(fVal) + fabs(fPrev)) + 2*ftol && i < maxIter) { //No improvement.  Try shortening the step.
			i++;
			xStep *= 0.5;
			xNext = *x - xStep;
			fNext = fFun(xNext, args);
			stepShortened = 1;
		}
		fPrev = fVal;
		xPrev = *x;
		*x = xNext;
		fVal = fFun(*x, args);
		if(!stepShortened && fabs(*x - xPrev) < tol * (1 + fabs(*x) + fabs(xPrev))) {
			return i+1;
		} else if(fabs(fVal) < ftol * (fabs(*x) + 1)) {
			return i+1;
		}
	}
	//If we're here, than either fVal and fPrev have opposite signs or we've reach the max number of iterations.
	double fLower, fUpper;
	double xLower, xUpper;
	if(fVal <= 0 && fPrev >= 0) {
		fLower = fVal;
		xLower = *x;
		fUpper = fPrev;
		xUpper = xPrev;
	} else if(fVal >= 0 && fPrev <= 0) {
		fLower = fPrev;
		xLower = xPrev;
		fUpper = fVal;
		xUpper = *x;
	} 
	
	int lastEnd = 0;
	*x = (fUpper * xLower - fLower * xUpper) / (fUpper - fLower);
	for(; i < maxIter; i++) {
		double fNext = fFun(*x, args);
		if(fabs(fNext) < ftol * (fabs(*x) + 1) || fabs(xUpper - xLower) < tol * (1 + fabs(xLower) + fabs(xUpper))) {
			return i+1;
		} else if(fNext < 0) {
			fLower = fNext;
			xLower = *x;
			if(lastEnd == -1) {
				*x = (2.0 * fUpper * xLower - fLower * xUpper) / (2.0 * fUpper - fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper) / (fUpper - fLower);
			}
			lastEnd = -1;
		} else {
			fUpper = fNext;
			xUpper = *x;
			if(lastEnd == 1){ 
				*x = (fUpper * xLower - 2.0 * fLower * xUpper) / (fUpper - 2.0 * fLower);
			} else {
				*x = (fUpper * xLower - fLower * xUpper) / (fUpper - fLower);
			}
			lastEnd = 1;
		}
	}
	return -1;
}
