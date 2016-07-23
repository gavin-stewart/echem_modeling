
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

