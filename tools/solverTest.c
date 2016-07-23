#include <math.h>
#include <stdio.h>
#include "solvers.c"
#include <math.h>

double fOverFPrime(double, double*);
double f(double);

void main() {
	double x = 3.0;
	int numSteps = solverNewton(100, &x, &fOverFPrime, 1e-10);
	printf("\tNewton's method: found solution %.10f in %d steps.\n", x, numSteps);
	x= 3.0;
	double a=3.0;
	double b = 3.2;
	numSteps = solverBisection(100, &x, a, b, &f, 1e-10);
	printf("\tBisection method: found solution %.10f in %d steps.\n", x, numSteps);
	numSteps = solverFalsePosition(500, &x, a, b, &f, 1e-10);
	printf("\tFalse position method: found solution %.10f in %d steps.\n", x, numSteps);
	numSteps = solverFPBisHybrid(100, &x, a, b, &f, 1e-10);
	printf("\tFalse position/bisection hybrid: found solution %.10f in %d steps.\n", x, numSteps);
	numSteps = solverNewtonBackstep(100, &x, &fOverFPrime, &f, 1e-10);
	printf("\tNewton's method with backstepping: found solution %.10f in %d steps.\n", x, numSteps);
}


double fOverFPrime(double x, double* fx) {
	*fx = sin(x);
	return tan(x);
}

double f(double x) {
	return sin(x);
}
