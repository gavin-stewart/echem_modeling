#include <math.h>

typedef struct Params {
    double potNext;
    double rho;
    double INext;
    double* ICurr;
    double gamma;
    double gamma1;
    double gamma2;
    double gamma3;
    double rhoDivh;
    double h;
    double* thetaCurr;
    double dEpsNext;
    double kappa;
    double eps_0;
} Params;

double discrep(double potNext, double rhof, double INext, double ICurr,
        double gammaf, double gamma1f, double gamma2f, double gamma3f,
        double rhoDivhf, double hf, int i, double* _theta, double dEpsNext,
        double kappaf, double eps_0) {
    double eta = potNext - rhof * INext;
    double cap = gammaf * (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
    eta -= eps_0;
    double expon = exp(0.5 * eta);
    double kRed = kappaf / expon;
    double kOx = kappaf * expon;
    _theta[i+1] = (_theta[i] + hf * kOx) / (1 + hf*(kOx+kRed));
    return cap * (dEpsNext - rhoDivhf * (INext - ICurr)) + (1.0-_theta[i+1])*kOx - _theta[i+1] * kRed - INext;
}

double discrepParam(double INext, void *params) {
    Params* p = (Params*)params;
    double eta = p->potNext - p->rho * INext;
    double cap = p->gamma * (1 + eta * (p->gamma1 + eta * (p->gamma2 + eta * p->gamma3)));
    eta -= p->eps_0;
    double expon = exp(0.5 * eta);
    double kRed = p->kappa / expon;
    double kOx = p->kappa * expon;
    *(p->thetaCurr + 1) = (*(p->thetaCurr) + p->h * kOx) / (1 + p->h*(kOx+kRed));
    return cap * (p->dEpsNext - p->rhoDivh * (INext - *(p->ICurr))) + (1.0 - *(p->thetaCurr + 1) ) * kOx - *(p->thetaCurr + 1)  * kRed - INext;
}

double discrepStep(double potNext, double rhof, double INext, double ICurr,
        double gammaf, double gamma1f, double gamma2f, double gamma3f,
        double rhoDivhf, double hf, int i, double* _theta, double dEpsNext,
        double kappaf, double eps_0, double* fVal) {
    double eta = potNext - rhof * INext;
    double cap = gammaf * 
        (1 + eta * (gamma1f + eta * (gamma2f + eta * gamma3f)));
    eta -= eps_0;
    double expon = exp(0.5 * eta);
    double kRed = kappaf / expon;
    double kOx = kappaf * expon;
    double denom = (1 + hf*(kOx+kRed));
    _theta[i+1] = (_theta[i] + hf * kOx) / denom; 
    double dThetaNext = hf * (-kOx - _theta[i+1] * (kRed - kOx)) / denom;
    *fVal = (cap * (dEpsNext - rhoDivhf * (INext - ICurr)) + 
            (1.0-_theta[i+1])*kOx - _theta[i+1] * kRed - INext);

    double dCap = gammaf * (gamma1f + eta * (2 * gamma2f + eta * 3 * gamma3f));
    double fPrime = -cap * rhoDivhf + 
                    dCap * (dEpsNext - rhoDivhf * (INext - ICurr)) -
                    rhof * 0.5 * (dThetaNext * (kOx + kRed) - 
                              (_theta[i+1] - 1) * kOx + _theta[i+1] * kRed) -1;
    return (*fVal) / fPrime;
}

double discrepStepParam(double INext, double* fVal, void * args) {
    Params* p = (Params*)args;
    double eta = p->potNext - p->rho * INext;
    double cap = p->gamma * 
        (1 + eta * (p->gamma1 + eta * (p->gamma2 + eta * p->gamma3)));
    eta -= p->eps_0;
    double expon = exp(0.5 * eta);
    double kRed = p->kappa / expon;
    double kOx = p->kappa * expon;
    double denom = (1 + p->h*(kOx+kRed));
    double dThetaNext = p->h * (-kOx - *(p->thetaCurr + 1) * 
                        (kRed - kOx)) / denom;
 
    *(p->thetaCurr + 1) = (*(p->thetaCurr) + p->h * kOx) /
                          (1 + p->h*(kOx+kRed));
    *fVal = cap * (p->dEpsNext - p->rhoDivh * (INext - *(p->ICurr))) +
                (1.0 - *(p->thetaCurr + 1) ) * kOx - *(p->thetaCurr + 1) 
                * kRed - INext;
    double dCap = p->gamma * 
                (p->gamma1 + eta * (2 * p->gamma2 + eta * 3 * p->gamma3));
    double fPrime = -cap * p->rhoDivh +
                 dCap * (p->dEpsNext - p->rhoDivh * (INext - *(p->ICurr))) -
                 p->rho * 0.5 * 
                    (dThetaNext * (kOx + kRed) - 
                    (*(p->thetaCurr + 1) - 1) * kOx +
                    *(p->thetaCurr + 1) * kRed) - 1;
    return (*fVal) / fPrime;
}


