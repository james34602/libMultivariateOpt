#include "rand_c.h"
double minArray(double *x, unsigned int N, unsigned int *ind);
void sort(double x_data[], const unsigned int xSize, unsigned int idx_data[]);
double fminsearch(double(*funcPtr)(double*, void*), void *userdat, double *x, unsigned int n, double TolX, double TolFun, unsigned int MaxIter, double *outX);
double fminsearchbnd(double(*funcPtr)(double*, void*), void *userdat, double *x0, double *lb, double *ub, unsigned int n, double TolX, double TolFun, unsigned int MaxIter, double *outX);
double differentialEvolution(double(*funcPtr)(double*, void*), void *userdat, double *initialSolution, unsigned int K, unsigned int N, unsigned int D, double *low, double *up, unsigned int MaxIter, double *gbest, pcg32x2_random_t *PRNG);