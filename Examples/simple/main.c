#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "../../gradfreeOpt/gradfreeOpt.h"
// Optimize simple function
double optSin(double *x, void *usd)
{
	return sin(*x);
}
int main()
{
	unsigned int i, j;
	unsigned int K = 5;
	unsigned int N = 3;
	pcg32x2_random_t PRNG;
	pcg32x2_srandom_r(&PRNG, 36u, 84u, 54u, 54u);
	unsigned int dim = 1;
	// 起始答案
	double initialLowFc = -2;
	double initialUpFc = 2;
	double *initialAns = (double*)malloc(K * N * dim * sizeof(double));
	for (i = 0; i < K * N; i++)
	{
		initialAns[i * dim] = c_rand(&PRNG) * (initialUpFc - initialLowFc) + initialLowFc;
	}
	double *low = (double*)malloc(dim * sizeof(double));
	double *up = (double*)malloc(dim * sizeof(double));
	for (j = 0; j < dim; j++)
	{
		low[j] = -2.0;
		up[j] = 2.0;
	}
	// DE
	void *userdataPtr = 0;
	double *gbestDE = (double*)malloc(dim * sizeof(double));
	double *gbestfminsearch = (double*)malloc(dim * sizeof(double));
	double gmin = differentialEvolution(optSin, userdataPtr, initialAns, K, N, dim, low, up, 1000, gbestDE, &PRNG);
	double fval = fminsearchbnd(optSin, userdataPtr, gbestDE, low, up, dim, 1e-8, 1e-8, 1000, gbestfminsearch);
	printf("%1.14lf %1.14lf\n", gmin, fval);
	for (i = 0; i < dim; i++)
		printf("%1.14lf,", gbestfminsearch[i]);
	free(initialAns);
	free(low);
	free(up);
	free(gbestDE);
	free(gbestfminsearch);
	return 0;
}