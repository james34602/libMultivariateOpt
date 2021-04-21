#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <vld.h>
typedef struct
{
	unsigned long long inuse, capacity, grow_num;
	double *data;
} sample_vector;
void init_sample_vector(sample_vector *s, int init_capacity, int grow)
{
	s->data = (double*)malloc(init_capacity * sizeof(double));
	s->inuse = 0;
	s->capacity = init_capacity;
	s->grow_num = grow;
}
void push_back_sample_vector(sample_vector *s, double *x, int lenS)
{
	if ((s->inuse + lenS + (s->grow_num >> 1)) > s->capacity)
	{
		s->capacity += (s->grow_num + lenS);
		s->data = (double*)realloc(s->data, s->capacity * sizeof(double));
	}
	memcpy(s->data + s->inuse, x, lenS * sizeof(double));
	s->inuse += lenS;
}
void clear_sample_vector(sample_vector *s)
{
	s->inuse = 0;
}
void free_sample_vector(sample_vector *s)
{
	free(s->data);
}
#include <math.h>
#include <float.h>
#include "linear_interpolation.h"
//#include <vld.h>
#include "misc.h"
#include "../../gradfreeOpt/gradfreeOpt.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
void validatePeaking(double gain, double fc, double Q, double fs_tf, double *b0, double *b1, double *b2, double *a1, double *a2)
{
	double A = pow(10.0, gain / 40.0);
	double w0 = 2.0 * M_PI * pow(10.0, fc) / fs_tf;
	double sn = sin(w0);
	double cn = cos(w0);
	double alpha = sn / (2.0 * Q);

	double a0_pk = 1.0 + alpha / A;
	*a1 = (-2.0 * cn) / a0_pk;
	*a2 = (1.0 - alpha / A) / a0_pk;

	*b0 = (1.0 + alpha * A) / a0_pk;
	*b1 = (-2.0 * cn) / a0_pk;
	*b2 = (1.0 - alpha * A) / a0_pk;
}
void validateMagCal(double b0, double b1, double b2, double a1, double a2, double *phi, int len, double fs_tf, double *out)
{
	for (int i = 0; i < len; i++)
	{
		double termSqr1 = b0 + b1 + b2;
		double termSqr2 = 1.0 + a1 + a2;
		double term1 = (termSqr1 * termSqr1) + ((b0 * b2 * phi[i]) - (b1 * (b0 + b2) + (4.0 * b0 * b2))) * phi[i];
		double term2 = (termSqr2 * termSqr2) + ((a2 * phi[i]) - (a1 * (1.0 + a2) + (4.0 * a2))) * phi[i];
		if (term1 < DBL_EPSILON)
			term1 = DBL_EPSILON;
		if (term2 < DBL_EPSILON)
			term2 = DBL_EPSILON;
		double eq_op = 10.0 * (log10(term1) - log10(term2));
		out[i] += eq_op;
	}
}
typedef struct
{
	double fs;
	double *phi;
	unsigned int gridSize;
	double *target;
	unsigned int numBands;
	double *tmp;
} optUserdata;
double peakingCostFunctionMap(double *x, void *usd)
{
	optUserdata *userdata = (optUserdata*)usd;
	double *fc = x;
	double *Q = x + userdata->numBands;
	double *gain = x + userdata->numBands * 2;
	double b0, b1, b2, a1, a2;
	memset(userdata->tmp, 0, userdata->gridSize * sizeof(double));
	for (unsigned int i = 0; i < userdata->numBands; i++)
	{
		validatePeaking(gain[i], fc[i], Q[i], userdata->fs, &b0, &b1, &b2, &a1, &a2);
		validateMagCal(b0, b1, b2, a1, a2, userdata->phi, userdata->gridSize, userdata->fs, userdata->tmp);
		//printf("Band: %d, %1.14lf, %1.14lf, %1.14lf\n", i + 1, fc[i], Q[i], gain[i]);
	}
	double meanAcc = 0.0;
	for (unsigned int i = 0; i < userdata->gridSize; i++)
	{
		double error = userdata->tmp[i] - userdata->target[i];
		meanAcc += error * error;
	}
	meanAcc = meanAcc / (double)userdata->gridSize;
	return meanAcc;
}
#include "peakfinder.h"
unsigned int* peakfinder_wrapper(double *x, unsigned int n, double sel, unsigned int extrema, unsigned int *numPeaks)
{
	unsigned int *peakInds = (unsigned int*)malloc(n * sizeof(unsigned int));
	*numPeaks = peakfinder(n, x, sel, extrema, peakInds); // 1u = maxima, 0u = minima
	return peakInds;
}
void diff(double *y, double *f, unsigned int sz)
{
	--sz;
	for (unsigned int i = 0; i < sz; i++)
		f[i] = y[i + 1] - y[i];
}
void derivative(double *x, unsigned int n, unsigned int NorderDerivative, double *dx, double *dif)
{
	// DERIVATIVE Compute derivative while preserving dimensions.
	memcpy(dx, x, n * sizeof(double));
	for (unsigned int i = 0; i < NorderDerivative; i++)
	{
		diff(dx, dif, n);
		dx[0] = dif[0];
		for (unsigned int j = 1; j < n - 1; j++)
		{
			dx[j] = (dif[j] + dif[j - 1]) * 0.5;
		}
		dx[n - 1] = dif[n - 2];
	}
}
char* readTextFile(char *filename)
{
	char *buffer = 0;
	long length;
	FILE *textFile = fopen(filename, "rb");
	if (textFile)
	{
		fseek(textFile, 0, SEEK_END);
		length = ftell(textFile);
		fseek(textFile, 0, SEEK_SET);
		buffer = (char*)malloc(length + 1);
		if (buffer)
			fread(buffer, 1, length, textFile);
		fclose(textFile);
		buffer[length] = '\0';
	}
	return buffer;
}
// Optimize peaking filters(IIR SOS)
int main()
{
	sample_vector xAxis, yAxis;
	init_sample_vector(&xAxis, 128, 64);
	init_sample_vector(&yAxis, 128, 64);
	char *buf = readTextFile("flt_freqList.txt");
	char *err, *p = buf;
	double val;
	while (*p) {
		val = strtod(p, &err);
		if (p == err)
			p++;
		else if ((err == NULL) || (*err == 0))
		{
			push_back_sample_vector(&xAxis, &val, 1);
			break;
		}
		else
		{
			push_back_sample_vector(&xAxis, &val, 1);
			p = err + 1;
		}
	}
	free(buf);
	buf = readTextFile("target.txt");
	p = buf;
	while (*p) {
		val = strtod(p, &err);
		if (p == err)
			p++;
		else if ((err == NULL) || (*err == 0))
		{
			push_back_sample_vector(&yAxis, &val, 1);
			break;
		}
		else
		{
			push_back_sample_vector(&yAxis, &val, 1);
			p = err + 1;
		}
	}
	free(buf);
	unsigned int i, j;
	unsigned int K = 5;
	unsigned int N = 3;
	double fs = 44100.0;
	unsigned int ud_gridSize = yAxis.inuse;
	double *flt_freqList = xAxis.data;
	double *target = yAxis.data;
	unsigned int numMaximas, numMinimas;
	unsigned int *maximaIndex = peakfinder_wrapper(target, ud_gridSize, 0.1, 1, &numMaximas);
	unsigned int *minimaIndex = peakfinder_wrapper(target, ud_gridSize, 0.1, 0, &numMinimas);
	double *flt_fc = (double*)malloc((numMaximas + numMinimas) * sizeof(double));
	unsigned int *idx = (unsigned int*)malloc((numMaximas + numMinimas) * sizeof(unsigned int));
	for (i = 0; i < numMaximas; i++)
		flt_fc[i] = flt_freqList[maximaIndex[i]];
	for (i = numMaximas; i < (numMaximas + numMinimas); i++)
		flt_fc[i] = flt_freqList[minimaIndex[i - numMaximas]];
	sort(flt_fc, (numMaximas + numMinimas), idx);
	//for (i = 0; i < (numMaximas + numMinimas); i++)
	//	printf("%1.14lf\n", flt_fc[i]);
	double smallestJump = 0.0;
	double lowestFreq2Gen = 200.0;
	double highestFreq2Gen = 14000.0;
	pcg32x2_random_t PRNG;
	pcg32x2_srandom_r(&PRNG, 36u, 84u, 54u, 54u);
	double *dFreqDiscontin = (double*)malloc((numMaximas + numMinimas) * sizeof(double));
	double *dif = (double*)malloc((numMaximas + numMinimas) * sizeof(double));
	while (smallestJump <= 20.0)
	{
		derivative(flt_fc, numMaximas + numMinimas, 1, dFreqDiscontin, dif);
		unsigned int smIdx;
		smallestJump = minArray(dFreqDiscontin, numMaximas + numMinimas, &smIdx);
		double newFreq = c_rand(&PRNG) * (highestFreq2Gen - lowestFreq2Gen) + lowestFreq2Gen;
		flt_fc[smIdx] = newFreq;
		sort(flt_fc, (numMaximas + numMinimas), idx);
	}
	unsigned int numBands;
	if (flt_fc[0] > 80.0)
	{
		numBands = numMaximas + numMinimas + 2;
		double *tmp = (double*)malloc(numBands * sizeof(double));
		memcpy(tmp + 2, flt_fc, (numMaximas + numMinimas) * sizeof(double));
		tmp[0] = 20.0; tmp[1] = 60.0;
		free(flt_fc);
		flt_fc = tmp;
	}
	else if (flt_fc[0] > 40.0)
	{
		numBands = numMaximas + numMinimas + 1;
		double *tmp = (double*)malloc(numBands * sizeof(double));
		memcpy(tmp + 1, flt_fc, (numMaximas + numMinimas) * sizeof(double));
		tmp[0] = 20.0;
		free(flt_fc);
		flt_fc = tmp;
	}
	else
		numBands = numMaximas + numMinimas;
	double *flt_peak_g = (double*)malloc(numBands * sizeof(double));
	for (i = 0; i < numBands; i++)
		flt_peak_g[i] = npointWndFunction(flt_fc[i], flt_freqList, target, ud_gridSize);
	double lowFc = 20;
	double upFc = fs / 2 - 1;
	double lowQ = 0.2;
	double upQ = 16;
	double lowGain = target[0];
	double upGain = target[0];
	for (i = 1; i < ud_gridSize; i++)
	{
		if (target[i] < lowGain)
			lowGain = target[i];
		if (target[i] > upGain)
			upGain = target[i];
	}
	lowGain -= 5.0;
	upGain += 5.0;
	double *initialQ = (double*)malloc(numBands * sizeof(double));
	for (i = 0; i < numBands; i++)
	{
		initialQ[i] = fabs(randn_pcg32x2(&PRNG) * (5 - 0.7) + 0.7);
		flt_fc[i] = log10(flt_fc[i]);
	}
	//for (i = 0; i < numBands; i++)
	//	printf("%1.14lf %1.14lf\n", initialQ[i], flt_fc[i]);
	double *phi = (double*)malloc(ud_gridSize * sizeof(double));
	for (i = 0; i < ud_gridSize; i++)
	{
		double term1 = sin(M_PI * flt_freqList[i] / fs);
		phi[i] = 4.0 * term1 * term1;
	}
	unsigned int dim = numBands * 3;
	// 起始答案
	double initialLowGain = -1.5;
	double initialUpGain = 1.5;
	double initialLowQ = -0.5;
	double initialUpQ = 0.5;
	double initialLowFc = -log10(2);
	double initialUpFc = log10(2);
	double *initialAns = (double*)malloc(K * N * dim * sizeof(double));
	for (i = 0; i < K * N; i++)
	{
		for (j = 0; j < numBands; j++)
		{
			initialAns[i * dim + j] = flt_fc[j] + c_rand(&PRNG) * (initialUpFc - initialLowFc) + initialLowFc;
			initialAns[i * dim + numBands + j] = initialQ[j] + c_rand(&PRNG) * (initialUpQ - initialLowQ) + initialLowQ;
			initialAns[i * dim + numBands * 2 + j] = flt_peak_g[j] + c_rand(&PRNG) * (initialUpGain - initialLowGain) + initialLowGain;
		}
	}
	//fp = fopen("bm.txt", "wb");
	//for (i = 0; i < K * N; i++)
	//{
	//	for (j = 0; j < dim; j++)
	//		fprintf(fp, "%1.14lf ", initialAns[i * dim + j]);
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	double *low = (double*)malloc(dim * sizeof(double));
	double *up = (double*)malloc(dim * sizeof(double));
	for (j = 0; j < numBands; j++)
	{
		low[j] = log10(lowFc); low[numBands + j] = lowQ; low[numBands * 2 + j] = lowGain;
		up[j] = log10(upFc); up[numBands + j] = upQ; up[numBands * 2 + j] = upGain;
	}
	// DE
	double *tmpDat = (double*)malloc(ud_gridSize * sizeof(double));
	optUserdata userdat;
	userdat.fs = fs;
	userdat.numBands = numBands;
	userdat.phi = phi;
	userdat.target = target;
	userdat.tmp = tmpDat;
	userdat.gridSize = ud_gridSize;
	void *userdataPtr = (void*)&userdat;
	double *gbestDE = (double*)malloc(dim * sizeof(double));
	double *gbestfminsearch = (double*)malloc(dim * sizeof(double));
	// Method 1
	double gmin = differentialEvolution(peakingCostFunctionMap, userdataPtr, initialAns, K, N, dim, low, up, 10, gbestDE, &PRNG);
	// Method 2
	double fval = fminsearchbnd(peakingCostFunctionMap, userdataPtr, initialAns, low, up, dim, 1e-8, 1e-8, 10, gbestfminsearch);
	printf("%1.14lf %1.14lf\n", gmin, fval);
	for (i = 0; i < dim; i++)
		printf("%1.14lf,", gbestfminsearch[i]);
	free(maximaIndex);
	free(minimaIndex);
	free(flt_fc);
	free(idx);
	free(dFreqDiscontin);
	free(dif);
	free(flt_peak_g);
	free(initialQ);
	free(phi);
	free(initialAns);
	free(low);
	free(up);
	free(tmpDat);
	free(gbestDE);
	free(gbestfminsearch);
	free_sample_vector(&xAxis);
	free_sample_vector(&yAxis);
	return 0;
}