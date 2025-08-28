#include <stdio.h>
#include <math.h>
#include <float.h>
#include <mex.h>
#if !defined(MAX)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif
#if !defined(MIN)
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#include "fftConv.c"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Check for proper number of arguments
	if (nrhs != 3)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverLoad:invalidNumInputs", "Please specify struct, impulse response");
	else if (nlhs > 0)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverLoad:maxlhs", "Too many output arguments.");
	FFTConvolver1x1 *nes = (FFTConvolver1x1*)mxGetPr(prhs[0]);
	size_t dm1 = mxGetM(prhs[0]);
	size_t dn1 = mxGetN(prhs[0]);
	if (MIN(dm1, dn1) != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverLoad:maxlhs", "Input struct must be a vector.");
	unsigned int blockSize = (unsigned int)(*((double*)mxGetPr(prhs[1])));
	double *X = (double*)mxGetPr(prhs[2]);
	size_t dm2 = mxGetM(prhs[2]);
	size_t dn2 = mxGetN(prhs[2]);
	size_t arrayLen = MAX(dm2, dn2);
	if (MIN(dm2, dn2) != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverLoad:maxlhs", "Queuing vector must be a vector.");
	FFTConvolver1x1LoadImpulseResponse(nes, blockSize, X, arrayLen);
	return;
}