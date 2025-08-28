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
	if (nrhs != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverFree:invalidNumInputs", "Please specify struct");
	else if (nlhs > 0)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverFree:maxlhs", "Too many output arguments.");
	FFTConvolver1x1 *nes = (FFTConvolver1x1*)mxGetPr(prhs[0]);
	size_t dm1 = mxGetM(prhs[0]);
	size_t dn1 = mxGetN(prhs[0]);
	if (MIN(dm1, dn1) != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverFree:maxlhs", "Input struct must be a vector.");
	if (nes->_segmentsRe)
		FFTConvolver1x1Free(nes);
	return;
}