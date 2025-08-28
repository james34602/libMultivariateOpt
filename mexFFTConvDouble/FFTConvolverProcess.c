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
	if (nrhs != 2)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverProcess:invalidNumInputs", "Please specify struct");
	else if (nlhs > 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverProcess:maxlhs", "Too many output arguments.");
	FFTConvolver1x1 *nes = (FFTConvolver1x1*)mxGetPr(prhs[0]);
	size_t dm1 = mxGetM(prhs[0]);
	size_t dn1 = mxGetN(prhs[0]);
	if (MIN(dm1, dn1) != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverProcess:maxlhs", "Input struct must be a vector.");
	double *X = (double*)mxGetPr(prhs[1]);
	size_t dm2 = mxGetM(prhs[1]);
	size_t dn2 = mxGetN(prhs[1]);
	size_t arrayLen = MAX(dm2, dn2);
	if (MIN(dm2, dn2) != 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverProcess:maxlhs", "Queuing vector must be a vector.");
	// Create a matrix for the return argument
	if (arrayLen == dm2)
		plhs[0] = mxCreateNumericMatrix(arrayLen, 1, mxDOUBLE_CLASS, mxREAL);
	else
		plhs[0] = mxCreateNumericMatrix(1, arrayLen, mxDOUBLE_CLASS, mxREAL);
	// Assign pointers to the various parameters
	double *Y = (double*)mxGetPr(plhs[0]);
	FFTConvolver1x1Process(nes, X, Y, arrayLen);
	return;
}