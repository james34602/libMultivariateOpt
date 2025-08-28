#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mex.h>
#include "fftConv.c"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Check for proper number of arguments
	if (nlhs > 1)
		mexErrMsgIdAndTxt("MATLAB:FFTConvolverInit:maxlhs", "Too many output arguments.");
    // Create a matrix for the return argument
    plhs[0] = mxCreateNumericMatrix(1, sizeof(FFTConvolver1x1), mxUINT8_CLASS, mxREAL);
    FFTConvolver1x1 *nes = (FFTConvolver1x1*)mxGetPr(plhs[0]);
    FFTConvolver1x1Init(nes);
    return;
}