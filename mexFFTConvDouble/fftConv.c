#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
extern unsigned int upper_power_of_two(unsigned int v);
/**
* @class FFTConvolver1x1
* @brief Implementation of a partitioned FFT convolution algorithm with uniform block size
*
* Some notes on how to use it:
*
* - After initialization with an impulse response, subsequent data portions of
*   arbitrary length can be convolved. The convolver internally can handle
*   this by using appropriate buffering.
*
* - The convolver works without "latency" (except for the required
*   processing time, of course), i.e. the output always is the convolved
*   input for each processing call.
*
* - The convolver is suitable for real-time processing which means that no
*   "unpredictable" operations like allocations, locking, API calls, etc. are
*   performed during processing (all necessary allocations and preparations take
*   place during initialization).
*/
typedef struct
{
	unsigned int _blockSize;
	unsigned int _segSize;
	unsigned int _segCount;
	unsigned int _segCountMinus1;
	unsigned int _fftComplexSize;
	double **_segmentsRe;
	double **_segmentsIm;
	double **_segmentsIRRe;
	double **_segmentsIRIm;
	double *_fftBuffer;
	unsigned int *bit;
	double *sine;
	double *_preMultiplied[2];
	double *_overlap;
	unsigned int _current;
	double *_inputBuffer;
	unsigned int _inputBufferFill;
	double gain; // double32, it's perfectly safe to have blockSize == 2097152, however, it's impractical to have such large block
	void(*fft)(double*, const double*);
} FFTConvolver1x1;
#include "codelet.h"
unsigned int fhtIntegerLog2(unsigned int v)
{
	unsigned int i = 0;
	while (v > 1)
	{
		++i;
		v >>= 1;
	}
	return i;
}
unsigned fhtRevBits(unsigned int x, unsigned int bits)
{
	unsigned int y = 0;
	while (bits--)
	{
		y = (y + y) + (x & 1);
		x >>= 1;
	}
	return y;
}
void fhtbitReversalTbl(unsigned *dst, unsigned int n)
{
	unsigned int bits = fhtIntegerLog2(n);
	for (unsigned int i = 0; i < n; ++i)
		dst[i] = fhtRevBits(i, bits);
}
void fhtsinHalfTbldouble(double *dst, unsigned int n)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	for (unsigned int i = 0; i < n; ++i)
		dst[i] = (double)sin(twopi_over_n * i);
}
unsigned int upper_power_of_two(unsigned int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return ++v;
}
void FFTConvolver1x1Init(FFTConvolver1x1 *conv)
{
	conv->bit = 0;
	conv->_blockSize = 0;
	conv->_segSize = 0;
	conv->_segCount = 0;
	conv->_fftComplexSize = 0;
	conv->_segmentsRe = 0;
	conv->_segmentsIm = 0;
	conv->_segmentsIRRe = 0;
	conv->_segmentsIRIm = 0;
	conv->_current = 0;
	conv->_fftBuffer = 0;
	conv->_inputBuffer = 0;
	conv->_overlap = 0;
	conv->_inputBufferFill = 0;
	conv->_preMultiplied[0] = 0;
}
void FFTConvolver1x1Free(FFTConvolver1x1 *conv)
{
	if (conv->_segmentsRe)
	{
		for (unsigned int i = 0; i < conv->_segCount; ++i)
		{
			free(conv->_segmentsRe[i]);
			free(conv->_segmentsIm[i]);
			free(conv->_segmentsIRRe[i]);
			free(conv->_segmentsIRIm[i]);
		}
		free(conv->_segmentsRe);
		free(conv->_segmentsIm);
		free(conv->_segmentsIRRe);
		free(conv->_segmentsIRIm);
		conv->_segmentsRe = 0;
		conv->_segmentsIm = 0;
		conv->_segmentsIRRe = 0;
		conv->_segmentsIRIm = 0;
	}
	if (conv->_preMultiplied[0])
	{
		free(conv->_preMultiplied[0]);
		free(conv->_preMultiplied[1]);
		conv->_preMultiplied[0] = 0;
	}
	if (conv->bit)
	{
		free(conv->bit);
		free(conv->sine);
		conv->bit = 0;
	}
	conv->_blockSize = 0;
	conv->_segSize = 0;
	conv->_segCount = 0;
	conv->_fftComplexSize = 0;
	if (conv->_fftBuffer)
	{
		free(conv->_fftBuffer);
		conv->_fftBuffer = 0;
	}
	if (conv->_overlap)
	{
		free(conv->_overlap);
		conv->_overlap = 0;
	}
	if (conv->_inputBuffer)
	{
		free(conv->_inputBuffer);
		conv->_inputBuffer = 0;
	}
	conv->_current = 0;
	conv->_inputBufferFill = 0;
}
extern void fhtbitReversalTbl(unsigned *dst, unsigned int n);
extern void fhtsinHalfTbldouble(double *dst, unsigned int n);
int FFTConvolver1x1LoadImpulseResponse(FFTConvolver1x1 *conv, unsigned int blockSize, const double* ir, unsigned int irLen)
{
	if (blockSize == 0)
		return 0;
	if (irLen == 0)
		return 0;

	if (conv->bit)
		FFTConvolver1x1Free(conv);
	conv->_blockSize = upper_power_of_two(blockSize);
	conv->_segSize = 2 * conv->_blockSize;
	conv->_segCount = (unsigned int)ceil((double)irLen / (double)conv->_blockSize);
	conv->_segCountMinus1 = conv->_segCount - 1;
	conv->_fftComplexSize = (conv->_segSize >> 1) + 1;

	// FFT
	if (conv->_segSize == 2)
		conv->fft = DFT2;
	else if (conv->_segSize == 4)
		conv->fft = DFT4;
	else if (conv->_segSize == 8)
		conv->fft = DFT8;
	else if (conv->_segSize == 16)
		conv->fft = DFT16;
	else if (conv->_segSize == 32)
		conv->fft = DFT32;
	else if (conv->_segSize == 64)
		conv->fft = DFT64;
	else if (conv->_segSize == 128)
		conv->fft = DFT128;
	else if (conv->_segSize == 256)
		conv->fft = DFT256;
	else if (conv->_segSize == 512)
		conv->fft = DFT512;
	else if (conv->_segSize == 1024)
		conv->fft = DFT1024;
	else if (conv->_segSize == 2048)
		conv->fft = DFT2048;
	else if (conv->_segSize == 4096)
		conv->fft = DFT4096;
	else if (conv->_segSize == 8192)
		conv->fft = DFT8192;
	else if (conv->_segSize == 16384)
		conv->fft = DFT16384;
	else if (conv->_segSize == 32768)
		conv->fft = DFT32768;
	else if (conv->_segSize == 65536)
		conv->fft = DFT65536;
	else if (conv->_segSize == 131072)
		conv->fft = DFT131072;
	else if (conv->_segSize == 262144)
		conv->fft = DFT262144;
	else if (conv->_segSize == 524288)
		conv->fft = DFT524288;
	else if (conv->_segSize == 1048576)
		conv->fft = DFT1048576;
	conv->bit = (unsigned int*)malloc(conv->_segSize * sizeof(unsigned int));
	conv->sine = (double*)malloc(conv->_segSize * sizeof(double));
	fhtbitReversalTbl(conv->bit, conv->_segSize);
	fhtsinHalfTbldouble(conv->sine, conv->_segSize);
	conv->_fftBuffer = (double*)malloc(conv->_segSize * sizeof(double));

	// Prepare segments
	conv->_segmentsRe = (double**)malloc(conv->_segCount * sizeof(double*));
	conv->_segmentsIm = (double**)malloc(conv->_segCount * sizeof(double*));
	for (unsigned int i = 0; i < conv->_segCount; ++i)
	{
		conv->_segmentsRe[i] = (double*)malloc(conv->_fftComplexSize * sizeof(double));
		conv->_segmentsIm[i] = (double*)malloc(conv->_fftComplexSize * sizeof(double));
		memset(conv->_segmentsRe[i], 0, conv->_fftComplexSize * sizeof(double));
		memset(conv->_segmentsIm[i], 0, conv->_fftComplexSize * sizeof(double));
	}

	// Prepare IR
	conv->_segmentsIRRe = (double**)malloc(conv->_segCount * sizeof(double*));
	conv->_segmentsIRIm = (double**)malloc(conv->_segCount * sizeof(double*));
	for (unsigned int i = 0; i < conv->_segCount; ++i)
	{
		double* segmentRe = (double*)malloc(conv->_fftComplexSize * sizeof(double));
		double* segmentIm = (double*)malloc(conv->_fftComplexSize * sizeof(double));
		const unsigned int remaining = irLen - (i * conv->_blockSize);
		const unsigned int sizeCopy = (remaining >= conv->_blockSize) ? conv->_blockSize : remaining;
		for (unsigned int j = 0; j < sizeCopy; j++)
			conv->_fftBuffer[conv->bit[j]] = ir[i*conv->_blockSize + j];
		for (unsigned int j = sizeCopy; j < conv->_segSize; j++)
			conv->_fftBuffer[conv->bit[j]] = 0.0;
		conv->fft(conv->_fftBuffer, conv->sine);
		segmentRe[0] = conv->_fftBuffer[0] * 2.0;
		segmentIm[0] = 0.0;
		for (unsigned int j = 1; j < conv->_fftComplexSize; j++)
		{
			unsigned int symIdx = conv->_segSize - j;
			segmentRe[j] = conv->_fftBuffer[j] + conv->_fftBuffer[symIdx];
			segmentIm[j] = conv->_fftBuffer[j] - conv->_fftBuffer[symIdx];
		}
		conv->_segmentsIRRe[i] = segmentRe;
		conv->_segmentsIRIm[i] = segmentIm;
	}

	// Prepare convolution buffers
	conv->_preMultiplied[0] = (double*)malloc(conv->_fftComplexSize * sizeof(double));
	conv->_preMultiplied[1] = (double*)malloc(conv->_fftComplexSize * sizeof(double));
	memset(conv->_preMultiplied[0], 0, conv->_fftComplexSize * sizeof(double));
	memset(conv->_preMultiplied[1], 0, conv->_fftComplexSize * sizeof(double));
	conv->_overlap = (double*)malloc(conv->_blockSize * sizeof(double));
	memset(conv->_overlap, 0, conv->_blockSize * sizeof(double));

	// Prepare input buffer
	conv->_inputBuffer = (double*)malloc(conv->_blockSize * sizeof(double));
	memset(conv->_inputBuffer, 0, conv->_blockSize * sizeof(double));
	conv->_inputBufferFill = 0;

	// Reset current position
	conv->_current = 0;
	conv->gain = 1.0 / ((double)conv->_segSize * 2.0);
	return 1;
}
int FFTConvolver1x1RefreshImpulseResponse(FFTConvolver1x1 *conv, const double* ir, unsigned int irLen)
{
	if (irLen == 0)
		return 0;
	// Prepare IR
	for (unsigned int i = 0; i < conv->_segCount; ++i)
	{
		double* segmentRe = conv->_segmentsIRRe[i];
		double* segmentIm = conv->_segmentsIRIm[i];
		const unsigned int remaining = irLen - (i * conv->_blockSize);
		const unsigned int sizeCopy = (remaining >= conv->_blockSize) ? conv->_blockSize : remaining;
		for (unsigned int j = 0; j < sizeCopy; j++)
			conv->_fftBuffer[conv->bit[j]] = ir[i*conv->_blockSize + j];
		for (unsigned int j = sizeCopy; j < conv->_segSize; j++)
			conv->_fftBuffer[conv->bit[j]] = 0.0;
		conv->fft(conv->_fftBuffer, conv->sine);
		segmentRe[0] = conv->_fftBuffer[0] * 2.0;
		segmentIm[0] = 0.0;
		for (unsigned int j = 1; j < conv->_fftComplexSize; j++)
		{
			unsigned int symIdx = conv->_segSize - j;
			segmentRe[j] = conv->_fftBuffer[j] + conv->_fftBuffer[symIdx];
			segmentIm[j] = conv->_fftBuffer[j] - conv->_fftBuffer[symIdx];
		}
	}
	return 1;
}
void FFTConvolver1x1Process(FFTConvolver1x1 *conv, const double* input, double* output, unsigned int len)
{
	unsigned int j, symIdx;
	unsigned int processed = 0;
	while (processed < len)
	{
		const int inputBufferWasEmpty = (conv->_inputBufferFill == 0);
		const unsigned int processing = min(len - processed, conv->_blockSize - conv->_inputBufferFill);
		const unsigned int inputBufferPos = conv->_inputBufferFill;
		memcpy(conv->_inputBuffer + inputBufferPos, input + processed, processing * sizeof(double));

		// Forward FFT
		for (j = 0; j < conv->_blockSize; j++)
			conv->_fftBuffer[conv->bit[j]] = conv->_inputBuffer[j];
		for (j = conv->_blockSize; j < conv->_segSize; j++)
			conv->_fftBuffer[conv->bit[j]] = 0.0;
		conv->fft(conv->_fftBuffer, conv->sine);
		conv->_segmentsRe[conv->_current][0] = conv->_fftBuffer[0];
		for (j = 1; j < conv->_fftComplexSize; j++)
		{
			symIdx = conv->_segSize - j;
			conv->_segmentsRe[conv->_current][j] = conv->_fftBuffer[j] + conv->_fftBuffer[symIdx];
			conv->_segmentsIm[conv->_current][j] = conv->_fftBuffer[j] - conv->_fftBuffer[symIdx];
		}

		// Complex multiplication
		const double *reA;
		const double *imA;
		const double *reB;
		const double *imB;
		unsigned int end4 = conv->_fftComplexSize - 1;
		if (inputBufferWasEmpty)
		{
			unsigned int segFrameIndex = (conv->_current + 1) % conv->_segCount;
			if (conv->_segCount > 1)
			{
				double *re = conv->_preMultiplied[0];
				double *im = conv->_preMultiplied[1];
				reA = conv->_segmentsIRRe[1];
				imA = conv->_segmentsIRIm[1];
				reB = conv->_segmentsRe[segFrameIndex];
				imB = conv->_segmentsIm[segFrameIndex];
				for (j = 0; j < end4; j += 4)
				{
					re[j + 0] = reA[j + 0] * reB[j + 0] - imA[j + 0] * imB[j + 0];
					re[j + 1] = reA[j + 1] * reB[j + 1] - imA[j + 1] * imB[j + 1];
					re[j + 2] = reA[j + 2] * reB[j + 2] - imA[j + 2] * imB[j + 2];
					re[j + 3] = reA[j + 3] * reB[j + 3] - imA[j + 3] * imB[j + 3];
					im[j + 0] = reA[j + 0] * imB[j + 0] + imA[j + 0] * reB[j + 0];
					im[j + 1] = reA[j + 1] * imB[j + 1] + imA[j + 1] * reB[j + 1];
					im[j + 2] = reA[j + 2] * imB[j + 2] + imA[j + 2] * reB[j + 2];
					im[j + 3] = reA[j + 3] * imB[j + 3] + imA[j + 3] * reB[j + 3];
				}
				re[end4] = reA[end4] * reB[end4] - imA[end4] * imB[end4];
				im[end4] = reA[end4] * imB[end4] + imA[end4] * reB[end4];
				for (unsigned int i = 2; i < conv->_segCount; ++i)
				{
					segFrameIndex = (conv->_current + i) % conv->_segCount;
					re = conv->_preMultiplied[0];
					im = conv->_preMultiplied[1];
					reA = conv->_segmentsIRRe[i];
					imA = conv->_segmentsIRIm[i];
					reB = conv->_segmentsRe[segFrameIndex];
					imB = conv->_segmentsIm[segFrameIndex];
					for (j = 0; j < end4; j += 4)
					{
						re[j + 0] += reA[j + 0] * reB[j + 0] - imA[j + 0] * imB[j + 0];
						re[j + 1] += reA[j + 1] * reB[j + 1] - imA[j + 1] * imB[j + 1];
						re[j + 2] += reA[j + 2] * reB[j + 2] - imA[j + 2] * imB[j + 2];
						re[j + 3] += reA[j + 3] * reB[j + 3] - imA[j + 3] * imB[j + 3];
						im[j + 0] += reA[j + 0] * imB[j + 0] + imA[j + 0] * reB[j + 0];
						im[j + 1] += reA[j + 1] * imB[j + 1] + imA[j + 1] * reB[j + 1];
						im[j + 2] += reA[j + 2] * imB[j + 2] + imA[j + 2] * reB[j + 2];
						im[j + 3] += reA[j + 3] * imB[j + 3] + imA[j + 3] * reB[j + 3];
					}
					re[end4] += reA[end4] * reB[end4] - imA[end4] * imB[end4];
					im[end4] += reA[end4] * imB[end4] + imA[end4] * reB[end4];
				}
			}
		}
		reA = conv->_segmentsIRRe[0];
		imA = conv->_segmentsIRIm[0];
		reB = conv->_segmentsRe[conv->_current];
		imB = conv->_segmentsIm[conv->_current];
		const double *srcRe = conv->_preMultiplied[0];
		const double *srcIm = conv->_preMultiplied[1];
		double real, imag;
		conv->_fftBuffer[0] = reB[0] * reA[0] + srcRe[0];
		for (j = 1; j < conv->_fftComplexSize; ++j)
		{
			symIdx = conv->_segSize - j;
			real = reB[j] * reA[j] - imB[j] * imA[j] + srcRe[j];
			imag = reB[j] * imA[j] + imB[j] * reA[j] + srcIm[j];
			conv->_fftBuffer[conv->bit[j]] = (real + imag) * 0.5f;
			conv->_fftBuffer[conv->bit[symIdx]] = (real - imag) * 0.5f;
		}
		// Backward FFT
		conv->fft(conv->_fftBuffer, conv->sine);

		// Add overlap
		double *result = output + processed;
		const double *a = conv->_fftBuffer + inputBufferPos;
		const double *b = conv->_overlap + inputBufferPos;
		end4 = (processing >> 2) << 2;
		for (j = 0; j < end4; j += 4)
		{
			result[j + 0] = (a[j + 0] + b[j + 0]) * conv->gain;
			result[j + 1] = (a[j + 1] + b[j + 1]) * conv->gain;
			result[j + 2] = (a[j + 2] + b[j + 2]) * conv->gain;
			result[j + 3] = (a[j + 3] + b[j + 3]) * conv->gain;
		}
		for (j = end4; j < processing; ++j)
			result[j] = (a[j] + b[j]) * conv->gain;

		// Input buffer full => Next block
		conv->_inputBufferFill += processing;
		if (conv->_inputBufferFill == conv->_blockSize)
		{
			// Input buffer is empty again now
			memset(conv->_inputBuffer, 0, conv->_blockSize * sizeof(double));
			conv->_inputBufferFill = 0;
			// Save the overlap
			memcpy(conv->_overlap, conv->_fftBuffer + conv->_blockSize, conv->_blockSize * sizeof(double));
			// Update current segment
			conv->_current = (conv->_current > 0) ? (conv->_current - 1) : conv->_segCountMinus1;
		}
		processed += processing;
	}
}
void FFTConvolver1x1ProcessNoReturn(FFTConvolver1x1 *conv, const double* input, unsigned int len)
{
	unsigned int j, symIdx;
	unsigned int processed = 0;
	while (processed < len)
	{
		const int inputBufferWasEmpty = (conv->_inputBufferFill == 0);
		const unsigned int processing = min(len - processed, conv->_blockSize - conv->_inputBufferFill);
		const unsigned int inputBufferPos = conv->_inputBufferFill;
		memcpy(conv->_inputBuffer + inputBufferPos, input + processed, processing * sizeof(double));

		// Forward FFT
		for (j = 0; j < conv->_blockSize; j++)
			conv->_fftBuffer[conv->bit[j]] = conv->_inputBuffer[j];
		for (j = conv->_blockSize; j < conv->_segSize; j++)
			conv->_fftBuffer[conv->bit[j]] = 0.0;
		conv->fft(conv->_fftBuffer, conv->sine);
		conv->_segmentsRe[conv->_current][0] = conv->_fftBuffer[0];
		for (j = 1; j < conv->_fftComplexSize; j++)
		{
			symIdx = conv->_segSize - j;
			conv->_segmentsRe[conv->_current][j] = conv->_fftBuffer[j] + conv->_fftBuffer[symIdx];
			conv->_segmentsIm[conv->_current][j] = conv->_fftBuffer[j] - conv->_fftBuffer[symIdx];
		}
		// Input buffer full => Next block
		conv->_inputBufferFill += processing;
		if (conv->_inputBufferFill == conv->_blockSize)
		{
			// Input buffer is empty again now
			memset(conv->_inputBuffer, 0, conv->_blockSize * sizeof(double));
			conv->_inputBufferFill = 0;
			// Update current segment
			conv->_current = (conv->_current > 0) ? (conv->_current - 1) : conv->_segCountMinus1;
		}
		processed += processing;
	}
}