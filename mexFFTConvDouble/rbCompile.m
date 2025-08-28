mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverInit codelet.c FFTConvolverInit.c
mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverFree codelet.c FFTConvolverFree.c
mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverLoad codelet.c FFTConvolverLoad.c
mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverLoadRefreshNewIR codelet.c FFTConvolverLoadRefreshNewIR.c
mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverProcess codelet.c FFTConvolverProcess.c
mex -v COMPFLAGS="$COMPFLAGS /GL /fp:fast" -output FFTConvolverProcessNoReturn codelet.c FFTConvolverProcessNoReturn.c