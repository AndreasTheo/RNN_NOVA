#include <iomanip>
#include "tanh.h"


//range [-1,1]
/*
========================================================================
//tanHyp
========================================================================
*/
double TanHyp(double x) {

	//if (x < -45.0) return -1.0;
	//else if (x > 45.0) return 1.0;
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

/*
========================================================================
//TanhDeriv
========================================================================
*/
double TanhDeriv(double x) {
	// tanh derivative used in back propagation
	return 1.0 - x * x;
}

/*
========================================================================
//TanhFuncVecFunc
========================================================================
*/
double* TanhVecFunc(double wHx[], const int aSize) {

	double* arr = new double[aSize];

	for (intptr_t oN = 0; oN < (aSize - 1); oN++) //Skip bias
	{
		arr[oN] = TanHyp(wHx[oN]);
	}

	return arr;
}

/*
========================================================================
//TanhDerivVecFunc
========================================================================
*/
double* TanhDerivVecFunc(double inpV[], const int aSize) {

	double* arr = new double[aSize];

	for (intptr_t oN = 0; oN < (aSize - 1); oN++) //Skip bias
	{
		arr[oN] = TanhDeriv(inpV[oN]);
	}

	return arr;
}

