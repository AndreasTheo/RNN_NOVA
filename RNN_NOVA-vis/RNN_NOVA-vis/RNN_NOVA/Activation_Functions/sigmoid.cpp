#include <iomanip>
#include "sigmoid.h"


//range [0,1]
/*
========================================================================
//
========================================================================
*/
double Sigmoid(double x)
{

	return 1 / (1 + exp(-x));
}
/*
========================================================================
//
========================================================================
*/
double SigmoidDeriv(double x)
{
	return x * (1 - x);
}

//range [-1,1]
/*
========================================================================
//
========================================================================
*/
double SigmoidRangeOne(double x)
{
	return 2 / (1 + exp(-2 * x)) - 1;
}
double SigmoidRangeOneDeriv(double x)
{
	return 1 - (pow(x, 2));
}

/*
========================================================================
//
========================================================================
*/
double* SigmoidVecFunc(double wHx[], const int aSize)
{
	double* arr = new double[aSize];

	for (intptr_t oN = 0; oN < (aSize - 1); oN++) //Skip bias
	{
		arr[oN] = Sigmoid(wHx[oN]);
	}

	return arr;
}

/*
========================================================================
//
========================================================================
*/
double* SigmoidDerivVecFunc(double inpV[], const int aSize)
{
	double* arr = new double[aSize];

	for (intptr_t oN = 0; oN < (aSize - 1); oN++) //Skip bias
	{
		arr[oN] = SigmoidDeriv(inpV[oN]);
	}

	return arr;
}