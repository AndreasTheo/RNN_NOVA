#pragma once
//#include "../../stdafx.h"
#include <iomanip>
#include <iostream>
#ifndef SOFTMAX_HEADER
#define SOFTMAX_HEADER

double sumA;

/*
========================================================================
//
========================================================================
*/
double SoftmaxDeriv(double x)
{
	return x * (1 - x); //speed of x - (x*x)?
}

/*
========================================================================
//
========================================================================
*/
double* SoftmaxVecFunc(double wHx[],const int aSize)
{
	//cout << "aSize:" << aSize << '\n';

	int oLS = aSize;
	//used to prevent arithmetic overflow
	double max = 0.0;
	for (intptr_t oN = 0; oN < (oLS); oN++)
	{
		if (wHx[oN] > max)
		{
			max = wHx[oN];
		}
	}

	//sum vector
	sumA = 0.0;
	for (intptr_t oN = 0; oN < (oLS); oN++)
	{
		sumA += exp(wHx[oN] - max);
	}

	double* rV = new double[oLS];
	//produces softmax with arithmetic overflow prevention
	for (intptr_t oN = 0; oN < (oLS); oN++)
	{
		rV[oN] = exp(wHx[oN] - max) / (sumA);
	}
	return rV;
}

/*
========================================================================
//
========================================================================
*/
double* SoftmaxDerivVecFunc(double inpV[], const int aSize)
{
	double* arr = new double[aSize];

	for (intptr_t oNi = 0; oNi < (aSize - 1); oNi++) //Skip bias
	{
		double sums = 0.0;
		for (intptr_t oNj = 0; oNj < (aSize - 1); oNj++) //Skip bias
		{
			if (oNj != oNi)
			{
				sums += -(inpV[oNi] * inpV[oNj]);
			}
			//cout << "sums:" << sums << '\n';
		}
		arr[oNi] = SoftmaxDeriv(inpV[oNi]) + sums;
	}

	return arr;
}




#endif