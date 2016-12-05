#pragma once
//the cross_entropy requires its inputs to be interpreted as probabilities
#include <unordered_map>;
#include "..\networkparameters.h"
#include "..\Activation_Functions\softmax.h"
#include "..\Activation_Functions\sigmoid.h"
#include "..\Activation_Functions\tanh.h"

#ifndef CROSSENTROPY_HEADER
#define CROSSENTROPY_HEADER

//log likelihood
//multinomial distribution
//softmax
double CrossEntropy_Multiclass_Cost(double* y, double* t, const int aSize)
{
	double cost = 0;

	for (intptr_t i = 0; i < (aSize - 1);i++)
	{
		cost += t[i] * log(y[i]);
	}
	cost = -cost;

	return cost;
}


void CrossEntropy_Multiclass_DerivToYGRAD(double* gradV, double* y, double* aFDeriv, const int aSize, double* t) //softmax
{
	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{
		gradV[oN] = -(y[oN] - t[oN]);
		//gradV[oN] = -(t[oN] / y[oN]) * aFDeriv[oN];
		//gradV[oN] = -(t[oN] / y[oN]) * (y[oN] * (1-y[oN]));
	}
}




double CrossEntropy_Binary_Cost(double* y,double* t, const int aSize)
{
	double cost = 0;

	for (intptr_t i = 0; i < (aSize - 1);i++)
	{
		cost += t[i] * log(y[i]) + (1 - t[i]) * log(1 - y[i]);
	}
	cost = -cost;

	return cost;
}

void CrossEntropy_Binary_DerivToYGRAD(double* gradV, double* y, double* aFDeriv, const int aSize, double* t)
{
		for (intptr_t oN = 0; oN < (aSize - 1); oN++)
		{
			double delta = -((y[oN] - t[oN]) / (y[oN] * (1 - y[oN])));
			gradV[oN] = (delta) * aFDeriv[oN];
		}
}

void CrossEntropy_Binary_DerivTo_Sigmoid_YGRAD(double* gradV, double* y, double* aFDeriv, const int aSize, double* t)
{
	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{
		gradV[oN] = -(y[oN] - t[oN]);
	}
}

void CrossEntropy_Binary_DerivTo_Tanh_YGRAD(double* gradV, double* y, double* aFDeriv, const int aSize, double* t)
{
	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{
		double temp = y[oN] * (1 - y[oN]);
		gradV[oN] = (-t[oN] + temp) / temp;
	}
}


#endif