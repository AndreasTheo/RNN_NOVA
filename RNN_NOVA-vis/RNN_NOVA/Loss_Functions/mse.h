#pragma once
//#include "../../stdafx.h"
//#include <iomanip>
//#include <unordered_map>;
#include "..\networkparameters.h"


#ifndef SQUAREDERROR_HEADER
#define SQUAREDERROR_HEADER



double MSE_Cost(double* yV, double* tV, const int aSize) {

	double cost = 0;
	for (intptr_t oN = 0; oN < (aSize - 1); oN++)// We skip the delta calc for the bias
	{  
		cost += ( (tV[oN] - yV[oN]) * (tV[oN] - yV[oN]) ) / 2;
	}
	cost = -cost;

	return cost;
}


void MSE_DerivToYGRAD(double* gradV, double* y,double* aFDeriv, const int aSize, double* t) {

	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{   //cost(y, t) = 1/2sum(t-y)^2 - although given that deriv of the cost = sum(t-y) we can 
		//simply just calculate the delta term without the cost func.
		double delta = (t[oN] - y[oN]);

			gradV[oN] = (delta * aFDeriv[oN]);

	}
							

}
void MSE_DerivTo_Tanh_YGRAD(double* gradV, double* y, const int aSize, double* t) {
	
	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{
		gradV[oN] = (t[oN] - y[oN]) * (1 - (y[oN] * y[oN]));
	}
}

void MSE_DerivTo_Sigmoid_YGRAD(double* gradV, double* y, const int aSize, double* t) {

	for (intptr_t oN = 0; oN < (aSize - 1); oN++)
	{
		gradV[oN] = (t[oN] - y[oN]) * (y[oN] * (1.0 - y[oN]));
	}
}






#endif