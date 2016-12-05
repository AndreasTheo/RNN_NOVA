#pragma once
//#include "../../stdafx.h"
//#include <iomanip>

#ifndef SIGMOID_HEADER
#define SIGMOID_HEADER

 //range [0,1]
 double Sigmoid(double x);
 double SigmoidDeriv(double x);

 //range [-1,1]
 double SigmoidRangeOne(double x);
 double SigmoidRangeOneDeriv(double x);
 //range [0,1]
 double* SigmoidVecFunc(double wHx[], const int aSize);
 double* SigmoidDerivVecFunc(double inpV[], const int aSize);




#endif
