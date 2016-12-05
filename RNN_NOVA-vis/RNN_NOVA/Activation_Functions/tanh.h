#pragma once
//#include "../../stdafx.h"
//#include <iomanip>

#ifndef TANH_HEADER
#define TANH_HEADER


//range [-1,1]
double TanHyp(double x);
double TanhDeriv(double x);

//range [-1,1]
double* TanhVecFunc(double wHx[], const int aSize);
double* TanhDerivVecFunc(double inpV[], const int aSize);
#endif
