#pragma once
//#include "../../stdafx.h"
#include <iomanip>
#include <iostream>
#ifndef SOFTMAX_HEADER
#define SOFTMAX_HEADER

double SoftmaxDeriv(double x);
double* SoftmaxVecFunc(double wHx[], const int aSize);
double* SoftmaxDerivVecFunc(double inpV[], const int aSize);

#endif