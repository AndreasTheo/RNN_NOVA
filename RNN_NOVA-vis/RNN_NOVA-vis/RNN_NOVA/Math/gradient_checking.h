#pragma once
#ifndef GRADCHECKING_HEADER
#define GRADCHECKING_HEADER
#include <functional>
#include <iostream>
#include "..\Activation_Functions\softmax.h"


using namespace std::placeholders;

	
	double GradApprox(double x, function<double(double)> f)
{
	double res = (f(x + 1e-4) - f(x - 1e-4)) / (2.0 * 1e-4);
	return res;
}

	void VecCompareGrads(double grad[], double res[], const int aSize)
	{
		for (int i = 0; i < (aSize); i++)
		{
			cout << i << "- - - - - - - - - - - - - - " << '\n';
			cout << i << " - gradCheck: " << res[i] << '\n';
			cout << i << " - grad: " << grad[i] << '\n';
			cout << i << "- - - - - - - - - - - - - - " << '\n';
		}

	}

	double* VecChainGrads(double resA[], double resB[], const int aSize)
	{
		double* resC = new double[aSize];
		for (int i = 0; i < (aSize); i++)
		{
			resC[i] = resA[i] * resB[i];
		}

		return resC;
	}


	double* VecGradApprox(double x[], function<double*(double*,int)> f, const int aSize)
	{
		try {
			double e = 1e-4;
			double* tempPlus = new double[aSize];
			double* tempMinus = new double[aSize];
			double* res = new double[aSize];
			for (int i = 0; i < (aSize); i++)
			{
				for (int k = 0; k < aSize; k++)
				{
					// we send the whole vector to the function as f(x[k] = x + 1e-4) and f(x[k] = x - 1e-4) where k = i
					// incase the function (such as softmax) uses for example the vectors sum
					// when i = 0: f([(x + 1e-4),_,_,_,_]) & f([(x - 1e-4),_,_,_,_]); 
					// when i = 1: f([_,(x + 1e-4),_,_,_]) & f([_,(x - 1e-4),_,_,_]);
					if (k == i) 
					{
						tempPlus[k] = (x[k] + e);
						tempMinus[k] = (x[k] - e);
					}
					else
					{
						tempPlus[k] = (x[k]);
						tempMinus[k] = (x[k]);
					}
				}
				tempPlus = f(tempPlus,aSize);
				tempMinus = f(tempMinus,aSize);

				res[i] = (tempPlus[i] - tempMinus[i]) / (2.0 * 1e-4);
				
			}
			return res;
		}
		catch (int err) {
			//error
		}
	}

	//Overload with 3 function parameters for f used for loss functions
	double* VecGradApprox(double y[], double target[], function<double(double*, double*, int)> f, const int aSize)
	{
		try {
			double e = 1e-4;
			double* tempPlus = new double[aSize];
			double* tempMinus = new double[aSize];
			double* res = new double[aSize];
			for (int i = 0; i < (aSize); i++)
			{
				for (int k = 0; k < aSize; k++)
				{
					if (k == i)
					{
						tempPlus[k] = (y[k] + e);
						tempMinus[k] = (y[k] - e);
					}
					else
					{
						tempPlus[k] = (y[k]);
						tempMinus[k] = (y[k]);
					}
				}
				double res1 = f(tempPlus, target, aSize);
				double res2 = f(tempMinus, target, aSize);

				tempPlus = &res1;
				tempMinus = &res2;

				res[i] = (tempPlus[i] - tempMinus[i]) / (2.0 * 1e-4);
			}

			return res;
		}
			catch (int err) {
				//error
			}
	}

#endif