#pragma once
//#include "../../stdafx.h"
//#include <iomanip>
#include <stdexcept>
#include <unordered_map>
#include "..\Loss_Functions\mse.h"
#include "..\Loss_Functions\cross_entropy.h"
#include "..\Activation_Functions\softmax.h"
#include "..\Activation_Functions\sigmoid.h"
#include "..\Activation_Functions\tanh.h"
#include "..\Math\gradient_checking.h"
#include "..\networkparameters.h"
#include <cmath>
#ifndef LFDIRECTORY_HEADER
#define LFDIRECTORY_HEADER

using namespace std;

bool gradCheck = 0;

const static std::unordered_map<std::string, int> string_to_case_LF {
	{ "mse",1 },
	{ "crossentropy",2 },
	{ "exponential",3 }
};

double LF(double* yV, const int aSize, double* targetV, std::string* lF_Type, std::string* prevAF_Type, bool regularize) {
	double error;
	switch (string_to_case_LF.at(*lF_Type)) {
	case 1:
	{
		error = MSE_Cost(yV, targetV, aSize);
	}
	break;
	case 2:
	{
		if (*prevAF_Type == "softmax") //multiclass
		{
			error = CrossEntropy_Multiclass_Cost(yV, targetV, aSize);
		}
		else
		{
			error = CrossEntropy_Binary_Cost(yV, targetV, aSize);
		}
	}
	break;
	}

	error = error / (aSize - 1);
	if (error > 0)
	{
		error = sqrt(error);
	}
	else
	{
		error = sqrt(-error);
	}
	return error;
}

void LFtoYDeriv(double* gradV, double* yV, const int aSize, double* targetV, std::string* lF_Type, std::string* prevAF_Type, bool regularize) {

	double* yDeriv = new double[aSize]; //Activation Function Derivative

	//regularization
	if (regularize)
	{
		//L1
		//L2
	}

	switch (string_to_case_LF.at(*lF_Type)) {
	case 1:
	{
		if (*prevAF_Type == "sigmoid")
		{
			MSE_DerivTo_Sigmoid_YGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else if (*prevAF_Type == "tanh")
		{
			MSE_DerivTo_Tanh_YGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else if (*prevAF_Type == "softmax")
		{
			yDeriv = SoftmaxDerivVecFunc(yV, aSize);
			MSE_DerivToYGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else
		{
			//error
		}


		#if defined(GRADCHECKING_HEADER) 
		{
			if (gradCheck)
			{
				double* yDerivTemp = new double[aSize];
					for (intptr_t i = 0; i < aSize;i++)
					{
						yDerivTemp[i] = 1;
					}
				MSE_DerivToYGRAD(gradV, yV, yDerivTemp, aSize, targetV);
				//VecGradApprox(yV, targetV, MSE_Cost, aSize);
			}
		}
		#endif

	}
	break;
	case 2:
	{
		if (*prevAF_Type == "sigmoid") //binary
		{
			CrossEntropy_Binary_DerivTo_Sigmoid_YGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else if (*prevAF_Type == "tanh")
		{
			yDeriv = TanhDerivVecFunc(yV, aSize);
			CrossEntropy_Binary_DerivToYGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else if (*prevAF_Type == "softmax" && (aSize>2)) //multiclass
		{
			yDeriv = SoftmaxDerivVecFunc(yV, aSize);
			CrossEntropy_Multiclass_DerivToYGRAD(gradV, yV, yDeriv, aSize, targetV);
		}
		else if (*prevAF_Type == "softmax") //multiclass
		{
			cout << "Softmax Multiclass requires more than one net output";
			throw std::exception("Softmax Multiclass requires more than one net output");
		}
		else
		{
			throw std::exception();
		}

		#if defined(GRADCHECKING_HEADER) 
		{
			if (gradCheck)
			{
				if (*prevAF_Type == "softmax")
				{
					//double* yDerivTemp = new double[aSize];
					//for (intptr_t i = 0; i < aSize;i++)
					//{
					//	yDerivTemp[i] = 1;
					//}
					//CrossEntropy_Multiclass_DerivToYGRAD(gradV, yV, yDerivTemp, aSize, targetV);
					//double* res = new double[aSize];
					//double* res2 = new double[aSize];
					//double* res3 = new double[aSize];
					////res = VecGradApprox(yV, targetV, CrossEntropy_Multiclass_Cost, aSize);

					//res2 = VecGradApprox(yV, SoftmaxVecFunc, aSize);
					////res3 = VecChainGrads(res, res2, aSize);
					//yDeriv = SoftmaxDerivVecFunc(yV, aSize);
					//VecCompareGrads(yDeriv, res2, aSize);
					////VecCompareGrads(gradV,res3,aSize);
				}
				else
				{
					double* yDerivTemp = new double[aSize];
					for (intptr_t i = 0; i < aSize;i++)
					{
						yDerivTemp[i] = 1;
					}
					CrossEntropy_Binary_DerivToYGRAD(gradV, yV, yDerivTemp, aSize, targetV);
					//VecGradApprox(yV, targetV, CrossEntropy_Binary_Cost, aSize);
				}
			}
		}
		#endif

	}
	break;
	}

}









#endif
