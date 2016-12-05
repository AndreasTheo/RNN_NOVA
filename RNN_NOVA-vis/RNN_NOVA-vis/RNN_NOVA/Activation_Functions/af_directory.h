#pragma once
//#include "../../stdafx.h"
//#include <iomanip>
#include <unordered_map>;

#ifndef AFDIRECTORY_HEADER
#define AFDIRECTORY_HEADER

using namespace std;

const static std::unordered_map<std::string, int> string_to_case_AF{
	{ "sigmoid",1 },
	{ "tanh",2 },
	{ "softmax",3 }
};


void AF(double* outV, double* inpV, const int aSize, std::string* AF_Type)
{
	//cout << "size: " << aSize << '\n';
	double* rV = new double[aSize];

	switch (string_to_case_AF.at(*AF_Type)) {
	case 1:
	{
		//cout << "sig";
		rV = SigmoidVecFunc(inpV, aSize);
	}
	break;
	case 2:
	{
		//rV = SigmoidVecFunc(inpV, aSize);
		//cout << "tanh";
		rV = TanhVecFunc(inpV, aSize);
	}
	break;
	case 3:
	{
		rV = SoftmaxVecFunc(inpV, aSize);
	}
	break;
	}

	for (intptr_t n = 0; n < (aSize); n++)
	{
		outV[n] = rV[n];
	}
}


#endif


