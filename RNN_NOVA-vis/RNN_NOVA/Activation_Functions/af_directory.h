#pragma once
//#include "../../stdafx.h"
//#include <iomanip>
//#include <unordered_map>;

#ifndef AFDIRECTORY_HEADER
#define AFDIRECTORY_HEADER

using namespace std;



double* AF(double* inpV, const int aSize, std::string* AF_Type) {
	const std::unordered_map<std::string, int> string_to_case_AF{
		{ "sigmoid",1 },
		{ "tanh",2 },
		{ "softmax",3 }
	};
	double* rV;

	switch (string_to_case_AF.at(*AF_Type)) {
	case 1:
	{
		rV = SigmoidVecFunc(inpV, aSize);
	}
	break;
	case 2:
	{
		rV = TanhVecFunc(inpV, aSize);
	}
	break;
	case 3:
	{
		rV = SoftmaxVecFunc(inpV, aSize);
	}
	break;
	default:
	{
		rV = new double[aSize];
	}
	}
	return rV;
}


#endif


