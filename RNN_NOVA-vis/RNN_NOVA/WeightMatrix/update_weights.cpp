//#include "../../stdafx.h"
#include "weightmatrix.h"


/*
======================================================================================
//Update Weights Matrix
======================================================================================
*/
void WeightMatrix<double>::Update_Weights(const int* layerSize) {

	for (intptr_t r= 0; r < _rows; r++)
	{
		for (intptr_t c = 0; c < _cols; c++)
		{
			//_dataP[(r*_colsPadded) + c] += ClipGradient((_dataP_Combo[(r*_colsPadded) + c]))/(*layerSize); 
			_dataP[(r*_colsPadded) + c] += ClipGradient((_dataP_Combo[(r*_colsPadded) + c]));
			//_dataP[(r*_colsPadded) + c] += (_dataP_Combo[(r*_colsPadded) + c]);
			_dataP_Combo[(r*_colsPadded) + c] = 0;
		}
	}
}


const double lClipV = -0.05;
const double rClipV = 0.05;
const bool clip = true;
/*
======================================================================================
//Clip Gradient
======================================================================================
*/
template<class T>
double WeightMatrix<T>::ClipGradient(double value) {

	double clippedV = value;
	if (clip)
	{
		if (clippedV < lClipV)
		{
			clippedV = lClipV;
		}
		else if (clippedV > rClipV)
		{
			clippedV = rClipV;
		}
	}

	return clippedV;
}
