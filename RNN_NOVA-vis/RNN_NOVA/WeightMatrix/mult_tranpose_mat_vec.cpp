//#include "../../stdafx.h"
#include "weightmatrix.h"

/*
======================================================================================
//Cache Oblivious Transpose
======================================================================================
*/
//http://users.cecs.anu.edu.au/~Alistair.Rendell/papers/coa.pdf
template<class T>
void WeightMatrix<T>::CacheOblivTranpose(const intptr_t rStart, const intptr_t rEnd, const intptr_t cStart, const intptr_t cEnd) {
	intptr_t r = rEnd - rStart, c = cEnd - cStart;
	if (r <= 16 && c <= 16) {
		for (intptr_t i = rStart; i < rEnd; i++) {
			for (intptr_t j = cStart; j < cEnd; j++) {
				_dataPT[j * _rows + i] = _dataP[i * _cols + j];
			}
		}
	}
	else
	{
		intptr_t rDivPoint = rStart + (r / 2);
		intptr_t cDivPoint = cStart + (c / 2);
		CacheOblivTranpose(rStart, rDivPoint, cStart, cDivPoint);
		CacheOblivTranpose(rStart, rDivPoint, cDivPoint, cEnd);
		CacheOblivTranpose(rDivPoint, rEnd, cStart, cDivPoint);
		CacheOblivTranpose(rDivPoint, rEnd, cDivPoint, cEnd);
	}
}

/*
======================================================================================
//Transpose Matrix multiplied by Vector
======================================================================================
*/
void WeightMatrix<double>::Mult_Tranpose_Mat_Vec(double vec[], double res[]) {

	CacheOblivTranpose(0, _rows, 0, _cols);

	for (intptr_t c = 0; c < _cols; c++)
	{
		double sum = 0.0;

		for (intptr_t r = 0; r < (_rows); r++)
		{
			sum += _dataPT[(c*_rowsPadded) + r] * vec[r];
		}
		res[c] = sum;
	}
}

