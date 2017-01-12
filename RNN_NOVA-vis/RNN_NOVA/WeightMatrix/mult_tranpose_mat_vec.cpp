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
	int r = rEnd - rStart, c = cEnd - cStart;
	if (r <= 16 && c <= 16) {
		for (intptr_t i = rStart; i < rEnd; i++) {
			for (intptr_t j = cStart; j < cEnd; j++) {
				_dataPT[j * _rows + i] = _dataP[i * _cols + j];
			}
		}
	}
	else if (r >= c) {
		CacheOblivTranpose(rStart, rStart + (r / 2), cStart, cEnd);
		CacheOblivTranpose(rStart + (r / 2), rEnd, cStart, cEnd);
	}
	else {
		CacheOblivTranpose(rStart, rEnd, cStart, cStart + (c / 2));
		CacheOblivTranpose(rStart, rEnd, cStart + (c / 2), cEnd);
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

	/*for (intptr_t c = 0; c < _cols; c++)
	{
		double sum = 0.0;

		for (intptr_t r = 0; r < (_rows); r++)
		{
			sum += _dataP[(r*_rowsPadded) + c] * vec[r];
		}
		res[c] = sum;
	}*/
}

