//#include "../../stdafx.h"
#include "weightmatrix.h"

/*
======================================================================================
//Cache Oblivious Transpose
======================================================================================
*/
//http://users.cecs.anu.edu.au/~Alistair.Rendell/papers/coa.pdf
template<class T>
void WeightMatrix<T>::CacheOblivTranpose(const int rowb, const int rowe, const int colb, const int cole) {
	int r = rowe - rowb, c = cole - colb;
	if (r <= 16 && c <= 16) {
		for (intptr_t i = rowb; i < rowe; i++) {
			for (intptr_t j = colb; j < cole; j++) {
				_dataPT[j * _rows + i] = _dataP[i * _cols + j];
			}
		}
	}
	else if (r >= c) {
		CacheOblivTranpose(rowb, rowb + (r / 2), colb, cole);
		CacheOblivTranpose(rowb + (r / 2), rowe, colb, cole);
	}
	else {
		CacheOblivTranpose(rowb, rowe, colb, colb + (c / 2));
		CacheOblivTranpose(rowb, rowe, colb + (c / 2), cole);
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

