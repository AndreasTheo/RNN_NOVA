//#include "../../stdafx.h"
#include "weightmatrix.h"
#include "..\parallelism_settings.h"


#ifdef ALLOW_SSE
//FASTEST SSE
void FillColMultsVec(intptr_t* vec, intptr_t blockSize, intptr_t* cols)
{
	intptr_t colsPP = 0;

	for (intptr_t b = 0; b < blockSize; b++)
	{
		vec[b] = colsPP;
		colsPP += *cols;
	}
}
void M128DZeroVector(__m128d* accVector, intptr_t blockSize)
{

	accVector[0] = _mm_setzero_pd();

	//Fastest cpu instruction to set a registry to zero
	accVector[1] = _mm_xor_pd(accVector[0], accVector[0]);
	for (intptr_t z = 2; z < blockSize; z += 2)
	{
		accVector[z] = _mm_xor_pd(accVector[0], accVector[0]);
		accVector[z + 1] = _mm_xor_pd(accVector[0], accVector[0]);
	}
}
__m128d LoadMultPD(const __m128d* v1, const double* A, const intptr_t* c)
{
	__m128d	m1 = _mm_load_pd(A + *c);
	return _mm_mul_pd(*v1, m1);
}


double* WeightMatrix<double>::Mult_Mat_Vec_NxM_Mx1(double vec[]) {
	const int alignment = 16;
	double* res = static_cast<double*>(_aligned_malloc(_rowsPadded * sizeof(double), alignment)); //dynamic array allocation


	intptr_t matRowPos = 0;
	const intptr_t blockSize = 2;
	const intptr_t colTBS = _colsPadded*blockSize;

	intptr_t colMults[blockSize];
	FillColMultsVec(colMults, blockSize, &_colsPadded);
	__declspec(align(16)) __m128d accVec[blockSize];

	for (intptr_t r = 0; r < _rows; r += blockSize)
	{
		M128DZeroVector(accVec, blockSize);

		for (intptr_t c = 0; c < _cols; c += 2)
		{
			__m128d v1 = _mm_load_pd(vec + c);

			accVec[0] = _mm_add_pd(accVec[0], LoadMultPD(&v1, &_dataP[matRowPos], &c));
			accVec[1] = _mm_add_pd(accVec[1], LoadMultPD(&v1, &_dataP[matRowPos + colMults[1]], &c));

			for (intptr_t v = 2; v < blockSize; v += 2)
			{
				accVec[v] = _mm_add_pd(accVec[v], LoadMultPD(&v1, &_dataP[matRowPos + colMults[v]], &c));
				accVec[v + 1] = _mm_add_pd(accVec[v + 1], LoadMultPD(&v1, &_dataP[matRowPos + colMults[v + 1]], &c));
			}
		}
		res[r] = accVec[0].m128d_f64[0] + accVec[0].m128d_f64[1];
		res[r + 1] = accVec[1].m128d_f64[0] + accVec[1].m128d_f64[1];

		for (intptr_t z = 2; z < blockSize; z += 2)
		{
			res[r + z] = accVec[z].m128d_f64[0] + accVec[z].m128d_f64[1];
			res[r + z + 1] = accVec[z + 1].m128d_f64[0] + accVec[z + 1].m128d_f64[1];
		}
		matRowPos += colTBS;
	}
	return res;
}





#else
/*
=====================================================================================
//FeedForward Layer to Layer - multiply matrix NxM by vector Mx1
=====================================================================================
*/
double* WeightMatrix<double>::Mult_Mat_Vec_NxM_Mx1(double vec[]) {

	const int alignment = 16;
	double* res;
	res = static_cast<double*>(_aligned_malloc(_rowsPadded * sizeof(double), alignment)); //dynamic array allocation

	for (intptr_t r = 0; r < _rows; r++)
	{
		double acc = 0;
		for (intptr_t c = 0; c < _cols; c++)
		{
				acc += _dataP[(r*_colsPadded) + c] * vec[c];
		}

		res[r] = acc;
	}
	return res;
}

#endif // SSE



