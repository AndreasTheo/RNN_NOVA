//#include "../../stdafx.h"
#include "weightmatrix.h"







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

/*
======================================================================================
//FeedForward Layer to Layer - multiply matrix NxM by vector Mx1 (SSE)
======================================================================================
*/

double* WeightMatrix<double>::Mult_Mat_Vec_NxM_Mx1_SSE(double vec[]) {

	//if (((unsigned long)vec & 15) == 0) { //check alignment
	//__m128d	m1 = _mm_load_pd(_dataM[0]);

	const int alignment = 16;
	double* res;
	res = static_cast<double*>(_aligned_malloc(_rowsPadded * sizeof(double), alignment)); //dynamic array allocation

	for (intptr_t r = 0; r < unrollData.rWholeN; r += 4) {

		__m128d acc = _mm_setzero_pd();
		//Fastest cpu instruction to set a registry to zero
		__m128d acc_r2 = _mm_xor_pd(acc, acc);
		__m128d acc_r3 = _mm_xor_pd(acc, acc);
		__m128d acc_r4 = _mm_xor_pd(acc, acc);

		for (intptr_t c = 0; c < unrollData.cWholeN; c += 4) {

			__m128d v1 = _mm_load_pd(vec + c), v2 = _mm_load_pd(vec + c + 2);

			__m128d	m1 = _mm_load_pd(&_dataP[(r*_colsPadded) + c]), m2 = _mm_load_pd(&_dataP[(r*_colsPadded) + c + 2]);
			//1
			{
				acc = _mm_add_pd(acc,
					_mm_add_pd(_mm_mul_pd(v1, m1), _mm_mul_pd(v2, m2)));
			}
			m1 = _mm_load_pd(&_dataP[((r + 1)*_colsPadded) + c]); m2 = _mm_load_pd(&_dataP[((r + 1)*_colsPadded) + c + 2]);
			//2
			{
				acc_r2 = _mm_add_pd(acc_r2,
					_mm_add_pd(_mm_mul_pd(v1, m1), _mm_mul_pd(v2, m2)));
			}
			m1 = _mm_load_pd(&_dataP[((r + 2)*_colsPadded) + c]); m2 = _mm_load_pd(&_dataP[((r + 2)*_colsPadded) + c + 2]);
			//3
			{
				acc_r3 = _mm_add_pd(acc_r3,
					_mm_add_pd(_mm_mul_pd(v1, m1), _mm_mul_pd(v2, m2)));
			}
			m1 = _mm_load_pd(&_dataP[((r + 3)*_colsPadded) + c]); m2 = _mm_load_pd(&_dataP[((r + 3)*_colsPadded) + c + 2]);
			//4
			{
				acc_r4 = _mm_add_pd(acc_r4,
					_mm_add_pd(_mm_mul_pd(v1, m1), _mm_mul_pd(v2, m2)));
			}

		}

		////Remaining from cols if not a multiple of 4 (with padding can only be a multiple of 2)
		if (unrollData.cRemainder > 0) {
			int c = unrollData.cWholeN;
			__m128d v1 = _mm_load_pd(vec + c);
			__m128d	m1 = _mm_load_pd(&_dataP[(r*_colsPadded) + c]);
			//1
			{
				acc = _mm_add_pd(acc, _mm_mul_pd(v1, m1));
			}
			m1 = _mm_load_pd(&_dataP[((r + 1)*_colsPadded) + c]);
			//2
			{
				acc_r2 = _mm_add_pd(acc_r2, _mm_mul_pd(v1, m1));
			}
			m1 = _mm_load_pd(&_dataP[((r + 2)*_colsPadded) + c]);
			//3
			{
				acc_r3 = _mm_add_pd(acc_r3, _mm_mul_pd(v1, m1));
			}
			m1 = _mm_load_pd(&_dataP[((r + 3)*_colsPadded) + c]);
			//4
			{
				acc_r4 = _mm_add_pd(acc_r4, _mm_mul_pd(v1, m1));
			}
		}
		res[r] = acc.m128d_f64[0] + acc.m128d_f64[1];
		res[r + 1] = acc_r2.m128d_f64[0] + acc_r2.m128d_f64[1];
		res[r + 2] = acc_r3.m128d_f64[0] + acc_r3.m128d_f64[1];
		res[r + 3] = acc_r4.m128d_f64[0] + acc_r4.m128d_f64[1];

	}


	//Remaining from rows
	for (intptr_t r = unrollData.rWholeN; r < _rows; r++) {

		__m128d acc = _mm_setzero_pd();

		for (intptr_t c = 0; c < unrollData.cWholeN; c += 4) {

			__m128d v1 = _mm_load_pd(vec + c), v2 = _mm_load_pd(vec + c + 2);

			__m128d	m1 = _mm_load_pd(&_dataP[(r*_colsPadded) + c]), m2 = _mm_load_pd(&_dataP[(r*_colsPadded) + c + 2]);
			//1
			{
				acc = _mm_add_pd(acc,
					_mm_add_pd(_mm_mul_pd(v1, m1), _mm_mul_pd(v2, m2)));
			}
		}

		////Remaining from cols if not a multiple of 4 (with padding, can only be a multiple of 2)
		if (unrollData.cRemainder > 0) {

			int c = unrollData.cWholeN;
			__m128d v1 = _mm_load_pd(vec + c);
			__m128d	m1 = _mm_load_pd(&_dataP[(r*_colsPadded) + c]);
			//1
			{
				acc = _mm_add_pd(acc, _mm_mul_pd(v1, m1));
			}
		}

		res[r] = acc.m128d_f64[0] + acc.m128d_f64[1];

	}
	//}
	//else {

	//	cerr << "vector not aligned";

	//}
	return res;
}

