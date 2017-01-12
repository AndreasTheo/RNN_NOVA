//#include "../../stdafx.h"
#include "..\weightmatrix.h"
#include "..\..\parallelism_settings.h"


////#ifdef ALLOW_SSE
///*
//================================================
//Calculate Weights SSE
//================================================
//*/
//void WeightMatrix<double>::Calc_Weights_ADAGRAD(double prevVec[], const double grad[], const double* eta2) {
//
//	__m128d etaL = _mm_set_pd(*eta2, *eta2);
//	__m128d tenExpNegEight = _mm_set_pd(1e-8, 1e-8);
//
//	/* | update: hidden layer to input layer weights | */
//	for (intptr_t r = 0; r < _rows; r += 2)
//	{
//
//
//		__m128d gradL = _mm_set_pd(grad[r], grad[r]);
//		__m128d gradL_r2 = _mm_set_pd(grad[r + 1], grad[r + 1]);
//
//
//		for (intptr_t c = 0; c < _cols; c += 2)
//		{
//			{
//				// 1 ////////////////////
//				__m128d prevL = _mm_load_pd(prevVec + c);
//				__m128d dx = _mm_mul_pd(prevL, gradL);
//
//				//mat dWCache +=///////////////////////////
//				__m128d dWCacheTemp = _mm_mul_pd(dx, dx);
//				_dataP_dWCache[(r*_colsPadded) + c] += dWCacheTemp.m128d_f64[0];
//				_dataP_dWCache[(r*_colsPadded) + c + 1] += dWCacheTemp.m128d_f64[1];
//				__m128d dWCache = _mm_set_pd(_dataP_dWCache[(r*_colsPadded) + c], _dataP_dWCache[(r*_colsPadded) + c + 1]);
//				////////////////////////////////////////////////////////////////////////////////////////////////////////////
//				__m128d numerator = _mm_mul_pd(etaL, dx);
//				__m128d denominator = _mm_sqrt_pd(_mm_add_pd(tenExpNegEight, dWCache));
//				__m128d update3 = _mm_div_pd(numerator, denominator);
//
//				_dataP_Combo[(r*_colsPadded) + c] += update3.m128d_f64[0];
//				_dataP_Combo[(r*_colsPadded) + c + 1] += update3.m128d_f64[1];
//
//				// 2 ////////////////////
//				dx = _mm_mul_pd(prevL, gradL_r2);
//
//				//mat dWCache +=///////////////////////////
//				dWCacheTemp = _mm_mul_pd(dx, dx);
//				_dataP_dWCache[((r + 1)*_colsPadded) + c] += dWCacheTemp.m128d_f64[0];
//				_dataP_dWCache[((r + 1)*_colsPadded) + c + 1] += dWCacheTemp.m128d_f64[1];
//				dWCache = _mm_set_pd(_dataP_dWCache[((r + 1)*_colsPadded) + c], _dataP_dWCache[((r + 1)*_colsPadded) + c + 1]);
//				////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//				numerator = _mm_mul_pd(etaL, dx);
//				denominator = _mm_sqrt_pd(_mm_add_pd(tenExpNegEight, dWCache));
//				update3 = _mm_div_pd(numerator, denominator);
//
//				_dataP_Combo[((r + 1)*_colsPadded) + c] += update3.m128d_f64[0];
//				_dataP_Combo[((r + 1)*_colsPadded) + c + 1] += update3.m128d_f64[1];
//
//			}
//
//		}
//	}
//
//}

//#else


/*
================================================
Calculate Weights
================================================
*/
void WeightMatrix<double>::Calc_Weights_ADAGRAD(double prevVec[], const double grad[], const double* eta2) {

	/* | update: hidden layer to input layer weights | */
	for (intptr_t r = 0; r < (_rows); r++)
	{
		for (intptr_t c = 0; c < _cols; c++)
		{
			double dWCache = 0.0;
			double dx = prevVec[c] * grad[r];

			_dataP_dWCache[(r*_colsPadded) + c] += (dx*dx);

			dWCache = _dataP_dWCache[(r*_colsPadded) + c];

			_dataP_Combo[(r*_colsPadded) + c] += (*eta2 * dx) / ((sqrt)((dWCache)+1e-8));
		}
	}
}

//#endif

