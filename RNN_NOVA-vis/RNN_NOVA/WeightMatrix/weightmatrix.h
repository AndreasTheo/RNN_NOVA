#pragma once
//#include "../../stdafx.h"
#include <iostream>
#include <immintrin.h> 
#include <memory>
#include "..\Matrixes\paddedvector.h"

using namespace std;

#ifndef WEIGHTMATRIX_HEADER
#define WEIGHTMATRIX_HEADER

template<class T>
class WeightMatrix {

public:
	//||DEFAULTS||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	WeightMatrix<T>(const WeightMatrix&) = default;               // copy constructor
	WeightMatrix<T>(WeightMatrix&&) = default;                     // move constructor
	WeightMatrix<T>& operator=(const WeightMatrix&) & = default;  // copy assignment operator
	WeightMatrix<T>& operator=(WeightMatrix&&) & = default;       // move assignment operator
	virtual ~WeightMatrix<T>() {
		try {
			if (_dataP)
			{
				_aligned_free(_dataP);
			}
			if (_dataP_dWCache)
			{
				_aligned_free(_dataP_dWCache);
			}
			if (_dataP_Combo)
			{
				_aligned_free(_dataP_Combo);
			}
			if (_dataPT)
			{
				_aligned_free(_dataPT);
			}
		}
		catch (int err) {
			cerr << "Failed to deconstruct weightmatrixes \n";
			cerr << err << '\n';
		}

	} // destructor
	WeightMatrix<T>(const int, const int); // constructor
	WeightMatrix<T>();
	//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

	int size() const { return _rows * _cols; }
	int rows() const { return _rows; }
	int cols() const { return _cols; }
	int rowsPadded() const { return _rowsPadded; }
	int colsPadded() const { return _colsPadded; }

	//Matrix Initialization Function
	void InitMatrix(const intptr_t r, const intptr_t c)
	{
		//exception handling for a matrix of less than 1X1
		try {

			if (r < 1 || c < 1)
				throw 1;
		}
		catch (int err) {

			cerr << "An error has occurred!\n";
			if (err == 1)
			{
				cerr << "Weight Matrix has invalid dimensionality: [rows < 1] or [cols < 1]. \n\n";
			}
		}

		_rowsPadded = _rows = r;
		_colsPadded = _cols = c;

		if (r % 2 != 0) {

			_rowsPadded++;

		}

		if (c % 2 != 0) {

			_colsPadded++;

		}

		const intptr_t size = (_rowsPadded * _colsPadded); //2d addressing

		const int alignment = 16;

		try {

			_dataP = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment)); //dynamic array allocation
			_dataP_dWCache = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment));
			_dataP_Combo = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment));
			_dataPT = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment));
		}
		catch (int err) {

			cerr << "Failed to create dynamic weightmatrix arrays \n";

		}

		for (intptr_t i = 0; i < _rowsPadded; i++) {

			for (intptr_t j = 0;j < _colsPadded;j++)
			{
				_dataP[(i*_colsPadded) + j] = 0;
				_dataP_dWCache[(i*_colsPadded) + j] = 0;
				_dataP_Combo[(i*_colsPadded) + j] = 0;
				_dataPT[(i*_colsPadded) + j] = 0;
			}
		}


		unrollOrganizer(&_rowsPadded, &_colsPadded, 4); //- eventually with a cpu benchmark optimizer
	}
	//Weight Initialization Functions
	void RandomizeWeights(const double* min, const double* max);
	void FillWithConstants(const double* constant);
	//void Name(std::string n) { name = n; }

	//Data Access Functions
	T& operator()(const int r, const  int c);
	T operator()(const int r, const  int c) const;
	void PrintMatrix();

	//Calculation Functions
	void unrollOrganizer(const intptr_t* r, const intptr_t* c, const intptr_t unrollAmount);
	double* Mult_Mat_Vec_NxM_Mx1(double vec[]);
	void Calc_Weights_ADAGRAD(double prevVec[], const double grad[], const double* eta2);
	void Mult_Tranpose_Mat_Vec(double vec[], double res[]);
	void CacheOblivTranpose(const intptr_t rb, const intptr_t re, const intptr_t cb, const intptr_t ce);
	void Update_Weights(const int* layerSize);
	double ClipGradient(double value);

	//storage and info

private:
	T* _dataP; //weight matrix
	T* _dataP_dWCache;//Used in weight backprop calculations (ADAGRAD)
	T* _dataP_Combo; //combination of weight additions (summed throughout time t for rnns)
	T* _dataPT; //transpose weight matrix
	intptr_t _rows, _cols;
	intptr_t _rowsPadded, _colsPadded;
	bool _isRPadded, _isCPadded;
	//std::string name;

	struct UnrollDataContainer {

		int rWholeN;
		int cWholeN;
		int rRemainder;
		int cRemainder;

	};

	UnrollDataContainer unrollData;
};

template<typename T>
inline T& WeightMatrix<T>::operator()(const int r, const int c) {
	return _dataP[(r*_colsPadded) + c];
}

template<typename T>
inline T WeightMatrix<T>::operator()(const int r, const int c) const {
	return _dataP[(r*_colsPadded) + c];
}


/*
========================================================================
//Constructor
========================================================================
*/
template<class T>
WeightMatrix<T>::WeightMatrix() {
}
template<class T>
WeightMatrix<T>::WeightMatrix(const int r, const int c) {
	InitMatrix(r, c);
}

/*
========================================================================
//Randomizes a value where 0<value<1
========================================================================
*/
inline double fRand(const double fMin, const double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

/*
========================================================================================
//Gets a randomized value foreach weight throughout the weight matrix where 0<value<1
========================================================================================
*/
template<class T>
inline void WeightMatrix<T>::RandomizeWeights(const double* min, const double* max) {

	try
	{
		for (intptr_t i = 0; i < _rows; i++)
		{
			for (intptr_t j = 0;j < _cols;j++)
			{
				_dataP[(i*_colsPadded) + j] = fRand(*min, *max);
			}
		}
	}
	catch (int err)
	{
		cerr << "Failed to randomize weightmatrix values\n";
	}
}


/*
========================================================================
//Fill array with parameter value
========================================================================
*/
template<class T>
inline void WeightMatrix<T>::FillWithConstants(const double* value) {

	try
	{
		for (intptr_t i = 0; i < _rows; i++)
		{
			for (intptr_t j = 0;j < _cols;j++)
			{
				_dataP[(i*_colsPadded) + j] = *value;
			}
		}
	}
	catch (int err)
	{
		cerr << "Failed to fill weight matrix with constant value.\n\n";
	}



}


/*
================================================
unrollOrganizer
================================================
*/
template<class T>
inline void WeightMatrix<T>::unrollOrganizer(const intptr_t* r, const  intptr_t* c, const intptr_t unrollAmount) {

	unrollData.rWholeN = (*r / unrollAmount) * unrollAmount;
	unrollData.rRemainder = *r - unrollData.rWholeN;
	unrollData.cWholeN = (*c / unrollAmount) * unrollAmount;
	unrollData.cRemainder = *c - unrollData.cWholeN;

}
///*
//========================================================================
////Print Matrix Data
//========================================================================
//*/
template<class T>
inline void WeightMatrix<T>::PrintMatrix() {
	cout << "Matrix:" << '\n';
	for (intptr_t i = 0; i < _rows; i++) {
		for (intptr_t j = 0; j < _cols; j++) {

			cout << setw(5) << _dataP[i * _colsPadded + j] << "  ";
		}
		cout << '\n';
	}
	cout << '\n';
}




#endif