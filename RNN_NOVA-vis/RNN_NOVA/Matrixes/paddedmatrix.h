#pragma once
//#include "../../stdafx.h"
#include <iostream>
#include <immintrin.h> 
#include <memory>
#include <iomanip> 
#include "../Neuron/lstm_block.h"
#include <string>

using namespace std;

#ifndef PADDEDMATRIX_HEADER
#define PADDEDMATRIX_HEADER

template<class T>
class PaddedMatrix {

public:
	//||DEFAULTS||
	PaddedMatrix<T>(const PaddedMatrix&) = default;               // copy constructor
	PaddedMatrix<T>(PaddedMatrix&&) = default;                     // move constructor
	PaddedMatrix<T>& operator=(const PaddedMatrix&) & = default;  // copy assignment operator
	PaddedMatrix<T>& operator=(PaddedMatrix&&) & = default;       // move assignment operator
	PaddedMatrix<T>(const int, const int); // constructor
	PaddedMatrix<T>(); // constructor
	~PaddedMatrix<T>(); // destructor

	//Matrix Initialization Functions
	void InitMatrix(const intptr_t r, const intptr_t c)
	{
		//Exception Handling for dimensions less than 1x1
		try {

			if (r < 1 || c < 1)
				throw 1;
		}
		catch (int err) {

			cerr << "An error has occurred!\n";
			if (err == 1)
			{
				cerr << "Matrix has invalid dimensionality: [rows < 1] or [cols < 1]. \n\n";
			}
		}

		_rowsPadded = _rows = r;
		_colsPadded = _cols = c;

		if (r % 2 != 0) {

			_rowsPadded++;
			_isRPadded = true;
		}

		if (c % 2 != 0) {

			_colsPadded++;
			_isCPadded = true;
		}

		const intptr_t size = (_rowsPadded * _colsPadded); //2d addressing

		const int alignment = 16;

		try {

			_dataP = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment)); //dynamic array allocation
		}
		catch (int err) {

			cerr << "Failed to create dynamic matrix arrays \n";

		}
		ZerizeMatrix();


	}
	void ZerizeMatrix();

	//Data Access Functions
	T* operator[](const int r);
	T* operator[](intptr_t r);
	int size() const { return _rows * _cols; }
	int rows() const { return _rows; }
	int cols() const { return _cols; }
	int rowsPadded() const { return _rowsPadded; }
	int colsPadded() const { return _colsPadded; }

	//Calculation Functions
	void Mult_Tranpose_Mat_Vec(double vec[], double res[]);
	void CacheOblivTranpose(const int rb, const int re, const int cb, const int ce);

	//Informative Functions
	void PrintMatrix();
	//void Name(std::string n) { name = n; }


private:
	T* _dataP; //matrix
	intptr_t _rows, _cols;
	intptr_t _rowsPadded, _colsPadded;
	bool _isRPadded, _isCPadded;
	//std::string name;
};

template<typename T>
inline T* PaddedMatrix<T>::operator[](const int r) {
	return &_dataP[(r*_colsPadded)];
}
template<typename T>
inline T* PaddedMatrix<T>::operator[](intptr_t r) {
	return &_dataP[(r*_colsPadded)];
}

/*
========================================================================
//Constructor(s) & Destructor
========================================================================
*/
template<class T>
PaddedMatrix<T>::PaddedMatrix() {
}
template<class T>
PaddedMatrix<T>::PaddedMatrix(const int r, const int c) {
	InitMatrix(r, c);
}
template<class T>
PaddedMatrix<T>::~PaddedMatrix() {

	try {
		if (_dataP)
		{
			_aligned_free(_dataP);
		}
	}
	catch (int err) {
		cerr << "Failed to deconstruct matrixes \n";
		cerr << err << '\n';
	}
}
template<class T>
inline void PaddedMatrix<T>::ZerizeMatrix() {
	try {
		for (intptr_t i = 0; i < _rowsPadded; i++) {

			for (intptr_t j = 0;j < _colsPadded;j++)
			{
				_dataP[(i*_colsPadded) + j] = 0;
			}
		}
	}
	catch (int err) {

		cerr << "Could not zerize matrix \n";

	}
}

template<>
inline void PaddedMatrix<LSTMBlock>::ZerizeMatrix() {

}

///*
//========================================================================
////Print Matrix Data
//========================================================================
//*/
template<class T>
inline void PaddedMatrix<T>::PrintMatrix() {
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