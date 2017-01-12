#pragma once
//#include "../../stdafx.h"
#include <iostream>
#include <immintrin.h> 
#include <memory>
#include <iomanip> 
#include "../Neuron/lstm_block.h"

using namespace std;

#ifndef PADDEDVECTOR_HEADER
#define PADDEDVECTOR_HEADER

template<class T>
class PaddedVector {

public:
	//||DEFAULTS||
	PaddedVector<T>(const PaddedVector&) = default;               // copy constructor
	PaddedVector<T>(PaddedVector&&) = default;                     // move constructor
	PaddedVector<T>& operator=(const PaddedVector&) & = default;  // copy assignment operator
	PaddedVector<T>& operator=(PaddedVector&&) & = default;       // move assignment operator
	virtual ~PaddedVector<T>(); // destructor
	PaddedVector<T>(const int); // constructor
	PaddedVector<T>(); // constructor

	//Vector Initialization Functions
	void InitVec(const intptr_t c)
	{
		//exception handling for a matrix of less than 1X1
		try {
			if (c < 1)
				throw 1;
		}
		catch (int err) {

			cerr << "An error has occurred!\n";
			if (err == 1)
			{
				cerr << "Matrix has invalid dimensionality: [rows < 1] or [cols < 1]. \n\n";
			}
		}
		_colsPadded = _cols = c;

		if (c % 2 != 0) {

			_colsPadded++;
			_isCPadded = true;
		}

		const intptr_t size = (_colsPadded); //2d addressing

		const int alignment = 16;

		try {

			_dataP = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment)); //dynamic array allocation
		}
		catch (int err) {

			cerr << "Failed to create dynamic matrix arrays \n";

		}
		ZerizeVector();

	}
	void ZerizeVector();

	//Data Access Functions
	T& operator[](const int c) const;
	T& operator[](intptr_t c) const;
	int size() const { return _cols; }
	int cols() const { return _cols; }
	int colsPadded() const { return _colsPadded; }

	//Informative Functions
	void PrintVector();
	//void Name(std::string n) { name = n; }

private:
	T* _dataP; //vector
	intptr_t _cols;
	intptr_t  _colsPadded;
	bool _isCPadded;
	//std::string name;
};

template<typename T>
inline T& PaddedVector<T>::operator[](const  int c) const {
	return _dataP[c];
}
template<typename T>
inline T& PaddedVector<T>::operator[](intptr_t c) const {
	return _dataP[c];
}
/*
========================================================================
//Constructor(s) & Destructor
========================================================================
*/
template<class T>
PaddedVector<T>::PaddedVector() {
}
template<class T>
PaddedVector<T>::PaddedVector(const int c) {
	InitVec(c);
}
template<class T>
PaddedVector<T>::~PaddedVector() {
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
inline void PaddedVector<T>::ZerizeVector() {
	try {
		for (intptr_t j = 0;j < _colsPadded;j++)
		{
			_dataP[j] = 0;
		}
	}
	catch (int err) {

		cerr << "Could not zerize matrix \n";

	}
}

template<>
inline void PaddedVector<LSTMBlock>::ZerizeVector() {

}

///*
//========================================================================
////Print Matrix Data
//========================================================================
//*/
template<class T>
inline void PaddedVector<T>::PrintVector() {
	cout << "Vector:" << '\n';
	for (intptr_t j = 0; j < _cols; j++) {

		cout << setw(5) << _dataP[j] << "  ";
	}
	cout << '\n';
}


#endif