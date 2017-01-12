#pragma once
#ifndef DATAPROCESSING_HEADER
#define DATAPROCESSING_HEADER

class DataProcessing {

private:

public:
	double* Normalize(double vec[], int length);
	double* Denormalize(double vec[], int length);
	double Max(double vec[], int length);
	double Min(double vec[], int length);
	double max;
	double min;
};


#endif
