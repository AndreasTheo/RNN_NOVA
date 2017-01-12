#include "..\Data_Processing\data_processing.h"




double* DataProcessing::Normalize(double vec[],int length)
{

	double maxTemp = 0;
	double minTemp = 0;
	for (int i = 0; i < length; i++)
	{
		if (vec[i] > maxTemp)
		{
			maxTemp = vec[i];
		}
		else if (vec[i] < minTemp)
		{
			minTemp = vec[i];
		}

	}

	max = maxTemp;
	min = minTemp;

	double* normalized = new double[length];

	for ( int i = 0; i < length; i++ )
	{
		normalized[i] = (vec[i] - min) / (max - min);
	}

	return normalized;
}

double* DataProcessing::Denormalize(double vec[], int length)
{
	double* denormalized = new double[length];

	for (int i = 0; i < length; i++)
	{
		denormalized[i] = (vec[i] * (max - min) + min);
	}
	return denormalized;
}

