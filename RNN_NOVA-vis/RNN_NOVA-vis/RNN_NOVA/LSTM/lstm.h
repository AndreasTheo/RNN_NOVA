

#pragma once
#ifndef LSTM_HEADER
#define LSTM_HEADER
#include "..\WeightMatrix\weightmatrix.h"

class LSTM
{

private:

public:
	void initNet();
	void FeedForward(double* input, int length);
	void BackProp(double* target);
	void NetInfo(bool readNetOutputs, bool readOutputGrads, bool readLSTMBlocks);
	void Run();
	void InitData();
	void UpdateWeights();
	void ResetdHNextGrads();

};









#endif