

#pragma once
#ifndef LSTM_HEADER
#define LSTM_HEADER
#include "..\WeightMatrix\weightmatrix.h"
#include <vector>
using std::vector;

class LSTM {

private:

public:
	void initNet(double eta ,double alpha ,int adaptGrad ,double rAESmoothing ,double wSV ,int weightInitType ,std::string yAFType ,std::string lFType);
	void FeedForward(double* input, int length);
	void BackProp(double* target);
	void NetInfo(bool readNetOutputs, bool readOutputGrads, bool readLSTMBlocks);
	void Run(int epochs);
	void InitData();
	void UpdateWeights();
	void ResetdHNextGrads();
	double VisNetErrorData();
	vector<vector<double>> VisNetOutputData();
	vector<vector<double>> VisOutputData();
};









#endif