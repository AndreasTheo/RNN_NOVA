

#pragma once
#ifndef LSTM_HEADER
#define LSTM_HEADER
#include "..\WeightMatrix\weightmatrix.h"
#include "..\Matrixes\paddedmatrix.h"
#include "..\Matrixes\paddedvector.h"
#include <vector>
using std::vector;

class LSTM {

private:
	//Network Layer
	PaddedMatrix<double> iNOut;
	PaddedMatrix<LSTMBlock> lstmBlocks;
	PaddedMatrix<double> hNOut;
	PaddedMatrix<double> oNOut;
	//Weight Matrices
	//matrixes are in transpose order by default to benefit matrix loop caching 
	//(e.g 'hI(hL, iL)' is used for input to hidden layers on FeedForward
	// whilst the naming iH(iL,hL) would be more representative of the FeedForward direction)	
	WeightMatrix<double> oH;
	WeightMatrix<double> cellWX;
	WeightMatrix<double> cellWH;
	WeightMatrix<double> iGWX;
	WeightMatrix<double> iGWH;
	WeightMatrix<double> oGWX;
	WeightMatrix<double> oGWH;
	WeightMatrix<double> fGWX;
	WeightMatrix<double> fGWH;
	////Results (used temporarily)
	PaddedVector<double> res4;
	////Gradients
	PaddedMatrix<double> oNGrad;
	PaddedVector<double> dHGrad;
	////LSTMBlock gradients
	PaddedMatrix<double> hNGrad;
	PaddedVector<double> cell_Grad;
	PaddedVector<double> iG_Grad;
	PaddedVector<double> oG_Grad;
	PaddedVector<double> fG_Grad;
	////////////////////////////////////
	PaddedMatrix<double> inputData;
	PaddedMatrix<double> outputData;
	//////////////////////////////////
	struct Net_Params {
		int          epochs;
		double	     eta;
		long double	 alpha;
		int			 adaptGrad;
		long double	 rAESmoothing;
		double       wSV;
		int          weightInitType;
		std::string  yAFType;
		std::string  lFType;
	};

	Net_Params   net_Params;
	
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
	LSTM::LSTM() {}
	virtual LSTM::~LSTM() { cout << "LSTM Deconstructor"; };
};









#endif