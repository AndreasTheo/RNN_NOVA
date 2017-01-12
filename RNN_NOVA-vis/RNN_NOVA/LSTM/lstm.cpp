#include <iostream>
#include <assert.h>
#include <iomanip>
#include "lstm.h"
#include "..\Neuron\lstm_block.h"
#include "..\Math\gradient_checking.h"
#include "..\Activation_Functions\softmax.h"
#include "..\Activation_Functions\sigmoid.h"
#include "..\Activation_Functions\tanh.h"
#include "..\networkparameters.h"
#include "..\Loss_Functions\lf_directory.h"
#include "..\Activation_Functions\af_directory.h"
#include "..\Data_Processing\data_processing.h"


using namespace std;

	int			 t = 1; //current network's time counter
	long double	 recentAvgError = 0.5; //start RAE at a neutral error value, 0 < RAE < 1; 
	const int    tS = 32*8; //timeSteps  (based on data & run params)
	bool         allowDHNext = false;
	bool         readNetOutputs = true;
	bool         readGrads = true;
	bool         readNetArray[3] = {true,false,false}; //entry 1: readNetOutputs, entry 2: readOutputGrads, entry 3:readLstmBlocks
	
	__declspec(align(16)) double* res3;
	__declspec(align(16)) double* cell_xInWx;
	__declspec(align(16)) double* cell_pHWh;
	__declspec(align(16)) double* iG_xInWx;
	__declspec(align(16)) double* iG_pHWh;
	__declspec(align(16)) double* oG_xInWx;
	__declspec(align(16)) double* oG_pHWh;
	__declspec(align(16)) double* fG_xInWx;
	__declspec(align(16)) double* fG_pHWh;
//Network layers
__declspec(align(16)) double errorAtT[tS] = { 0 };
//
DataProcessing dProcessing;



/*
================================================
Init Net
________________________________________________
eta - The learning rate, which controls how much the weights are adjusted at each update. 0 < eta < 1
alpha - The momentum, for updating the current weights in regard to the previous delta weights, 0 < alpha < 1
adaptGrad (weight update types): 0 for AdaGradient, 1 for Momentum Based.
wSV - Weight start value or max weight value for value initialization.
weightInitType: 0 - Random weight initalization, 1 - Weights are filled with a set value, 2 - Weights are loaded from directory.
yAFType (activation function at y layer): tanh/sigmoid/softmax
lFType (loss function at y layer): mse/crossentropy
================================================
*/
void LSTM::initNet(double eta ,double alpha ,int adaptGrad ,double rAESmoothing ,double wSV ,int weightInitType ,std::string yAFType ,std::string lFType) {
	try {
		//Network Layer
		iNOut.InitMatrix(tS, iL);
		lstmBlocks.InitMatrix(tS, hL);
		hNOut.InitMatrix(tS, hL);
		oNOut.InitMatrix(tS, oL);
		////Weight Matrices & Result Vectors
		oH.InitMatrix(oL, hL);	 
		cellWX.InitMatrix(hL, iL); 
		cellWH.InitMatrix(hL, hL);
		iGWX.InitMatrix(hL, iL);
		iGWH.InitMatrix(hL, hL); 
		oGWX.InitMatrix(hL, iL); 
		oGWH.InitMatrix(hL, hL);  
		fGWX.InitMatrix(hL, iL);
		fGWH.InitMatrix(hL, hL); 
		////////////////////////////////////////
		res4.InitVec(hL);
		////////////////////////////////////////
		//////Gradients
		oNGrad.InitMatrix(tS, oL);
		dHGrad.InitVec(hL);
		//////LSTMBlock gradients
		hNGrad.InitMatrix(tS, hL);
		cell_Grad.InitVec(hL);
		iG_Grad.InitVec(hL);
		oG_Grad.InitVec(hL);
		fG_Grad.InitVec(hL);




	//net parameters
	net_Params.eta = eta;
	net_Params.alpha = alpha;
	net_Params.adaptGrad = adaptGrad;
	net_Params.rAESmoothing = rAESmoothing;
	net_Params.wSV = wSV;
	net_Params.weightInitType = weightInitType;
	net_Params.yAFType = yAFType;
	net_Params.lFType = lFType;

	

		for (intptr_t cT = 0; cT < tS; cT++)
		{
			for (intptr_t hN = 0; hN < hL; hN++)
			{
				lstmBlocks[cT][hN].Init(oL, hN);
				hNOut[cT][hN] = 0;
				hNGrad[cT][hN] = 0;
			}
			for (intptr_t oN = 0; oN < oL; oN++)
			{
				oNGrad[cT][oN] = 0;
				oNOut[cT][oN] = 0;
			}

			//biases
			iNOut[cT][iL - 1] = 1;
			hNOut[cT][hL - 1] = 1;
			oNOut[cT][oL - 1] = 1;

		}
		//mats
		double min = 0.0;
		double max = net_Params.wSV;

		if (net_Params.weightInitType == 0)
		{
			oH.RandomizeWeights(&min, &max);
			cellWX.RandomizeWeights(&min, &max);
			cellWH.RandomizeWeights(&min, &max);
			iGWX.RandomizeWeights(&min, &max);
			iGWH.RandomizeWeights(&min, &max);
			oGWX.RandomizeWeights(&min, &max);
			oGWH.RandomizeWeights(&min, &max);
			fGWX.RandomizeWeights(&min, &max);
			fGWH.RandomizeWeights(&min, &max);
		}
		else if (net_Params.weightInitType == 1)
		{
			double wSV = net_Params.wSV;
			oH.FillWithConstants(&wSV);
			cellWX.FillWithConstants(&wSV);
			cellWH.FillWithConstants(&wSV);
			iGWX.FillWithConstants(&wSV);
			iGWH.FillWithConstants(&wSV);
			oGWX.FillWithConstants(&wSV);
			oGWH.FillWithConstants(&wSV);
			fGWX.FillWithConstants(&wSV);
			fGWH.FillWithConstants(&wSV);
		}

	}
	catch (int err) {
		cerr << "Failed to initialize network \n";
		cerr << err << '\n';
	}
}


/*
==================================================
FEED FORWARD

//      0--->>--0-->>--0 T at n
//      |       ^      |
//      0--->>--0-->>--0 T at 2
//      |       ^      |
//      0--->>--0-->>--0 T at 1
//      |       ^      |
//      0--->>--0-->>--0 T at 0
==================================================
*/
void LSTM::FeedForward(double* input, int length) {

	assert(iL == (length + 1));

	for (intptr_t i = 0; i < (iL - 1); i++) //Skip bias
	{
		iNOut[t][i] = input[i];
	}

	//Input & Context Layers to LSTMBlock Connections
	cell_xInWx = cellWX.Mult_Mat_Vec_NxM_Mx1(iNOut[t]);
	cell_pHWh  = cellWH.Mult_Mat_Vec_NxM_Mx1(hNOut[t - 1]);
	iG_xInWx   = iGWX.Mult_Mat_Vec_NxM_Mx1(iNOut[t]);
	iG_pHWh    = iGWH.Mult_Mat_Vec_NxM_Mx1(hNOut[t - 1]);
	oG_xInWx   = oGWX.Mult_Mat_Vec_NxM_Mx1(iNOut[t]);
	oG_pHWh    = oGWH.Mult_Mat_Vec_NxM_Mx1(hNOut[t - 1]);
	fG_xInWx   = fGWX.Mult_Mat_Vec_NxM_Mx1(iNOut[t]);
	fG_pHWh    = fGWH.Mult_Mat_Vec_NxM_Mx1(hNOut[t - 1]);

	//LSTMBlock Internals
	for (intptr_t hN = 0; hN < (hL-1); hN++) //Skip bias
	{
		double cell_xInWx_pHWh = (cell_xInWx[hN] + cell_pHWh[hN]);
		double iG_xInWx_pHWh = (iG_xInWx[hN] + iG_pHWh[hN]);
		double oG_xInWx_pHWh = (oG_xInWx[hN] + oG_pHWh[hN]);
		double fG_xInWx_pHWh = (fG_xInWx[hN] + fG_pHWh[hN]);

		hNOut[t][hN] = lstmBlocks[t][hN].LSTM_FeedForward(cell_xInWx_pHWh,iG_xInWx_pHWh, oG_xInWx_pHWh, fG_xInWx_pHWh, lstmBlocks[t-1][hN].cell);
	}
	//LSTMBlock to Output layer Connections
	res3 = oH.Mult_Mat_Vec_NxM_Mx1(hNOut[t]);
	double* oHRes = AF(res3, oL, &net_Params.yAFType);
	for (intptr_t n = 0; n < (oL); n++)
	{
		oNOut[t][n] = oHRes[n];
	}
	free(oHRes);
	_aligned_free(res3);
	_aligned_free(cell_xInWx);
	_aligned_free(cell_pHWh);
	_aligned_free(iG_xInWx);
	_aligned_free(iG_pHWh);
	_aligned_free(oG_xInWx);
	_aligned_free(oG_pHWh);
	_aligned_free(fG_xInWx);
	_aligned_free(fG_pHWh);
}


/*
========================
BACKPROPAGATION

//BPTT Method To Predict Next Input
//y[t] = (probabilities for next input)
//      0-0------<--<-<--0( <-Error at T  (after this step we keep the dhNext grads and add it to the grad of the corrisponding hLayer node when propagating through at the Error T-1)
//      | |              |
//      0-0------<--<-<--0 <-Error at T-1 (repeat above process until the bottom)
//      | |              |
//      0-0------<--<-<--0 <-Error at T-2
//      | |              |
//      0-0------<--<-<--0 <-Error at T-3 (update shared weights when the bottom is reached)
========================
*/
void LSTM::BackProp(double* target) {

	//Get Error and output Grad(s)
	errorAtT[t] = LF(oNOut[t], oL, target, &net_Params.lFType, &net_Params.yAFType, 0);
	LFtoYDeriv(oNGrad[t], oNOut[t], oL,target, &net_Params.lFType, &net_Params.yAFType,0);
	//Output Grad to hidden Grad
	oH.Mult_Tranpose_Mat_Vec(oNGrad[t], &res4[0]);

	//Propagate through lstm block with hidden grad to obtain (c_o_i_f) grads
	for (intptr_t hN = 0; hN < hL; hN++)
	{
		lstmBlocks[t][hN].LSTM_BackPropagate(res4[hN]);

		cell_Grad[hN] = lstmBlocks[t][hN].cell_Grad;
		oG_Grad[hN] = lstmBlocks[t][hN].oG_Grad;
		iG_Grad[hN] = lstmBlocks[t][hN].iG_Grad;
		fG_Grad[hN] = lstmBlocks[t][hN].fG_Grad;		
	}

	double eta = net_Params.eta;
	//Calc: output layer to hidden layer weights
	oH.Calc_Weights_ADAGRAD(hNOut[t], oNGrad[t], &eta);

	//Calc: hidden layer to (output layer weights) & (context layer weights)
	cellWX.Calc_Weights_ADAGRAD(iNOut[t],&cell_Grad[0], &eta);
	cellWH.Calc_Weights_ADAGRAD(hNOut[t - 1], &cell_Grad[0], &eta);
	iGWX.Calc_Weights_ADAGRAD(iNOut[t], &iG_Grad[0], &eta);
	iGWH.Calc_Weights_ADAGRAD(hNOut[t - 1], &iG_Grad[0], &eta);
	oGWX.Calc_Weights_ADAGRAD(iNOut[t], &oG_Grad[0], &eta);
	oGWH.Calc_Weights_ADAGRAD(hNOut[t - 1], &oG_Grad[0], &eta);
	fGWX.Calc_Weights_ADAGRAD(iNOut[t], &fG_Grad[0], &eta);
	fGWH.Calc_Weights_ADAGRAD(hNOut[t - 1], &fG_Grad[0], &eta);

	if (allowDHNext)
	{
		//dHNext gradients
		{
			cellWH.Mult_Tranpose_Mat_Vec(&cell_Grad[0], &res4[0]);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHCell_Grad = res4[hN];
			}
			iGWH.Mult_Tranpose_Mat_Vec(&iG_Grad[0], &res4[0]);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHIG_Grad = res4[hN];
			}
			oGWH.Mult_Tranpose_Mat_Vec(&oG_Grad[0], &res4[0]);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHOG_Grad = res4[hN];
			}
			fGWH.Mult_Tranpose_Mat_Vec(&fG_Grad[0], &res4[0]);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHFG_Grad = res4[hN];
			}
		}
	}
}


/*
================================================
Net Info
================================================
*/
void LSTM::NetInfo(bool readNetOutputs, bool readOutputGrads,bool readLSTMBlocks) {

	try {
		cout << "__________________________________________________" << '\n';

		if (readNetOutputs)
		{
			cout << '\n' << "o layer outputs: " << '\n';
			for (intptr_t cT = 0; cT < tS; cT++)
			{
				cout << "curTime: " << cT << " ------------------" << '\n';
				for (int i = 0; i < (oL - 1);i++)
				{
					cout << "target: " << std::setprecision(17) << outputData[cT][i] << '\n';
					cout << "output: " << std::setprecision(17) << oNOut[cT][i] << '\n';
					cout << "error: " << std::setprecision(17) << errorAtT[cT] << '\n';

					cout << "__" << '\n';
				}
			}
		}
		if (readOutputGrads)
		{
				cout << '\n' << "o layer Grads: " << '\n';
			for (int i = 0; i < (oL - 1);i++)
			{

				for (intptr_t cT = 0; cT < tS; cT++)
				{
					cout << "curTime: " << cT << " ------------------" << '\n';
					cout << "oGrad: " << std::setprecision(17) << oNGrad[cT][i] << '\n';
					cout << '\n';
				}

				cout << "__" << '\n';
			}
		}

		cout << "__________________________________________________" << '\n';
	}
	catch (int err) {
		cerr << "Failed to produce network information! \n";
		cerr << err << '\n';
	}


}


/*
================================================
Update Weights
================================================
*/
void LSTM::UpdateWeights() {

	oH.Update_Weights(&oL);
	cellWX.Update_Weights(&hL);
	cellWH.Update_Weights(&hL);
	iGWX.Update_Weights(&hL);
	iGWH.Update_Weights(&hL);
	oGWX.Update_Weights(&hL);
	oGWH.Update_Weights(&hL);
	fGWX.Update_Weights(&hL);
	fGWH.Update_Weights(&hL);

}



/*
================================================
ResetdHNextGrads
================================================
*/
void LSTM::ResetdHNextGrads() {

	for (intptr_t hN = 0; hN < (hL); hN++)
	{
		dHGrad[hN] = 0;
	}
}


/*
================================================
Run
================================================
*/
//#pragma unmanaged(push,off) //To prevent the code from running as managed code when called from clr (used for graphs).
void LSTM::Run(int epochs) {

	//cout << "Running net: " << '\n';

	//clock_t t1, t2;
	//t1 = clock();
	//Process
		for (intptr_t i = 0; i < epochs; i++)
		{
		/*	if (i > epochs - 2)
			{
				gradCheck = true;
			}*/

			for (int k = 1; k < tS; k++)
			{
				t = k;
				FeedForward(inputData[t],(iL - 1));
			}
			for (intptr_t k = (tS-1); k > 0; k--)
			{
				t = k;
				BackProp(outputData[t]);
			}
			UpdateWeights();
			ResetdHNextGrads();
		}
	//NetInfo(readNetArray[0], readNetArray[1], readNetArray[2]);
	//t2 = clock();
	//double diff((double)t2 - (double)t1);
	//cout << "time: " << (diff / (double)CLOCKS_PER_SEC) << endl;
		
}

/*
================================================
Error Data
================================================
*/
double LSTM::VisNetErrorData() {

		double errorSum = 0.0;
		for (int t = 0; t < (tS); t++)
		{
			errorSum += errorAtT[t];
		}
		errorSum = (errorSum / tS);
	return errorSum;
}

/*
================================================
Net Output Data
================================================
*/
vector<vector<double>> LSTM::VisNetOutputData() {

	vector<vector<double>> ary (tS,vector<double>(oL));

	for (int i = 0; i < tS; ++i)
	{
		for (int j = 0; j < oL; ++j)
		{
			ary[i][j] = oNOut[i][j];
		}
	}
	return ary;
}
/*
================================================
Output Data
================================================
*/
vector<vector<double>> LSTM::VisOutputData() {

	vector<vector<double>> ary(tS, vector<double>(oL));

	for (int i = 0; i < tS; ++i)
	{
		for (int j = 0; j < oL; ++j)
		{
			ary[i][j] = outputData[i][j];
		}
	}
	return ary;
}
/*
================================================
Data
================================================
*/
void LSTM::InitData() {
	//Data
	inputData.InitMatrix(tS, iL);
	outputData.InitMatrix(tS, oL - 1);

	double* iWave = new double[tS];
	double* oWave = new double[tS];
	//Sinewave Dataset
	for (intptr_t i = 0; i < (tS); i++)
	{

		iWave[i] = cos(3.14159/8 * i);
		double noise = rand() * 0.00075;
		oWave[i] = sin((3.14159 / (tS/8) * i) + noise );
		//oWave[i] = noise;
	}
	double* iWaveRes = dProcessing.Normalize(iWave, tS);
	double* oWaveRes = dProcessing.Normalize(oWave, tS);
	for (intptr_t i = 0; i < (tS); i++)
	{
		inputData[i][0] = iWaveRes[i];
		outputData[i][0] = oWaveRes[i];
	}
	//classification
	//
	//
	//

	for (intptr_t cT = 0; cT < tS; cT++)
	{
		cout << "target: " << std::setprecision(17) << outputData[cT][0] << '\n';
		cout << '\n';
	}
	delete[] iWave;
	delete[] oWave;
	delete[] iWaveRes;
	delete[] oWaveRes;
}