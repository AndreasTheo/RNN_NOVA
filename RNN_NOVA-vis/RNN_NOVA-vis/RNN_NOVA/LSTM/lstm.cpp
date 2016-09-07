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

using namespace std;

int t = 1;

typedef struct Net_Params //TODO: make use of the struct net param and move to networkparameters.h
{
	const int tS;
	int epochs;
	double	eta;
	long double	alpha;
	int			adaptGrad;
	long double	recentAvgError;
	long double	rAESmoothingFactor;
	const double wSV;
	int weightInitType;

} Net_Params;

//
	const int    tS = 32; //time steps
	int          epochs = 10000;
	double	     eta = 0.01; //The learning rate, which controls how much the weights are adjusted at each update. 0 < eta < 1
	long double	 alpha = 0.5;   //The momentum, for updating the current weights in regard to the previous delta weights, 0 < alpha < 1
	int			 adaptGrad = 0; //Weight update types: 0 for AdaGradient, 1 for Momentum Based.
	long double	 recentAvgError = 0.5; //start RAE at a neutral error value, 0 < RAE < 1; 
	long double	 rAESmoothingFactor = 100.0;
	const double wSV = 0.005; //weight start value or max weight value for value initialization
	double seed;
	int          weightInitType = 0; //0 - random weight initalization, 1 - weights are filled with a set value, 2 - weights are loaded from directory.
	std::string yAFType = "tanh";
	std::string lFType = "mse";
	bool allowDHNext = false;
	bool readNetOutputs = true;
	bool readGrads = true;
	bool readNetArray[3] = {true,false,false}; //entry 1: readNetOutputs, entry 2: readOutputGrads, entry 3:readLstmBlocks
//

//Layers///////
__declspec(align(16)) double iNOut[tS][iL] = { 0 };

__declspec(align(16)) LSTMBlock lstmBlocks[tS][hL];
__declspec(align(16)) double hNOut[tS][hL] = { 0 };
__declspec(align(16)) double oNOut[tS][oL] = { 0 };
__declspec(align(16)) double oNGrad[tS][oL] = { 0 };
__declspec(align(16)) double hNGrad[tS][hL] = { 0 };
__declspec(align(16)) double dHGrad[hL] = { 0 };
///////////////
__declspec(align(16)) double error[oL] = { 0 };
///////////////

//WEIGHTS/////////////////////////
//matrixes are in transpose order by default to benefit matrix loop caching 
//(e.g 'hI(hL, iL)' is used for input to hidden layers on FeedForward
// whilst the naming iH(iL,hL) would be more representative of the FeedForward direction)

WeightMatrix<double> oH(oL, hL);

// WEIGHT MATRIXES
WeightMatrix<double> cellWX(hL, iL);
WeightMatrix<double> cellWH(hL, hL);
WeightMatrix<double> iGWX(hL, iL);
WeightMatrix<double> iGWH(hL, hL);
WeightMatrix<double> oGWX(hL, iL);
WeightMatrix<double> oGWH(hL, hL);
WeightMatrix<double> fGWX(hL, iL);
WeightMatrix<double> fGWH(hL, hL);
//////////////////////////////////

//RES/////////////////////////////
__declspec(align(16)) double res[hL] = { 0 };
__declspec(align(16)) double res2[hL] = { 0 };
__declspec(align(16)) double* res3;
__declspec(align(16)) double res4[hL] = { 0 };

//////////////////////////////////
__declspec(align(16)) double* cell_xInWx;
__declspec(align(16)) double* cell_pHWh;
__declspec(align(16)) double* iG_xInWx;
__declspec(align(16)) double* iG_pHWh;
__declspec(align(16)) double* oG_xInWx;
__declspec(align(16)) double* oG_pHWh;
__declspec(align(16)) double* fG_xInWx;
__declspec(align(16)) double* fG_pHWh;
//////////////////////////////////
__declspec(align(16)) double cell_Grad[hL] = { 0 };
__declspec(align(16)) double iG_Grad[hL] = { 0 };
__declspec(align(16)) double oG_Grad[hL] = { 0 };
__declspec(align(16)) double fG_Grad[hL] = { 0 };
//////////////////////////////////
__declspec(align(16)) double inputData[tS][iL];
__declspec(align(16)) double outputData[tS][oL - 1];
//////////////////////////////////

/*
================================================
Init Net
================================================
*/
void LSTM::initNet() {
	try {
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

			//Biases
			iNOut[cT][iL - 1] = 1;
			hNOut[cT][hL - 1] = 1;
			oNOut[cT][oL - 1] = 1;

		}
		//Mats
		double min = 0.0;
		double max = wSV;

		if (weightInitType == 0)
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
		else if (weightInitType == 1)
		{
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
	//Input & Context Layers to LSTMBlock Connections
	}

	cell_xInWx = cellWX.Mult_Mat_Vec_NxM_Mx1_SSE(iNOut[t]);
	cell_pHWh  = cellWH.Mult_Mat_Vec_NxM_Mx1_SSE(hNOut[t - 1]);
	iG_xInWx   = iGWX.Mult_Mat_Vec_NxM_Mx1_SSE(iNOut[t]);
	iG_pHWh    = iGWH.Mult_Mat_Vec_NxM_Mx1_SSE(hNOut[t - 1]);
	oG_xInWx   = oGWX.Mult_Mat_Vec_NxM_Mx1_SSE(iNOut[t]);
	oG_pHWh    = oGWH.Mult_Mat_Vec_NxM_Mx1_SSE(hNOut[t - 1]);
	fG_xInWx   = fGWX.Mult_Mat_Vec_NxM_Mx1_SSE(iNOut[t]);
	fG_pHWh    = fGWH.Mult_Mat_Vec_NxM_Mx1_SSE(hNOut[t - 1]);

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
	AF(oNOut[t], res3, oL, &yAFType);
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
void LSTM::BackProp(double* target)
{
	//Get Error and output Grad(s)
	LF(oNGrad[t], oNOut[t], oL,target, &lFType, &yAFType,0);
	//output Grad to hidden Grad
	oH.Mult_Tranpose_Mat_Vec(oNGrad[t], res4);

	//Propagate through lstm block with hidden grad to obtain (c_o_i_f) grads
	for (intptr_t hN = 0; hN < hL; hN++)
	{
		lstmBlocks[t][hN].LSTM_BackPropagate(res4[hN]);

		cell_Grad[hN] = lstmBlocks[t][hN].cell_Grad;
		oG_Grad[hN] = lstmBlocks[t][hN].oG_Grad;
		iG_Grad[hN] = lstmBlocks[t][hN].iG_Grad;
		fG_Grad[hN] = lstmBlocks[t][hN].fG_Grad;		
	}

	//calc: output layer to hidden layer weights
	oH.Calc_Weights_ADAGRAD_SSE(hNOut[t], oNGrad[t], &eta);

	//calc: hidden layer to (output layer weights) & (context layer weights)
	cellWX.Calc_Weights_ADAGRAD_SSE(iNOut[t],cell_Grad, &eta);
	cellWH.Calc_Weights_ADAGRAD_SSE(hNOut[t - 1], cell_Grad, &eta);
	iGWX.Calc_Weights_ADAGRAD_SSE(iNOut[t], iG_Grad, &eta);
	iGWH.Calc_Weights_ADAGRAD_SSE(hNOut[t - 1], iG_Grad, &eta);
	oGWX.Calc_Weights_ADAGRAD_SSE(iNOut[t], oG_Grad, &eta);
	oGWH.Calc_Weights_ADAGRAD_SSE(hNOut[t - 1], oG_Grad, &eta);
	fGWX.Calc_Weights_ADAGRAD_SSE(iNOut[t], fG_Grad, &eta);
	fGWH.Calc_Weights_ADAGRAD_SSE(hNOut[t - 1], fG_Grad, &eta);



	if (allowDHNext)
	{
		//dHNext gradients
		{
			cellWH.Mult_Tranpose_Mat_Vec(cell_Grad, res4);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHCell_Grad = res4[hN];
			}
			iGWH.Mult_Tranpose_Mat_Vec(iG_Grad, res4);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHIG_Grad = res4[hN];
			}
			oGWH.Mult_Tranpose_Mat_Vec(oG_Grad, res4);
			for (intptr_t hN = 0; hN < (hL); hN++)
			{
				lstmBlocks[t - 1][hN].dHOG_Grad = res4[hN];
			}
			fGWH.Mult_Tranpose_Mat_Vec(fG_Grad, res4);
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

		cout << '\n' << '\n' << "|| Neuron Outputs ||||||||||||||||||||||||||||||||||||||||" << '\n';

		//cout << '\n' << "o layer outputs: " << '\n';
		//for (int i = 0; i < iL;i++)
		//{
		//cout << " " << std::setprecision(17) << iNOut[i];
		//}
		//cout << '\n' << "h layer outputs: " << '\n';
		//for (int i = 0; i < hL;i++)
		//{
		//cout << " " << std::setprecision(17) << hNOut[i];
		//}
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
void LSTM::UpdateWeights()
{
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
void LSTM::ResetdHNextGrads()
{
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
void LSTM::Run() {

	cout << "Running net: " << '\n';

	initNet();

	clock_t t1, t2;
	t1 = clock();
	//Process
	
		for (intptr_t i = 0; i < epochs; i++)
		{
			if (i > epochs - 2)
			{
				gradCheck = true;
			}

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
	NetInfo(readNetArray[0], readNetArray[1], readNetArray[2]);
	t2 = clock();
	double diff((double)t2 - (double)t1);
	cout << "time: " << (diff / (double)CLOCKS_PER_SEC) << endl;

}


/*
================================================
Data
================================================
*/
void LSTM::InitData()
{

	//Sinewave Dataset
	for (intptr_t i = 0; i < (tS); i++)
	{
		inputData[i][0] = (0.25 * sin((2 * (3.14159) * i * 100) / 44000));
		for (intptr_t k = 0; k < (oL); k++)
		{
			outputData[i][k] = (0.25 * sin(((1+k) * (3.14159) * (i + 1) * 100) / 44000));
		}
	}

	for (intptr_t cT = 0; cT < tS; cT++)
	{
		cout << "target: " << std::setprecision(17) << outputData[cT][0] << '\n';
		cout << '\n';
	}
}