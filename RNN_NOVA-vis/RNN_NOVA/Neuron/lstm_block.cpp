#pragma once
#include "lstm_block.h"
#include "..\Activation_Functions\sigmoid.h"
#include "..\Activation_Functions\tanh.h"
#include <iostream>

using namespace std;
	void LSTMBlock::Init(int outCs, int i) {

		try 
		{

		output = 0, cellTM1 = 0;
		index = i;
		outConns = outCs;
		cellBias = 1, iGBias = 1, oGBias = 1, fGBias = 1;
		cell = 0;
		g = 0;
		iG = 0;
		fG = 0;
		h = 0;
		oG = 0;

		grad = 0, cell_Grad = 0, iG_Grad = 0, oG_Grad = 0, fG_Grad = 0;
		dHCell_Grad = 0, dHIG_Grad = 0, dHOG_Grad = 0, dHFG_Grad = 0;
		iGWC.w = 0; iGWC.dW = 0, iGWC.dWCache = 0, iGWC.dWCombo = 0;
		oGWC.w = 0; oGWC.dW = 0, oGWC.dWCache = 0, oGWC.dWCombo = 0;
		fGWC.w = 0; fGWC.dW = 0, fGWC.dWCache = 0, fGWC.dWCombo = 0;

		dWCache = 0;

		}
		catch (int err) {
			cerr << "Failed to initialize LSTMBlock \n";
			cerr << err << '\n';
		}
	}


	long double LSTMBlock::LSTM_FeedForward(long double cell_xInWx_pHWh, long double iG_xInWx_pHWh, long double oG_xInWx_pHWh, long double fG_xInWx_pHWh, long double cellPrev) {

		cellTM1 = cellPrev;
		//cellTM1 = 1;
		iG = Sigmoid(iG_xInWx_pHWh + iGBias + (iGWC.w * cellTM1));
		fG = Sigmoid(fG_xInWx_pHWh + fGBias + (fGWC.w * cellTM1));
		g = TanHyp(cell_xInWx_pHWh + cellBias);
		cell = (iG * g) + (fG * cellTM1);
		oG = Sigmoid(oG_xInWx_pHWh + oGBias + (oGWC.w * cellTM1));
		h = TanHyp(cell);
		output = oG * h;
		//LSTM_ReadGates();
		return (output);
	};

	void LSTMBlock::LSTM_ReadGates() {

		cout << '\n' << "-----------------------------------" << '\n';
		cout << "iG: " << std::setprecision(17) << iG << '\n';
		cout << "fG: " << std::setprecision(17) << fG << '\n';
		cout << "g: " << std::setprecision(17) << g << '\n';
		cout << "cell: " << std::setprecision(17) << cell << '\n';
		cout << "oG: " << std::setprecision(17) << oG << '\n';
		cout << "h: " << std::setprecision(17) << h << '\n';
		cout << "output: " << std::setprecision(17) << output << '\n';
		cout << '\n' << "-----------------------------------" << '\n';
		cout << '\n' << "-----------------------------------" << '\n';
		cout << "cell_Grad: " << std::setprecision(17) << cell_Grad << '\n';
		cout << "iG_Grad: " << std::setprecision(17) << iG_Grad << '\n';
		cout << "oG_Grad: " << std::setprecision(17) << oG_Grad << '\n';
		cout << "fG_Grad: " << std::setprecision(17) << fG_Grad << '\n';
		cout << '\n' << "-----------------------------------" << '\n';
	}

	void LSTMBlock::LSTM_BackPropagate(long double yGradWHO) {

		//dHNext
	    yGradWHO += (dHCell_Grad + dHFG_Grad + dHIG_Grad + dHOG_Grad);
		//calculate differential gradients
		cell_Grad = (yGradWHO * oG * TanhDeriv(h) * iG * TanhDeriv(g));
		iG_Grad = (yGradWHO * oG * TanhDeriv(h) * SigmoidDeriv(iG) * g);
		oG_Grad = (yGradWHO * SigmoidDeriv(oG) * h);
		fG_Grad = (yGradWHO * oG * TanhDeriv(h) * SigmoidDeriv(fG) * cellTM1);

		grad = cell_Grad;

		LSTM_BackPropagate_ICB(0, &iGWC, iG_Grad, ((1e-1) * 4), 0.5);
		LSTM_BackPropagate_ICB(0, &fGWC, oG_Grad, ((1e-1) * 4), 0.5);
		LSTM_BackPropagate_ICB(0, &oGWC, fG_Grad, ((1e-1) * 4), 0.5);
	}
	/* | Internal Cell Bridge | */
	void LSTMBlock::LSTM_BackPropagate_ICB(int adaptGrad,internalConn *weight, long double grad, long double eta, long double momentum) {

		//ADAGRAD
		double dx = cellTM1 * grad;
		weight->dWCache += (dx * dx);
		double mem = weight->dWCache;
		weight->dWCombo = ((eta * dx) / sqrt(mem + 1e-8));

		//Update on t
		weight->w += weight->dWCombo;
		weight->dWCombo = 0;



	}
