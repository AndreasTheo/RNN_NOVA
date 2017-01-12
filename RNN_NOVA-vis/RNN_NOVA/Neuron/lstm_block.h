//#include "../../stdafx.h"
#pragma once
#ifndef LSTMBLOCK_HEADER
#define LSTMBLOCK_HEADER
#include <iomanip>


	class LSTMBlock {

	public:

		struct internalConn {

			double w;
			double dW;
			double dWCache;
			double dWCombo;
		};

		//
		int         index;
		int			outConns;
		bool		isBias;
		//gradients
		long double grad, cell_Grad, iG_Grad, oG_Grad, fG_Grad;
		long double dHCell_Grad, dHIG_Grad, dHOG_Grad, dHFG_Grad;
		//outputs
		long double output, cellTM1;
		long double g;
		long double iG;
		long double oG;
		long double fG;
		long double h;
		long double cell;
		//outputs - bias
		long double cellBias, iGBias, oGBias, fGBias;
		//
		//internal weights
		internalConn  iGWC;
		internalConn  oGWC;
		internalConn  fGWC;

		//
		double  dWCache;

		void Init(intptr_t outConnections, intptr_t i);
		long double LSTM_FeedForward(long double cell_xInWx_pHWh, long double iG_xInWx_pHWh, long double oG_xInWx_pHWh, long double fG_xInWx_pHWh, long double cellPrev);
		void LSTM_BackPropagate(long double yGradWHO);
		void LSTM_ReadGates();
		void LSTM_BackPropagate_ICB(int adaptGrad,internalConn *weight, long double grad, long double eta, long double momentum);
		//
	};
#endif