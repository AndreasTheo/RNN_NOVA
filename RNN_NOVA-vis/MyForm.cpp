
#include "MyForm.h"
#include "RNN_NOVA\lstm\lstm.h"
#include "windows.h"
#include <iostream>
#include <fstream>
#include <string>
#include <msclr\marshal_cppstd.h>
#include "RNN_NOVA\networkparameters.h"
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif // win32

using namespace System;
using namespace System::Windows::Forms;
using namespace System::Threading;
using namespace System::Diagnostics;
using namespace System::IO;
using namespace System::Security::Permissions;
#include <vector>
using std::vector;

void sleepcp(int milliseconds)
{
#ifdef WIN32
	Sleep(milliseconds);
#else
	usleep(milliseconds * 1000);
#endif // win32
}


void LSTMNeuralNetwork(Object^ data) {
	try
	{
		LSTM lstm;
		RNN_NOVA_vis::MyForm ^ob = (RNN_NOVA_vis::MyForm^)data;
		lstm.initNet(0.01, 0.5, 0, 100.0, 0.05, 0, "tanh", "mse");
		lstm.InitData();
		double tempError = 1;   
		int c = 1;

		ob->DrawOutputDataC2_Invoke(lstm.VisOutputData());
		while ((tempError > 0.001) && (c < 5000))
		{
			lstm.Run(1);
			tempError = lstm.VisNetErrorData();
			vector<vector<double>> arr = lstm.VisNetOutputData();
			ob->DrawPointsC1and2_Invoke(c, tempError,arr);
			if (c % 100 == 0)
			{
				sleepcp(100);
			}
			c++;
		}
		lstm.NetInfo(true,false,false);
	}
	catch (Exception^ ex)
	{
	}
}

void Graphs() {

	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	RNN_NOVA_vis::MyForm form;

	Thread^ thread1 = gcnew Thread(gcnew ParameterizedThreadStart(&LSTMNeuralNetwork));
	//Thread^ thread1 = gcnew Thread(gcnew ThreadStart(Graphs));
	thread1->Name = "Graphs";
	thread1->Start(%form);

	Application::Run(%form);
}

[STAThread]
void Main() {
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);
	Graphs();
	//return 0;
}