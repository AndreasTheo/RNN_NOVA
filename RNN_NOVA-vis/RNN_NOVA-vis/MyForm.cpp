#include "MyForm.h"
#include "RNN_NOVA\lstm\lstm.h"
#include "windows.h"
LSTM lstm;

using namespace System;
using namespace System::Windows::Forms;


[STAThread]
void Main()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);


	lstm.InitData();
	lstm.Run();

	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);


	RNN_NOVA_vis::MyForm form;
	Application::Run(%form);
}
