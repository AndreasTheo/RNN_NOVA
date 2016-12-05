#pragma once

#include <vector>
using std::vector;

namespace RNN_NOVA_vis {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;


	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		delegate void Delegate_DrawPointsC1and2(double x, double y, cli::array<double^, 2> ^arr);
		delegate void Delegate_DrawOutputDataC2(cli::array<double^, 2> ^arr);

		/*void AddSeries(System::Windows::Forms::DataVisualization::Charting::Chart^ chart,System::String^ seriesName)
		{		
			System::Windows::Forms::DataVisualization::Charting::Series^  newSeries = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());			
			newSeries->ChartArea = L"ChartArea1";
			newSeries->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
			newSeries->Legend = L"Legend1";
			newSeries->Name = seriesName;
			chart->Series->Add(newSeries);
		}*/
		//void UpdateSeries(System::Windows::Forms::DataVisualization::Charting::Chart^ chart, System::String^ seriesName, vector<vector<double>> ary	)
		//{
		//}

		void DrawOutputDataC2_Invoke(vector<vector<double>> ary)
		{
			Delegate_DrawOutputDataC2^ action = gcnew Delegate_DrawOutputDataC2(this, &MyForm::DrawOutputDataC2);
			cli::array< double^, 2 >^ arr = gcnew cli::array< double^, 2 >(ary.size(), ary[0].size());
			for (int i = 0; i < ary.size();i++)
			{
				for (int j = 0; j < ary[0].size();j++)
				{
					arr[i, j] = ary[i][j];
				}
			}
			this->BeginInvoke(action, arr);
		}

		void DrawOutputDataC2(cli::array<double^, 2> ^arr)
		{
			if (!addingPoints)
			{
				addingPoints = true;
				this->chart2->Series["Series2"]->Points->Clear();

				for (int t = 0; t < arr->GetLength(0);t++)
				{
					this->chart2->Series["Series2"]->Points->AddXY(t, arr[t, 0]);
				}
				addingPoints = false;
			}
		}

		bool addingPoints;
		void DrawPointsC1and2_Invoke(double x, double y, vector<vector<double>> ary)
		{
			Delegate_DrawPointsC1and2^ action = gcnew Delegate_DrawPointsC1and2(this, &MyForm::DrawPointsC1and2);
			cli::array< double^, 2 >^ arr = gcnew cli::array< double^, 2 >(ary.size(), ary[0].size());
			for (int i = 0; i < ary.size();i++)
			{
				for (int j = 0; j < ary[0].size();j++)
				{
					arr[i,j] = ary[i][j];
				}
			}
			this->BeginInvoke(action, x,y,arr);
		}
		void DrawPointsC1and2(double x, double y, cli::array<double^, 2> ^arr)
		{
			if (!addingPoints)
			{
				addingPoints = true;
				//Console::WriteLine("error: " + y);
				this->chart1->Series["Series1"]->Points->AddXY(x, y);

				this->chart2->Series["Series1"]->Points->Clear();

				for (int t = 0; t < arr->GetLength(0);t++)
				{
					this->chart2->Series["Series1"]->Points->AddXY(t, arr[t, 0]);
				}
				addingPoints = false;
			}
		}
		
		MyForm(void)
		{
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::DataVisualization::Charting::Chart^  chart1;
	private: System::Windows::Forms::DataVisualization::Charting::Chart^  chart2;
	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea1 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			System::Windows::Forms::DataVisualization::Charting::Legend^  legend1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			System::Windows::Forms::DataVisualization::Charting::Series^  series1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
			System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea2 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			System::Windows::Forms::DataVisualization::Charting::Legend^  legend2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			System::Windows::Forms::DataVisualization::Charting::Series^  series2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
			System::Windows::Forms::DataVisualization::Charting::Series^  series3 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
			this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			this->chart2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart2))->BeginInit();
			this->SuspendLayout();
			// 
			// chart1
			// 
			chartArea1->Name = L"ChartArea1";
			this->chart1->ChartAreas->Add(chartArea1);
			legend1->Name = L"Legend1";
			this->chart1->Legends->Add(legend1);
			this->chart1->Location = System::Drawing::Point(13, 13);
			this->chart1->Name = L"chart1";
			series1->ChartArea = L"ChartArea1";
			series1->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
			series1->Legend = L"Legend1";
			series1->Name = L"Series1";
			this->chart1->Series->Add(series1);
			this->chart1->Size = System::Drawing::Size(500, 281);
			this->chart1->TabIndex = 0;
			this->chart1->Text = L"chart1";
			this->chart1->Click += gcnew System::EventHandler(this, &MyForm::chart1_Click);
			// 
			// chart2
			// 
			this->chart2->BackColor = System::Drawing::Color::OliveDrab;
			chartArea2->Name = L"ChartArea1";
			this->chart2->ChartAreas->Add(chartArea2);
			legend2->Name = L"Legend1";
			this->chart2->Legends->Add(legend2);
			this->chart2->Location = System::Drawing::Point(12, 300);
			this->chart2->Name = L"chart2";
			series2->ChartArea = L"ChartArea1";
			series2->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
			series2->Legend = L"Legend1";
			series2->Name = L"Series1";
			series3->ChartArea = L"ChartArea1";
			series3->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
			series3->Legend = L"Legend1";
			series3->Name = L"Series2";
			this->chart2->Series->Add(series2);
			this->chart2->Series->Add(series3);
			this->chart2->Size = System::Drawing::Size(501, 302);
			this->chart2->TabIndex = 1;
			this->chart2->Text = L"chart2";
			this->chart2->Click += gcnew System::EventHandler(this, &MyForm::chart2_Click);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(520, 611);
			this->Controls->Add(this->chart2);
			this->Controls->Add(this->chart1);
			this->Name = L"MyForm";
			this->Text = L"Graph";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart2))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void chart1_Click(System::Object^  sender, System::EventArgs^  e) {
	}
	private: System::Void chart2_Click(System::Object^  sender, System::EventArgs^  e) {
	}
};
}
