#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

class Neuron
{
public:
	double w1, w2;
	double wb;
	vector<double> X1, X2;
	vector<double> Y;
	double dw1, dw2;
	double db;

	double getActivation(const double sigma)  //Activation Function
	{
		//sigmoid function
		return 1 / (1 + exp(-sigma));
	}

	double feedForward(const double x1, const double x2)    //sigma calculation
	{
		const double sigma = w1*x1 + w2*x2 + wb * 1;

		return getActivation(sigma);
	}

	double feedForwardAndPrint(const double x1, const double x2)
	{
		printf("x1 : %.f, x2 : %.f, sigma : %f \n", x1, x2, feedForward(x1, x2));
		return 0.0;
	}

	Neuron()
	{
	}

	Neuron(double w1_input, double w2_input, double wb_input)
	{
		w1 = w1_input;
		w2 = w2_input;
		wb = wb_input;
	}

	~Neuron()
	{
	}

	void andGenerate()
	{
		//AND
		X1.push_back(0);
		X1.push_back(0);
		X1.push_back(1);
		X1.push_back(1);

		X2.push_back(0);
		X2.push_back(1);
		X2.push_back(0);
		X2.push_back(1);

		Y.push_back(0);
		Y.push_back(0);
		Y.push_back(0);
		Y.push_back(1);
	}

	void orGenerate()
	{
		//OR
		X1.push_back(0);
		X1.push_back(0);
		X1.push_back(1);
		X1.push_back(1);

		X2.push_back(0);
		X2.push_back(1);
		X2.push_back(0);
		X2.push_back(1);

		Y.push_back(0);
		Y.push_back(1);
		Y.push_back(1);
		Y.push_back(1);
	}

	double getActGrad(const double x1, const double x2, double output)
	{
		double ewxb = exp(w1*x1 + w2*x2 + wb);

		if (output == 1)
		{
			dw1 += -x1 / (1 + ewxb);
			dw2 += -x2 / (1 + ewxb);
			db += -1 / (1 + ewxb);
		}
		else
		{
			dw1 += (x1*ewxb) / (1 + ewxb);
			dw2 += (x2*ewxb) / (1 + ewxb);
			db += ewxb / (1 + ewxb);
		}
		return 0.0;
	}

	double getLogisticCost(double x1, double x2, double y)
	{
		double sigma = feedForward(x1, x2);
		double p = getActivation(sigma);

		if (y == 1)
		{
			return -log(p);
		}
		else
		{
			return -log(1 - p);
		}
	}

	double getTotalLogisticCost(vector<double> X1, vector<double>X2, vector<double> Y)
	{
		int n = static_cast<int>(X1.size());
		double cost = 0;

		for (int i = 0; i<n; i++){
			cost += getLogisticCost(X1[i], X1[i], Y[i]);
		}

		return cost / n;
	}

	void backProp(vector<double> X1, vector<double> X2, vector<double> Y)
	{
		const double alpha = 0.001;
		int n = static_cast<int>(X1.size());
		dw1 = 0;
		dw2 = 0;
		db = 0;

		for (int i = 0; i<n; i++){
			getActGrad(X1[i], X2[i], Y[i]);
		}

		w1 = w1 - alpha*dw1 / n;
		w2 = w2 - alpha*dw2 / n;
		wb = wb - alpha*db / n;
	}
};

int main()
{
	Neuron N(1.0, 1.0, 0.0);
	//N.andGenerate();
	N.orGenerate();

	double cost=0;

	for (int i = 0; i<100000; i++){
		if (i % 10000 == 0){
			printf("Training : %d  \n", i);
			N.feedForwardAndPrint(0, 0);
			N.feedForwardAndPrint(0, 1);
			N.feedForwardAndPrint(1, 0);
			N.feedForwardAndPrint(1, 1);
			printf("w1 = %f, w2 = %f, wb = %f  \n\n", N.w1, N.w2, N.wb);
			printf("Cost = %f \n\n", cost);
		}
		cost = N.getTotalLogisticCost(N.X1, N.X2, N.Y);
		N.backProp(N.X1, N.X2, N.Y);
	}

	return 0;
}
