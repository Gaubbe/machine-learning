#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <time.h>

#include "src/ml/Perceptron.h"
int main() 
{
	srand((unsigned int) time(0));
	Eigen::Vector2d andInputs[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double andOutputs[4] = {0, 0, 0, 1};

	ml::Perceptron p(2, 0.1F, new TanhActivationFunction);

	for(int loop = 0; loop < 1000; loop++) 
	{
		for(int i = 0; i < 4; i++)
		{
			int randomInput = rand() % 4;
			p.TrainOneExample(andInputs[randomInput], andOutputs[randomInput]);
		}
	}

	Eigen::MatrixXd output(11, 11);

	for (int x = 0; x < 11; x++) {
		for (int y = 0; y < 11; y++) {
			Eigen::Vector2d input(0.1 * x, 0.1 * y);
			p.FeedForward(input);
			output(x, y) = p.GetActivation();
		}
	}

	std::cout << output << std::endl;

	return 0;
}