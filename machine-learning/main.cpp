#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <time.h>
#include "src/ml/FeedForwardLayer.h"

int main() 
{
	srand((unsigned int) time(0));
	Eigen::Vector2d inputs[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	Eigen::VectorXd targets[4];
	
	targets[0] = Eigen::VectorXd::Zero(1);
	targets[1] = Eigen::VectorXd::Zero(1);
	targets[2] = Eigen::VectorXd::Zero(1);
	targets[3] = Eigen::VectorXd::Zero(1);

	targets[0](0) = -1; 
	targets[1](0) =  1;
	targets[2](0) =  1;
	targets[3](0) = -1;


	ml::FeedForwardLayer layer1(2, 2, new TanhActivationFunction);
	ml::FeedForwardLayer layer2(2, 1, new TanhActivationFunction);
	
	for (int loop = 0; loop < 1000; loop++) {
		for (int i = 0; i < 4; i++) {
			layer1.Forward(inputs[i]);
			layer2.Forward(layer1.GetActivations());
			layer2.ComputeOuputLayerErrors(targets[i]);
			layer2.Back(layer2.GetErrors());
			layer1.Back(layer2.GetPreviousErrors());

			layer1.ApplyWeightDeltas();
			layer2.ApplyWeightDeltas();
		}
	}

	Eigen::MatrixXd output(11, 11);
	for (int x = 0; x < 11; x++) {
		for (int y = 0; y < 11; y++) {
			Eigen::Vector2d input(0.1 * x, 0.1 * y);
			layer1.Forward(input);
			layer2.Forward(layer1.GetActivations());
			output(x, y) = layer2.GetActivations()(0);
		}
	}

	std::cout << output << std::endl << std::endl;
	std::cout << layer1.GetWeights() << std::endl << std::endl;
	std::cout << layer2.GetWeights() << std::endl << std::endl;

	return 0;
}