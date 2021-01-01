#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <time.h>
#include "src/ml/FeedForwardLayer.h"
#include "src/ml/Network.h"

#include "src/utils/FileUtils.h"

int main() 
{
	srand((unsigned int) time(0));
	Eigen::Vector2d inputs[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	Eigen::VectorXd targets[4];
	
	targets[0] = Eigen::VectorXd::Zero(1);
	targets[1] = Eigen::VectorXd::Zero(1);
	targets[2] = Eigen::VectorXd::Zero(1);
	targets[3] = Eigen::VectorXd::Zero(1);

	targets[0](0) =  0; 
	targets[1](0) =  1;
	targets[2](0) =  1;
	targets[3](0) =  0;


	ml::FeedForwardLayer layer1(2, 2, new SigmoidActiavtionFunction, 0.1);
	ml::FeedForwardLayer layer2(2, 1, new SigmoidActiavtionFunction, 0.1);
	ml::Network net;
	
	net.AddLayer(&layer1);
	net.AddLayer(&layer2);

	for (int loop = 0; loop < 10000; loop++) {
		for (int i = 0; i < 4; i++) {
			net.Forward(inputs[i]);
			net.Back(targets[i]);
		}
	}

	Eigen::MatrixXd output(11, 11);
	for (int x = 0; x < 11; x++) {
		for (int y = 0; y < 11; y++) {
			Eigen::Vector2d input(0.1 * x, 0.1 * y);
			net.Forward(input);
			output(x, y) = net.GetLastLayerActivations()(0);
		}
	}

	std::cout << output << std::endl << std::endl;
	std::cout << layer1.GetWeights() << std::endl << std::endl;
	std::cout << layer2.GetWeights() << std::endl << std::endl;

	std::cout << ReadFileIntoVector("main.cpp", false).data() << std::endl;

	return 0;
}