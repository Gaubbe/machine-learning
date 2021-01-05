#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <time.h>

#include "src/utils/FileUtils.h"
#include "src/ml/network/FeedForwardNetworkBuilder.h"

int main() 
{
	srand((unsigned int) time(0));
	auto net = ml::FeedForwardNetworkBuilder::CreateNetwork().AddLayer({28 * 28, 16}).AddLayer({16, 16}).AddLayer({16, 10}).Build();

	return 0;
}