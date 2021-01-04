#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <time.h>
#include "src/ml/FeedForwardLayer.h"
#include "src/ml/Network.h"

#include "src/utils/FileUtils.h"

static uint32_t SwitchEndian(uint32_t in)
{
	uint32_t result = 0;
	result |= (in & 0xFF000000) >> 24;
	result |= (in & 0x00FF0000) >> 8;
	result |= (in & 0x0000FF00) << 8;
	result |= (in & 0x000000FF) << 24;
	return result;
}

static std::vector<Eigen::Matrix<double, 28 * 28, 1>> GetMNISTImageData(const char* filePath, int number) 
{
	std::vector<uint8_t> fileData = ReadFileIntoVector(filePath, true);
	std::vector<Eigen::Matrix<double, 28 * 28, 1>> result(number);
	for (int i = 0; i < number; i++) {
		for (int x = 0; x < 28; x++) {
			for (int y = 0; y < 28; y++) {
				uint8_t pixelValue = *(uint8_t*) (fileData.data() + 16 + (i * 28 * 28) + y + (x * 28));
				double normalized = static_cast<double>(pixelValue) / 255.0;
				result[i](x + y * 28) = normalized;
			}
		}
	}
	return result;
}

static std::vector<Eigen::Matrix<double, 10, 1>> GetMNISTLabelData(const char* filePath, int number)
{
	std::vector<uint8_t> fileData = ReadFileIntoVector(filePath, true);
	std::vector<Eigen::Matrix<double, 10, 1>> result(number);
	for (int i = 0; i < number; i++) {
			uint8_t label = *(uint8_t*) (fileData.data() + 8 + i);
			result[i] = Eigen::Matrix<double, 10, 1>::Zero();
			result[i](label) = 1.0;
	}
	return result;
}

int main() 
{
	srand((unsigned int) time(0));
	std::vector<Eigen::Matrix<double, 28 * 28, 1>> trainingData = GetMNISTImageData("MNIST/train-images.idx3-ubyte", 60000);
	std::vector<Eigen::Matrix<double, 10, 1>> trainingLabels = GetMNISTLabelData("MNIST/train-labels.idx1-ubyte", 60000);
	std::vector<Eigen::Matrix<double, 28 * 28, 1>> testingData = GetMNISTImageData("MNIST/t10k-images.idx3-ubyte", 10000);
	std::vector<Eigen::Matrix<double, 10, 1>> testingLabels = GetMNISTLabelData("MNIST/t10k-labels.idx1-ubyte", 10000);
	ml::FeedForwardLayer firstHiddenLayer(28 * 28, 16);
	ml::FeedForwardLayer secondHiddenLayer(16, 16);
	ml::FeedForwardLayer outputLayer(16, 10);
	ml::Network net;

	net.AddLayer(&firstHiddenLayer);
	net.AddLayer(&secondHiddenLayer);
	net.AddLayer(&outputLayer);

	for (int epoch = 0; epoch < 100; epoch++) {
		std::cout << ((double) epoch / 100.0) * 100.0 << "%" <<std::endl;

		for (int example = 0; example < trainingData.size(); example++) {
			net.Forward(trainingData[example]);
			net.Back(trainingLabels[example]);

			if (example % 10 == 0)
				net.ApplyBatch();
		}
		net.ApplyBatch();
	}

	std::cout << "Testing..." << std::endl;

	double accuracy = 0.0;
	for (int testing = 0; testing < testingData.size(); testing++) {
		net.Forward(testingData[testing]);
		Eigen::VectorXd guess = net.GetLastLayerActivations();
		double guessValue = 0.0;
		int actualGuess = 0;
		int answer = 0;
		for (int i = 0; i < 10; i++) {
			if (guess(i) > guessValue) {
				guessValue = guess(i);
				actualGuess = i;
			}

			if (testingLabels[testing](i) == 1.0)
				answer = i;
		}

		if (answer == actualGuess)
			accuracy += 100.0;
	}

	std::cout << "Accurcy: " << accuracy / testingData.size() << "%" << std::endl;

	return 0;
}