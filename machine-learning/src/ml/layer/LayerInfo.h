#pragma once
#include "../ActivationFunction.h"
#include <memory>

namespace ml {
struct LayerInfo {
	int numInputs = -1;
	int numOutputs = -1;
	std::shared_ptr<ActivationFunction> activationFunction = std::make_shared<SigmoidActivationFunction>();
	double learningRate = 0.1;
};
}