#pragma once
#include <memory>
#include "../ActivationFunction.h"
#include "FeedForwardLayer.h"
#include "LayerInfo.h"

namespace ml {
class FeedForwardLayerBuilder{
private:
	LayerInfo m_Info;
public:
	static FeedForwardLayerBuilder CreateLayer();
	FeedForwardLayerBuilder& SetNumInputs(int numInputs);
	FeedForwardLayerBuilder& SetNumOutputs(int numOutputs);
	FeedForwardLayerBuilder& SetActivationFunction(std::shared_ptr<ActivationFunction> activationFunction);
	FeedForwardLayerBuilder& SetLearningRate(double learningRate);
	FeedForwardLayerBuilder& SetLayerInfo(LayerInfo info);
	std::shared_ptr<Layer> Build();
};
}