#include "FeedForwardLayerBuilder.h"

namespace ml {
FeedForwardLayerBuilder FeedForwardLayerBuilder::CreateLayer()
{
	return FeedForwardLayerBuilder();
}

FeedForwardLayerBuilder& FeedForwardLayerBuilder::SetNumInputs(int numInputs)
{
	this->m_Info.numInputs = numInputs;
	return *this;
}

FeedForwardLayerBuilder& FeedForwardLayerBuilder::SetNumOutputs(int numOutputs)
{
	this->m_Info.numOutputs = numOutputs;
	return *this;
}

FeedForwardLayerBuilder& FeedForwardLayerBuilder::SetActivationFunction(std::shared_ptr<ActivationFunction> activationFunction)
{
	this->m_Info.activationFunction = activationFunction;
	return *this;
}

FeedForwardLayerBuilder& FeedForwardLayerBuilder::SetLearningRate(double learningRate)
{
	this->m_Info.learningRate = learningRate;
	return *this;
}

FeedForwardLayerBuilder& FeedForwardLayerBuilder::SetLayerInfo(LayerInfo info)
{
	this->m_Info = info;
	return *this;
}

std::shared_ptr<Layer> FeedForwardLayerBuilder::Build()
{
	return std::make_shared<FeedForwardLayer>(this->m_Info.numInputs, this->m_Info.numOutputs, this->m_Info.activationFunction, this->m_Info.learningRate);
}

}