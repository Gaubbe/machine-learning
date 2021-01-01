#include "Layer.h"

namespace ml {
Layer::Layer(int numInputs, int numOutputs, ActiavtionFunction* activationFunction, double learningRate)
	:m_NumInputs(numInputs), m_NumOutputs(numOutputs), m_ActivationFunction(activationFunction), m_LearningRate(learningRate), m_IterationsSinceLastBatch(0)
{
}
}