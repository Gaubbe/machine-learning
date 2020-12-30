#include "FeedForwardLayer.h"

namespace ml {
FeedForwardLayer::FeedForwardLayer(int numInputs, int numOutputs, ActivationFuction* activationFunction, double learningRate)
	:Layer(numInputs, numOutputs, activationFunction, learningRate)
{
	this->m_Weights = Eigen::MatrixXd::Random(numOutputs, numInputs + 1);
}

void FeedForwardLayer::Forward(Eigen::VectorXd inputs)
{
	if (inputs.rows() == this->m_NumInputs) {
		AddBiasToInput(&inputs);
		this->m_Inputs = inputs;
		this->m_Outputs = this->m_Weights * inputs;
		this->m_Activations = this->m_Outputs.unaryExpr([this](double input) { return this->m_ActivationFunction->Function(input); });
	}
}

void FeedForwardLayer::Back(Eigen::VectorXd errors)
{
	this->m_Errors = errors;
	Eigen::VectorXd derivedOutputs = m_Outputs.unaryExpr([this](double input) { return this->m_ActivationFunction->DerivedFunction(input); });
	this->m_WeightDeltas = (this->m_Errors.cwiseProduct(derivedOutputs) * this->m_LearningRate) * this->m_Inputs.transpose();

	Eigen::MatrixXd weightsWitoutBiasTransposed = m_Weights.leftCols(this->m_NumInputs).transpose();
	this->m_PreviousErrors = weightsWitoutBiasTransposed * m_Errors;
}

void FeedForwardLayer::ApplyWeightDeltas()
{
	this->m_Weights += this->m_WeightDeltas;
}

void FeedForwardLayer::ComputeOuputLayerErrors(Eigen::VectorXd targets)
{
	this->m_Errors = targets - this->m_Activations;
}

void FeedForwardLayer::AddBiasToInput(Eigen::VectorXd* inputs)
{
	unsigned int lastIndex = inputs->rows();
	inputs->conservativeResize(lastIndex + 1);
	(*inputs)(lastIndex) = 1;
}
}