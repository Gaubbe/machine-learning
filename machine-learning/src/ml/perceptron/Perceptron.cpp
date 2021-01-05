#include "Perceptron.h"
#include <iostream>

namespace ml {
Perceptron::Perceptron(int numInputs, float learningRate, ActivationFunction* activationFunction)
	: m_NumInputs(numInputs), m_LearningRate(learningRate), m_ActivationFunction(activationFunction)
{
	this->m_Weights = Eigen::MatrixXd::Random(1, this->m_NumInputs + 1);
}

void Perceptron::FeedForward(Eigen::VectorXd inputs)
{
	this->AddBiasInput(&inputs);
	m_InputsWithBias = inputs;
	this->m_Output = (this->m_Weights * inputs)(0, 0);
	this->m_Activation = this->m_ActivationFunction->Function(this->m_Output);
}

void Perceptron::TrainOneExample(Eigen::VectorXd inputs, double target)
{
	this->FeedForward(inputs);
	m_Error = target - m_Output;
	Eigen::MatrixXd deltas = this->CalculateDeltas();
	//std::cout << m_Weights << std::endl;
	//std::cout << deltas << std::endl;
	m_Weights += deltas;
	//std::cout << m_Weights << std::endl << std::endl;
}

void Perceptron::AddBiasInput(Eigen::VectorXd* inputs)
{
	inputs->conservativeResize(inputs->rows() + 1);
	(*inputs)(inputs->rows() - 1) = 1;
}
Eigen::MatrixXd Perceptron::CalculateDeltas()
{
	return (m_InputsWithBias * (m_Error * this->m_ActivationFunction->DerivedFunction(m_Output) * m_LearningRate)).transpose();
}
}