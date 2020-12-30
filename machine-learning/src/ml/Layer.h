#pragma once
#include <Eigen/Dense>
#include "ActivationFunction.h"

namespace ml {
class Layer {
protected:
	int m_NumInputs;
	int m_NumOutputs;
	Eigen::VectorXd m_Inputs;
	Eigen::VectorXd m_Activations;
	ActivationFuction* m_ActivationFunction;
	double m_LearningRate;

public:
	Layer(int numInputs, int numOutputs, ActivationFuction* activationFunction = new SigmoidActiavtionFuction, double learningRate = 0.1);

	inline unsigned int GetNumInputs() { return m_NumInputs; }
	inline unsigned int GetNumOutputs() { return m_NumOutputs; }
	inline Eigen::VectorXd GetInputs() { return m_Inputs; }
	inline Eigen::VectorXd GetActivations() { return m_Activations; }
	inline ActivationFuction* GetActivationFunction() { return m_ActivationFunction; }

	virtual void Forward(Eigen::VectorXd inputs) = 0;
	virtual void Back(Eigen::VectorXd errors) = 0;
};
}