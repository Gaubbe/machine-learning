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
	Eigen::VectorXd m_PreviousLayerErrors;
	ActiavtionFunction* m_ActivationFunction;
	int m_IterationsSinceLastBatch;
	double m_LearningRate;

public:
	Layer(int numInputs, int numOutputs, ActiavtionFunction* activationFunction = new SigmoidActiavtionFunction, double learningRate = 0.1);

	inline unsigned int GetNumInputs() { return m_NumInputs; }
	inline unsigned int GetNumOutputs() { return m_NumOutputs; }
	inline Eigen::VectorXd GetInputs() { return m_Inputs; }
	inline Eigen::VectorXd GetPreviousLayerErrors() { return m_PreviousLayerErrors; }
	inline Eigen::VectorXd GetActivations() { return m_Activations; }
	inline ActiavtionFunction* GetActivationFunction() { return m_ActivationFunction; }

	virtual void Forward(Eigen::VectorXd inputs) = 0;
	virtual void Back(Eigen::VectorXd errors) = 0;
	virtual void ApplyBatch() = 0;
};
}