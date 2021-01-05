#pragma once
#include <Eigen/Dense>
#include <memory>
#include "../ActivationFunction.h"

namespace ml {
class Layer {
protected:
	int m_NumInputs;
	int m_NumOutputs;

	Eigen::VectorXd m_Inputs;
	Eigen::VectorXd m_Activations;
	Eigen::VectorXd m_PreviousLayerErrors;

	std::shared_ptr<ActivationFunction> m_ActivationFunction;
	int m_IterationsSinceLastBatch;
	double m_LearningRate;

public:
	Layer(int numInputs, int numOutputs, std::shared_ptr<ActivationFunction> activationFunction, double learningRate);

	inline unsigned int GetNumInputs() { return m_NumInputs; }
	inline unsigned int GetNumOutputs() { return m_NumOutputs; }
	inline Eigen::VectorXd GetInputs() { return m_Inputs; }
	inline Eigen::VectorXd GetPreviousLayerErrors() { return m_PreviousLayerErrors; }
	inline Eigen::VectorXd GetActivations() { return m_Activations; }
	inline std::shared_ptr<ActivationFunction> GetActivationFunction() { return m_ActivationFunction; }

	virtual void Forward(Eigen::VectorXd inputs) = 0;
	virtual void Back(Eigen::VectorXd errors) = 0;
	virtual void ApplyBatch() = 0;
};
}