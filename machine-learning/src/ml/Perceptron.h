#pragma once
#include <Eigen/Dense>
#include "ActivationFunction.h"

namespace ml {
class Perceptron {
private:
	int m_NumInputs;
	float m_LearningRate;
	Eigen::MatrixXd m_Weights;
	Eigen::VectorXd m_InputsWithBias;
	double m_Output;
	double m_Activation;

	double m_Error;

	ActivationFuction* m_ActivationFunction;
public:
	Perceptron(int numInputs, float learningRate = 0.1, ActivationFuction* activationFunction = new SigmoidActiavtionFuction);

	inline double GetOutput() { return m_Output; }
	inline double GetActivation() { return m_Activation; }
	inline double GetError() { return m_Error; }
	inline Eigen::MatrixXd GetWeights() { return m_Weights; }

	void FeedForward(Eigen::VectorXd inputs);
	void TrainOneExample(Eigen::VectorXd inputs, double target);
private:
	static void AddBiasInput(Eigen::VectorXd* inputs);
	Eigen::MatrixXd CalculateDeltas();
};
}