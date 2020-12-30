#pragma once
#include "Layer.h"

namespace ml {
class FeedForwardLayer : public Layer {
protected:
	Eigen::MatrixXd m_Weights;
	Eigen::MatrixXd m_WeightDeltas;
	Eigen::VectorXd m_Outputs;
	Eigen::VectorXd m_Errors;
	Eigen::VectorXd m_PreviousErrors;

public:
	FeedForwardLayer(int numInputs, int numOutputs, ActivationFuction* activationfunction = new SigmoidActiavtionFuction, double learningRate = 0.1);

	inline Eigen::MatrixXd GetWeights() { return m_Weights; }
	inline Eigen::MatrixXd GetWeightDeltas() { return m_WeightDeltas; }
	inline Eigen::VectorXd GetOutputs() { return m_Outputs; }
	inline Eigen::MatrixXd GetErrors() { return m_Errors; }
	inline Eigen::MatrixXd GetPreviousErrors() { return m_PreviousErrors; }

	void Forward(Eigen::VectorXd inputs) override;
	void Back(Eigen::VectorXd errors) override;

	void ApplyWeightDeltas();
	void ComputeOuputLayerErrors(Eigen::VectorXd targets);

private:
	static void AddBiasToInput(Eigen::VectorXd* inputs);
};
}