#pragma once
#include "Layer.h"

namespace ml {
class FeedForwardLayer : public Layer {
protected:
	Eigen::MatrixXd m_Weights;
	Eigen::MatrixXd m_WeightDeltas;
	Eigen::VectorXd m_Outputs;

public:
	FeedForwardLayer(int numInputs, int numOutputs, ActiavtionFunction* activationfunction = new SigmoidActiavtionFunction, double learningRate = 0.1);

	inline Eigen::MatrixXd GetWeights() { return m_Weights; }
	inline Eigen::MatrixXd GetWeightDeltas() { return m_WeightDeltas; }
	inline Eigen::VectorXd GetOutputs() { return m_Outputs; }

	void Forward(Eigen::VectorXd inputs) override;
	void Back(Eigen::VectorXd errors) override;

	void ApplyBatch() override;

private:
	static void AddBiasToInput(Eigen::VectorXd* inputs);
};
}