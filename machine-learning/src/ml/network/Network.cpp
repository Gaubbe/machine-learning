#include "Network.h"
namespace ml {
Network::Network(std::vector<std::shared_ptr<Layer>> layers)
	:m_Layers(layers)
{
}

void Network::Forward(Eigen::VectorXd inputs)
{
	Eigen::VectorXd layerInput = inputs;
	for (int i = 0; i < this->m_Layers.size(); i++) {
		this->m_Layers[i]->Forward(layerInput);
		layerInput = this->m_Layers[i]->GetActivations();
	}
}

void Network::Back(Eigen::VectorXd targets)
{
	Eigen::VectorXd errors = targets - this->m_Layers[this->m_Layers.size() - 1]->GetActivations();
	for (int i = m_Layers.size() - 1; i >= 0; i--) {
		this->m_Layers[i]->Back(errors);
		errors = this->m_Layers[i]->GetPreviousLayerErrors();
	}
}
void Network::ApplyBatch()
{
	for (int i = 0; i < m_Layers.size(); i++)
		this->m_Layers[i]->ApplyBatch();
}
}