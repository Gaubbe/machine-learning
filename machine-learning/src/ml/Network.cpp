#include "Network.h"
namespace ml {
Network::Network()
	:m_NumLayers(0)
{
}

void Network::AddLayer(Layer* layer)
{
	this->m_Layers.push_back(layer);
	this->m_NumLayers++;
}

void Network::Forward(Eigen::VectorXd inputs)
{
	Eigen::VectorXd layerInput = inputs;
	for (int i = 0; i < this->m_NumLayers; i++) {
		this->m_Layers[i]->Forward(layerInput);
		layerInput = this->m_Layers[i]->GetActivations();
	}
}

void Network::Back(Eigen::VectorXd targets)
{
	Eigen::VectorXd errors = targets - this->m_Layers[this->m_NumLayers - 1]->GetActivations();
	for (int i = m_NumLayers - 1; i >= 0; i--) {
		this->m_Layers[i]->Back(errors);
		errors = this->m_Layers[i]->GetPreviousLayerErrors();
	}
}
void Network::ApplyBatch()
{
	for (int i = 0; i < m_NumLayers; i++)
		this->m_Layers[i]->ApplyBatch();
}
}