#include <vector>

#include "Layer.h"
#include "ActivationFunction.h"

namespace ml {
class Network {
private:
	int m_NumLayers;
	std::vector<Layer*> m_Layers;
public:
	Network();

	inline int GetNumLayers() { return m_NumLayers; }
	inline Eigen::VectorXd GetLastLayerActivations() { return this->m_Layers[m_NumLayers - 1]->GetActivations(); }

	void AddLayer(Layer* layer);

	void Forward(Eigen::VectorXd inputs);
	void Back(Eigen::VectorXd targets);
};
}