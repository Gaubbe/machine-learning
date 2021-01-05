#include <vector>

#include "../layer/Layer.h"
#include "../ActivationFunction.h"

namespace ml {
class Network {
private:
	std::vector<std::shared_ptr<Layer>> m_Layers;
public:
	Network(std::vector<std::shared_ptr<Layer>> layers);

	inline int GetNumLayers() { return m_Layers.size(); }
	inline Eigen::VectorXd GetLastLayerActivations() { return this->m_Layers[m_Layers.size() - 1]->GetActivations(); }
	inline std::vector<std::shared_ptr<Layer>> GetLayers() { return m_Layers; }

	void Forward(Eigen::VectorXd inputs);
	void Back(Eigen::VectorXd targets);
	void ApplyBatch();
};
}