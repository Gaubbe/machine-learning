#pragma once
#include <memory>
#include <vector>

#include "../layer/Layer.h"
#include "../layer/LayerInfo.h"
#include "Network.h"

namespace ml {
class FeedForwardNetworkBuilder {
private:
	std::vector<std::shared_ptr<Layer>> m_Layers;
public:
	static FeedForwardNetworkBuilder CreateNetwork();
	FeedForwardNetworkBuilder& AddLayer(std::shared_ptr<Layer> layer);
	FeedForwardNetworkBuilder& AddLayer(LayerInfo layer);
	Network Build();
};
}