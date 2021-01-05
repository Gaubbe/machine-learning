#include "FeedForwardNetworkBuilder.h"
#include "../layer/FeedForwardLayerBuilder.h"

namespace ml {
FeedForwardNetworkBuilder FeedForwardNetworkBuilder::CreateNetwork()
{
	return FeedForwardNetworkBuilder();
}

FeedForwardNetworkBuilder& FeedForwardNetworkBuilder::AddLayer(std::shared_ptr<Layer> layer)
{
	this->m_Layers.push_back(layer);
	return *this;
}

FeedForwardNetworkBuilder& FeedForwardNetworkBuilder::AddLayer(LayerInfo layer)
{
	return this->AddLayer(FeedForwardLayerBuilder::CreateLayer().SetLayerInfo(layer).Build());
}

Network FeedForwardNetworkBuilder::Build()
{
	return Network(this->m_Layers);
}
}
