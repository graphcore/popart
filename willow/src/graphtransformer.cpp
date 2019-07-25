#include <popart/error.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/graphtransformer_impl.hpp>

namespace popart {

GraphTransformer::GraphTransformer(const std::string &modelProtoOrFilename)
    : impl(new GraphTransformerImpl(modelProtoOrFilename)) {}

GraphTransformer::~GraphTransformer() {}

std::string GraphTransformer::getModelProto() const {
  return impl->getModelProto();
}

void GraphTransformer::convertFloatsToHalfs() { impl->convertFloatsToHalfs(); }

void GraphTransformer::convertInitializersToConstants(
    const std::vector<TensorId> &ids) {
  impl->convertInitializersToConstants(ids);
}

void GraphTransformer::convertAllFixedPointInitializersToConstants() {
  impl->convertAllFixedPointInitializersToConstants();
}

void GraphTransformer::prepareNodesForTraining() {
  impl->prepareNodesForTraining();
}

void GraphTransformer::removeUnusedInputs() { impl->removeUnusedInputs(); }

} // namespace popart
