#include <poponnx/error.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/graphtransformer_impl.hpp>

namespace poponnx {

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

} // namespace poponnx
