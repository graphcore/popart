// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

void GraphTransformer::convertUINT8ToINT32() { impl->convertUINT8ToINT32(); }

void GraphTransformer::convertUINT16ToINT32() { impl->convertUINT16ToINT32(); }

void GraphTransformer::convertINT8ToINT32() { impl->convertINT8ToINT32(); }

void GraphTransformer::convertINT16ToINT32() { impl->convertINT16ToINT32(); }

void GraphTransformer::convertINT64ToINT32() { impl->convertINT64ToINT32(); }

void GraphTransformer::convertDoublesToFloats() {
  impl->convertDoublesToFloats();
}

void GraphTransformer::convertDoublesToHalfs() {
  impl->convertDoublesToHalfs();
}

void GraphTransformer::convertBFloats16ToFloat32() {
  impl->convertBFloats16ToFloat32();
}

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

void GraphTransformer::saveInitializersExternally(
    const std::vector<TensorId> &ids,
    const std::string &fn) {
  impl->saveInitializersExternally(ids, fn);
}

} // namespace popart
