// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_ONNXNAMES_HPP
#define GUARD_NEURALNET_ONNXTOONNX_ONNXNAMES_HPP

#include <cstddef>

namespace ONNX_NAMESPACE {
class GraphProto;
class NodeProto;
} // namespace ONNX_NAMESPACE

namespace popart {
namespace onnxpasses {
using GraphProto = ONNX_NAMESPACE::GraphProto;
using NodeProto  = ONNX_NAMESPACE::NodeProto;
} // namespace onnxpasses
} // namespace popart

#endif
