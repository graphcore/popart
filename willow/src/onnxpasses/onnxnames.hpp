// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_ONNXNAMES_HPP
#define GUARD_NEURALNET_ONNXTOONNX_ONNXNAMES_HPP

#include <cstddef>
#include <poprithmshosttensor.hpp>

namespace ONNX_NAMESPACE {
class GraphProto;
class NodeProto;
} // namespace ONNX_NAMESPACE

namespace popart {
namespace onnxpasses {
using GraphProto = ONNX_NAMESPACE::GraphProto;
using NodeProto  = ONNX_NAMESPACE::NodeProto;
using Constants  = std::map<std::string, poprithms::compute::host::Tensor>;
} // namespace onnxpasses
} // namespace popart

#endif
