// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_ONNXPASSES_ONNXNAMES_HPP_
#define POPART_WILLOW_SRC_ONNXPASSES_ONNXNAMES_HPP_

#include <map>
#include <string>
#include <poprithms/compute/host/tensor.hpp>

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

#endif // POPART_WILLOW_SRC_ONNXPASSES_ONNXNAMES_HPP_
