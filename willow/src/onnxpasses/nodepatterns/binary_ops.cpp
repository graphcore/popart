// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "binary_ops.hpp"

namespace popart {
namespace onnxpasses {
bool Remainder::go(const NodeProto &node) {
  if (node.op_type() == "Remainder") {
    const auto dividend         = node.input(0);
    const auto divisor          = node.input(1);
    const auto outName          = node.output(0);
    const auto innerFmodOutName = withUniqueSuffix(outName);
    const auto addOutName       = withUniqueSuffix(outName);
    binary(node, {dividend, divisor}, innerFmodOutName, "Fmod");
    binary(node, {divisor, innerFmodOutName}, addOutName, "Add")
        .set_domain("ai.onnx");
    binary(node, {addOutName, divisor}, outName, "Fmod");
    return true;
  }
  return false;
}
} // namespace onnxpasses
} // namespace popart