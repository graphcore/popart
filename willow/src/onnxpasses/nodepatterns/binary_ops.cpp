// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "binary_ops.hpp"

namespace popart {
namespace onnxpasses {
bool Remainder::go(const NodeProto &node) {
  if (node.op_type() == "Remainder") {
    remainderToFmod(node);
    return true;
  }
  return false;
}

void Remainder::remainderToFmod(const NodeProto &node) {
  const auto dividend         = node.input(0);
  const auto divisor          = node.input(1);
  const auto outName          = node.output(0);
  const auto innerFmodOutName = withUniqueSuffix(outName);
  const auto addOutName       = withUniqueSuffix(outName);
  binary(node, {dividend, divisor}, innerFmodOutName, "Fmod")
      .set_domain("ai.graphcore");
  binary(node, {divisor, innerFmodOutName}, addOutName, "Add")
      .set_domain("ai.onnx");
  binary(node, {addOutName, divisor}, outName, "Fmod")
      .set_domain("ai.graphcore");
}

bool Mod::go(const NodeProto &node) {
  if (node.op_type() == "Mod") {
    const auto &attributes = node.attribute();
    const bool isFmod      = std::any_of(
        attributes.cbegin(), attributes.cend(), [](const auto &attr) {
          return attr.name() == std::string{"fmod"} && attr.i() == 1;
        });
    if (isFmod)
      binary(node, {node.input(0), node.input(1)}, node.output(0), "Fmod")
          .set_domain("ai.graphcore");
    else
      remainderToFmod(node);
    return true;
  }
  return false;
}
} // namespace onnxpasses
} // namespace popart