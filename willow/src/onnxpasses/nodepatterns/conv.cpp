// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/conv.hpp>
#include <onnxpasses/suffixer.hpp>
#include <string>
#include <popart/error.hpp>

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;
bool ConvWithBias::go(const NodeProto &node) {

  // Nodes with more than 2 inputs: they have a bias, which will be removed from
  // the Node.
  if ((node.op_type() == "Conv" || node.op_type() == "ConvTranspose") &&
      node.input_size() > 2 && node.input(2) != "") {

    const auto biasName = node.input(2);
    const auto outName  = node.output(0);

    const auto intermediateName = withUniqueSuffix(outName);

    // Conv without bias
    auto &n = copy(node);
    setIO(n, {node.input(0), node.input(1)}, {intermediateName});

    // Add bias
    auto &add = copyUnderscorePrefixedAttributes(node);
    setIO(add, {intermediateName, biasName}, {outName});
    add.set_op_type("AddBias");
    add.set_domain("ai.graphcore");
    return true;
  }
  return false;
}

bool MultiConvWithBias::go(const NodeProto &node) {
  if (node.op_type() != "MultiConv") {
    return false;
  }

  if (node.input_size() != 3 * node.output_size()) {
    throw internal_error(
        "Wrong number of inputs for MultiConv op. Expected "
        "number of inputs to equal three times the number of outputs. "
        "MultiConv op has {} inputs and {} outputs.",
        node.input_size(),
        node.output_size());
  }

  // Build up correct inputs/outputs for an unbiased MultiConv
  auto &unbiasedMultiConv = copy(node);
  int numConvs            = node.input_size() / 3;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  for (int i = 0; i < numConvs; i++) {
    int inputIdx         = i * 3;
    const auto &dataId   = node.input(inputIdx);
    const auto &weightId = node.input(inputIdx + 1);
    const auto &biasId   = node.input(inputIdx + 2);
    const auto &outputId = node.output(i);

    inputs.emplace_back(dataId);
    inputs.emplace_back(weightId);

    if (biasId.empty()) {
      // No bias provided
      outputs.emplace_back(outputId);
    } else {
      // Add bias
      const auto intermediateName = withUniqueSuffix(outputId);
      outputs.emplace_back(intermediateName);

      auto &addBias = copyUnderscorePrefixedAttributes(node);
      setIO(addBias, {intermediateName, biasId}, {outputId});
      addBias.set_op_type("AddBias");
      addBias.set_domain("ai.graphcore");
    }
  }

  // Setup inputs/outputs for the unbiased MultiConv
  setIO(unbiasedMultiConv, inputs, outputs);
  return true;
}

} // namespace onnxpasses
} // namespace popart
