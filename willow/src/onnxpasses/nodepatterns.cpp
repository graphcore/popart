// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <memory>
#include <mutex>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns.hpp>
#include <onnxpasses/onnxtoonnx.hpp>
#include <onnxpasses/suffixer.hpp>
#include <string>
#include <popart/error.hpp>

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

bool ConvWithBias::go(const Node &node) {

  if ((node.op_type() == "Conv" || node.op_type() == "ConvTranspose") &&
      node.input_size() > 2 && node.input(2) != "") {

    const auto biasName = node.input(2);
    const auto outName  = node.output(0);

    const auto intermediateName = outName + suffixer.getAndIncr();

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

bool MultiConvWithBias::go(const Node &node) {
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
      const auto intermediateName = outputId + suffixer.getAndIncr();
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

bool Tan::go(const Node &node) {
  if (node.op_type() == "Tan") {
    const auto inName     = node.input(0);
    const auto outName    = node.output(0);
    const auto sinOutName = outName + suffixer.getAndIncr();
    unary(node, inName, sinOutName, "Sin");
    const auto cosOutName = outName + suffixer.getAndIncr();
    unary(node, inName, cosOutName, "Cos");
    binary(node, {sinOutName, cosOutName}, node.output(0), "Div");
    return true;
  }
  return false;
}

bool Asinh::go(const Node &node) {
  if (node.op_type() == "Asinh") {
    const auto asinhInName  = node.input(0);
    const auto outName      = node.output(0);
    const auto pow2OutName  = outName + suffixer.getAndIncr();
    const auto plus1OutName = outName + suffixer.getAndIncr();
    const auto sqrtOutName  = outName + suffixer.getAndIncr();
    const auto preLogName   = outName + suffixer.getAndIncr();
    powConstScalar(node, asinhInName, pow2OutName, ScalarInIndex::One, 2.0);
    addConstScalar(node, pow2OutName, plus1OutName, ScalarInIndex::One, 1.0);
    unary(node, plus1OutName, sqrtOutName, "Sqrt");
    binary(node, {asinhInName, sqrtOutName}, preLogName, "Add");
    unary(node, preLogName, outName, "Log");
    return true;
  }
  return false;
}

bool Acosh::go(const Node &node) {
  if (node.op_type() == "Acosh") {
    const auto asinhInName   = node.input(0);
    const auto outName       = node.output(0);
    const auto pow2OutName   = outName + suffixer.getAndIncr();
    const auto minus1OutName = outName + suffixer.getAndIncr();
    const auto sqrtOutName   = outName + suffixer.getAndIncr();
    const auto sumOutName    = outName + suffixer.getAndIncr();
    powConstScalar(node, asinhInName, pow2OutName, ScalarInIndex::One, 2.0);
    subConstScalar(node, pow2OutName, minus1OutName, ScalarInIndex::One, 1.0);
    unary(node, minus1OutName, sqrtOutName, "Sqrt");
    binary(node, {asinhInName, sqrtOutName}, sumOutName, "Add");
    unary(node, sumOutName, outName, "Log");
    return true;
  }
  return false;
}

bool Acos::go(const Node &node) {
  if (node.op_type() == "Acos") {
    const auto acosInName = node.input(0);
    const auto outName    = node.output(0);
    const auto asinOut    = outName + suffixer.getAndIncr();
    const auto diffOut    = outName + suffixer.getAndIncr();
    unary(node, acosInName, asinOut, "Asin");
    constexpr double piBy2 = 3.14159265358979323846 / 2.0;
    subConstScalar(node, asinOut, outName, ScalarInIndex::Zero, piBy2);
    return true;
  }
  return false;
}

bool Atanh::go(const Node &node) {
  if (node.op_type() == "Atanh") {
    const auto atanhInName  = node.input(0);
    const auto outName      = node.output(0);
    const auto onePlus      = outName + suffixer.getAndIncr();
    const auto oneMinus     = outName + suffixer.getAndIncr();
    const auto ratioName    = outName + suffixer.getAndIncr();
    const auto logRatioName = outName + suffixer.getAndIncr();
    addConstScalar(node, atanhInName, onePlus, ScalarInIndex::Zero, 1.0);
    subConstScalar(node, atanhInName, oneMinus, ScalarInIndex::Zero, 1.0);
    binary(node, {onePlus, oneMinus}, ratioName, "Div");
    unary(node, ratioName, logRatioName, "Log");
    mulConstScalar(node, logRatioName, outName, ScalarInIndex::Zero, 0.5);
    return true;
  }
  return false;
}
} // namespace onnxpasses
} // namespace popart
